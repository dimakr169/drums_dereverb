import os, time, math, random, argparse
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

from dataset.stereo_dataset import build_dataloaders
from configs.io import load_config, config_to_dict

from backbones.unet_stereo import UNetRI
from backbones.dit_stereo import TransformerDiffuser, reinit_projections_orthonormal
from backbones.metrics_torch import SISDR, SISDRi

# ---- Helpers ----
def set_runtime_flags(runtime_cfg):
    torch.backends.cuda.matmul.allow_tf32 = bool(getattr(runtime_cfg, "allow_tf32", False))
    torch.set_float32_matmul_precision(getattr(runtime_cfg, "matmul_precision", "high"))
    torch.backends.cudnn.benchmark = bool(getattr(runtime_cfg, "cudnn_benchmark", True))


def build_model(model_cfg):
    if model_cfg.backbone == "unet":
        return UNetRI(model_cfg)

    if model_cfg.backbone == "dit":
        model = TransformerDiffuser(model_cfg)
        if getattr(model_cfg, "reinit_projection", False):
            reinit_projections_orthonormal(model)
        return model

    raise ValueError(f"Unsupported backbone: {model_cfg.backbone}")


def save_resolved_config(cfg, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cfg_path = os.path.join(output_dir, "resolved_config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False, allow_unicode=True)

# ---- EMA wrapper ----
class EMAModel:
    def __init__(self, model: nn.Module, decay: float=0.999):
        self.decay = decay
        self.model = model
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.backup = None

    @torch.no_grad()
    def _sync_new_params(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and (n not in self.shadow):
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self):
        self._sync_new_params()
        for n, p in self.model.named_parameters():
            if not p.requires_grad: 
                continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self):
        self._sync_new_params()
        self.backup = {}
        for n, p in self.model.named_parameters():
            if not p.requires_grad: 
                continue
            self.backup[n] = p.detach().clone()
            p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self):
        if self.backup is None: 
            return
        for n, p in self.model.named_parameters():
            if not p.requires_grad: 
                continue
            p.data.copy_(self.backup[n])
        self.backup = None

    # --- add these two for safe checkpointing ---
    def state_dict(self):
        # clone tensors to detach from graph
        return {n: t.clone() for n, t in self.shadow.items()}

    def load_state_dict(self, state, strict: bool = False):
        """
        Supports both dict (new) and list (legacy) formats.
        If a list is given, it will be matched in the order of named_parameters()
        filtered by requires_grad.
        """
        if isinstance(state, list):
            # legacy: map list -> current trainable params in order
            i = 0
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if i < len(state):
                    self.shadow[n] = state[i].detach().clone()
                    i += 1
                elif strict:
                    raise RuntimeError("EMA list shorter than model params.")
            return

        # new: dict
        missing = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if n in state:
                self.shadow[n] = state[n].detach().clone()
            elif strict:
                missing.append(n)
        if strict and missing:
            raise RuntimeError(f"EMA missing keys: {missing}")
        

# ---- ISTFT helper: (B,4,F,T)->(B,2,T) ----
def istft_from_ri(ri, n_fft, hop, win_length, window, center: bool, length: int | None):
    # ri: (B,4,F,T) [L_R, L_I, R_R, R_I]
    # Make (i)STFT run in fp32 regardless of outer autocast
    with torch.cuda.amp.autocast(enabled=False):
        # ensure fp32 components before building complex
        L_real = ri[:, 0].float()
        L_imag = ri[:, 1].float()
        R_real = ri[:, 2].float()
        R_imag = ri[:, 3].float()

        L = torch.complex(L_real, L_imag)  # complex64
        R = torch.complex(R_real, R_imag)

        recL = torch.istft(L, n_fft=int(n_fft), hop_length=int(hop),
                           win_length=int(win_length), window=window.float(),
                           center=center, length=length)
        recR = torch.istft(R, n_fft=int(n_fft), hop_length=int(hop),
                           win_length=int(win_length), window=window.float(),
                           center=center, length=length)
        out = torch.stack([recL, recR], dim=1)  # (B,2,T)
    return out  # keep as fp32 (good for losses)


# ---- alpha schedule: cos^2 by default ----
def make_alpha_bar(diffusion_steps: int, device, kind="poly", power=3.0, beta=5.0, k=8.0):
    """
    Returns alpha_bar[0..T] with alpha_bar[0]=1 (clean), alpha_bar[T]=0 (reverb).
    kind: "poly" (default), "cos2", "exp", "sigmoid"
    - poly:    alpha = 1 - (t/T)^power              # p>=2 gives steep early, gentle late
    - cos2:    alpha = cos^2(0.5*pi*t/T)            # gentle to all
    - exp:     alpha = 1 - exp(-beta*(1 - t/T))     # beta≈3–8, similar shape to poly
    - sigmoid: alpha = sigmoid(k*(1 - 2*t/T))       # S-shaped; pick k≈6–10
    """
    T = diffusion_steps
    t = torch.arange(T+1, device=device, dtype=torch.float32)  # 0..T
    x = t / float(T)

    if kind == "poly":
        a = 1.0 - x.pow(power)
    elif kind == "cos2":
        a = torch.cos(0.5 * math.pi * x).pow(2)
    elif kind == "exp":
        a = 1.0 - torch.exp(-beta * (1.0 - x))
    elif kind == "sigmoid":
        a = torch.sigmoid(k * (1.0 - 2.0 * x))
    else:
        raise ValueError(f"Unknown schedule: {kind}")

    # Ensure exact endpoints
    a[0] = 1.0  # clean
    a[-1] = 0.0 # reverb
    return a


# ---- LR policies ----
def build_scheduler(optimizer, policy: str, base_lr: float, steps_per_epoch: int,
                    epochs: int, restart_epochs: int=0, warmup_epochs: int=0,
                    warmup_initial_lr: float=1e-6, cosine_floor_factor: float=1e-6):
    """
    Build learning rate scheduler.
    Supported:
      - fixed
      - cosine_restart
      - warmup_cosine
      - two_phase_cosine (constant LR for first warm_epochs, then cosine to floor)
    """
    if policy == "fixed":
        return None  # no scheduler; keep constant LR

    total_steps = steps_per_epoch * epochs

    # --------------------------------------------------
    # Cosine restart
    # --------------------------------------------------
    if policy == "cosine_restart":
        first_decay = max(steps_per_epoch * max(1, restart_epochs), 1)
        sched = CosineAnnealingWarmRestarts(
            optimizer, T_0=first_decay, T_mult=1, eta_min=base_lr * cosine_floor_factor
        )
        return sched

    # --------------------------------------------------
    # Warmup + cosine 
    # --------------------------------------------------
    if policy == "warmup_cosine":
        warmup_steps = int(steps_per_epoch * warmup_epochs)
        cosine_len = max(total_steps - warmup_steps, 1)
        def lr_lambda(step):
            if step < warmup_steps:
                # linear warmup from warmup_initial_lr -> base_lr
                return max(warmup_initial_lr, 1e-12) / base_lr + (
                    step / max(1, warmup_steps)
                ) * (1.0 - max(warmup_initial_lr, 1e-12) / base_lr)
            # cosine decay to floor factor
            k = (step - warmup_steps) / cosine_len
            return cosine_floor_factor + 0.5 * (1 - cosine_floor_factor) * (
                1 + math.cos(math.pi * k)
            )
        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    # --------------------------------------------------
    # Two-phase cosine (constant early, cosine later)
    # --------------------------------------------------
    if policy == "two_phase_cosine":
        warm_epochs = max(1, warmup_epochs or 2)  # constant LR for first 2 epochs by default
        floor = cosine_floor_factor
        def lr_lambda(global_step):
            epoch = global_step / max(1, steps_per_epoch)
            if epoch < warm_epochs:
                # constant LR (base)
                return 1.0
            # cosine from epoch warm_epochs -> epochs
            progress = (epoch - warm_epochs) / max(1e-8, (epochs - warm_epochs))
            return floor + 0.5 * (1 - floor) * (1 + math.cos(math.pi * progress))
        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    # --------------------------------------------------
    raise ValueError(f"Unknown lr_policy: {policy}")

# ---- Trainer clean version ----
class ColdRITrainer: 
    def __init__(self, model, cfg, dataloaders, output_dir, device="cuda"):
        self.model = model.to(device)
        self.cfg = cfg
        self.pre_params = cfg.data
        self.train_params = cfg.train
        self.model_params = cfg.model
        self.runtime_params = cfg.runtime

        self.train_loader, self.val_loader = dataloaders
        self.output_dir = output_dir
        self.device = device

        # paths
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "samples"), exist_ok=True)

        # diffusion params
        self.backbone = self.model_params.backbone
        self.diffusion_steps = self.train_params.diffusion_steps
        self.diffusion_mode = self.train_params.diffusion_mode
        self.alpha_mode = self.train_params.alpha_mode
        self.residual_mode = self.train_params.residual_mode
        self.alpha_bar = make_alpha_bar(
            self.diffusion_steps, device=self.device, kind=self.alpha_mode
        )

        # AMP
        self.use_amp = bool(getattr(self.runtime_params, "amp", True)) and str(device).startswith("cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        opt_name = self.train_params.optimizer.name.lower()
        fused = bool(getattr(self.train_params.optimizer, "fused", False)) and str(device).startswith("cuda")

        # optimizer
        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=float(self.train_params.learning_rate),
                betas=(self.train_params.beta1, self.train_params.beta2),
                eps=self.train_params.eps,
                weight_decay=float(getattr(self.train_params, "weight_decay", 0.0)),
            )
        elif opt_name == "adamw":
            adamw_kwargs = dict(
                params=self.model.parameters(),
                lr=float(self.train_params.learning_rate),
                betas=(self.train_params.beta1, self.train_params.beta2),
                eps=self.train_params.eps,
                weight_decay=float(getattr(self.train_params, "weight_decay", 0.0)),
            )
            if fused:
                adamw_kwargs["fused"] = True
            self.optimizer = torch.optim.AdamW(**adamw_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")


        # EMA
        self.ema = EMAModel(self.model, decay=self.train_params.ema_decay)

        # losses & metrics
        self.l1 = nn.L1Loss()
        self.sisdr  = SISDR("si_sdr")
        self.sisdri = SISDRi("si_sdri")

        # TB writers
        log_root = os.path.join(self.output_dir, "logs")
        self.tb_train = SummaryWriter(os.path.join(log_root, "train"))
        self.tb_val   = SummaryWriter(os.path.join(log_root, "validation"))

        # ckpt path
        self.ckpt_path = os.path.join(self.output_dir, "checkpoints", "latest.pt")

        # window for ISTFT
        self.center = self.pre_params.center
        self.window = torch.hann_window(self.pre_params.win, periodic=True, device=self.device)

        # LR scheduler (per-step)
        steps_per_epoch = len(self.train_loader)
        self.scheduler = build_scheduler(
            self.optimizer,
            policy=self.train_params.lr_policy,
            base_lr=float(self.train_params.learning_rate),
            steps_per_epoch=steps_per_epoch,
            epochs=self.train_params.epochs,
            restart_epochs=self.train_params.restart_epochs,
            warmup_epochs=self.train_params.warmup_epochs,
            warmup_initial_lr=self.train_params.warmup_initial_lr,
            cosine_floor_factor=self.train_params.cosine_floor_factor,
        )

        # loss weights
        self.delta_weight = float(getattr(self.train_params.loss, "delta_weight", 0.7))
        self.recon_weight = float(getattr(self.train_params.loss, "recon_weight", 0.3))
        self.audio_weight = float(getattr(self.train_params.loss, "audio_weight", 8.0))
        self.multiply_factor = float(getattr(self.train_params.loss, "multiply_factor", 50.0))

        self.grad_clip = float(getattr(self.train_params, "grad_clip", 1.0))

    def load_checkpoint(self):
        """Load model, optimizer, scaler, EMA and return (next_epoch, best_loss)."""
        print(f"Loading checkpoint from: {self.ckpt_path}")
        ckpt = torch.load(self.ckpt_path, map_location=self.device)

        # <-- NON-STRICT 
        incompatible = self.model.load_state_dict(ckpt["model"], strict=False)
        print("Loaded model with non-strict matching.")
        print("  Missing keys:", incompatible.missing_keys)
        print("  Unexpected keys:", incompatible.unexpected_keys)

        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.ema.load_state_dict(ckpt["ema"])

        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))

        print(f"Resuming from epoch {start_epoch} with best_loss={best_loss:.4f}")
        return start_epoch, best_loss   

    @torch.no_grad()
    def diffusion(self, reverb_ri, clean_ri, noise_level):
        # noise_level a_t in [0,1], shape (B,)
        a = noise_level.view(-1, 1, 1, 1)
        if self.diffusion_mode == "linear":
            return a * clean_ri + (1.0 - a) * reverb_ri
        elif self.diffusion_mode == "sqrt_pair":
            return torch.sqrt(a) * clean_ri + torch.sqrt(1.0 - a) * reverb_ri
        elif self.diffusion_mode == "sqrt_aggresive":
            return a * clean_ri + (1.0 - torch.sqrt(a)) * reverb_ri
        else:
            raise ValueError(f"Unknown diffusion_mode {self.diffusion_mode}")

    def get_signal_from_RI_stft(self, ri_stft):
        # ri_stft: (B,4,F,T) -> (B,2,T)
        n_fft = self.pre_params.fft
        hop = self.pre_params.hop
        win = self.pre_params.win
        length = getattr(self.pre_params, "wave_len", None)  # optional; or infer externally

        window = self.window
        if window.device != ri_stft.device:
            window = window.to(ri_stft.device)

        return istft_from_ri(ri_stft, n_fft=n_fft, hop=hop, win_length=win,
                             window=window, center=self.center, length=length)

    def _random_timesteps(self, bsize):
        # Uniform integers in [1, T]
        return torch.randint(low=1, high=self.diffusion_steps+1, size=(bsize,), device=self.device)

    def _levels_for(self, t):
        # returns (alpha_t, alpha_{t-1}) as (B,)
        a_t = self.alpha_bar.index_select(0, t)
        a_tm1 = self.alpha_bar.index_select(0, t-1)
        return a_t, a_tm1
    

    def _step(self, batch, train=True, global_step=0):
        reverb_ri, clean_ri = batch  # from DataLoader: (B,4,F,T)
        reverb_ri = reverb_ri.to(self.device, non_blocking=True)
        clean_ri  = clean_ri.to(self.device, non_blocking=True)

        bsize = reverb_ri.shape[0]
        timesteps = self._random_timesteps(bsize)  # (B,)
        a_t, a_tm1 = self._levels_for(timesteps)

        noised = self.diffusion(reverb_ri, clean_ri, a_t)
        noised_next = self.diffusion(reverb_ri, clean_ri, a_tm1)

        self.model.train(train)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            if self.residual_mode == "next_delta_norm":
                # Normalized velocity v_t = (x_{t-1}-x_t) / g_t,  g_t = a_{t-1}-a_t  (linear mix only)
                g = (a_tm1 - a_t).clamp_min(1e-6).view(-1,1,1,1)       # (B,1,1,1)

                est_v   = self.model(noised, timesteps)                # v̂_t
                est_ri  = noised + g * est_v                           # x̂_{t-1}
                target_v = (noised_next - noised) / g

                delta_loss = self.l1(est_v, target_v) * self.delta_weight
                recon_loss = self.l1(est_ri, noised_next) * self.recon_weight
                noise_loss = (delta_loss + recon_loss) * self.multiply_factor 
          
            elif self.residual_mode == "direct":
                if self.backbone == "dit":
                    raise ValueError("DiT currently supports only residual_mode='next_delta_norm'.")

                est_ri  = self.model(noised, timesteps)                  # x̂_{t-1}
                noise_loss = self.l1(est_ri, noised_next) * self.multiply_factor 

            # Audio-domain MAE
            est_wav = self.get_signal_from_RI_stft(est_ri)       # (B,2,T)
            tar_wav = self.get_signal_from_RI_stft(noised_next)  # ground truth
            audio_loss = self.l1(est_wav, tar_wav) * self.audio_weight * self.multiply_factor 
            
            # Total loss
            loss = noise_loss + audio_loss

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()
            self.ema.update()
            if self.scheduler is not None:
                self.scheduler.step()

        # Also compute the original *input* waveform for SI-SDRi
        with torch.no_grad():
            inp_wav = self.get_signal_from_RI_stft(reverb_ri)  # (B,2,T), reverberant input
            clean_wav = self.get_signal_from_RI_stft(clean_ri)   # clean reference

        return {
            "loss": loss.detach(),
            "noise": noise_loss.detach(),
            "audio": audio_loss.detach(),
            "est_wav": est_wav.detach(),
            "tar_wav": tar_wav.detach(),
            "inp_wav": inp_wav.detach(),  
            "clean_wav": clean_wav.detach(),
        }

    @torch.inference_mode()
    def reverse_diffusion(self, inp_ri, step_stop=0):
        # inp_ri: (B,4,F,T)
        bsize = inp_ri.shape[0]
        xs = []
        x = inp_ri
        for t in range(self.diffusion_steps, step_stop, -1):
            T = torch.full((bsize,), t, device=self.device, dtype=torch.long)
            if self.residual_mode == "next_delta_norm":
                a_t   = self.alpha_bar.index_select(0, T)          # (B,)
                a_tm1 = self.alpha_bar.index_select(0, T-1)
                g = (a_tm1 - a_t).clamp_min(1e-6).view(-1,1,1,1)
                v = self.model(x, T)
                x = x + g * v
            elif self.residual_mode == "direct":
                x = self.model(x, T)      # direct x̂_{t-1}     
            else:
                raise ValueError(f"Unsupported residual_mode: {self.residual_mode}")         

            xs.append(x.detach().to("cpu", non_blocking=False))
        return xs  # list of (B,4,F,T)

    @torch.no_grad()
    def generate_random_batch(self, epoch):
        out_root = os.path.join(self.output_dir, "samples", f"epoch_{epoch}")
        os.makedirs(out_root, exist_ok=True)

        try:
            batch = next(iter(self.val_loader))
        except StopIteration:
            return

        reverb_ri, clean_ri = [b.to(self.device) for b in batch]
        inp_ri = reverb_ri

        # swap-in EMA weights for generation
        self.ema.apply_shadow()
        preds = self.reverse_diffusion(inp_ri)  # list of (B,4,F,T)
        self.ema.restore()

        # save first N examples per batch
        sr = getattr(self.pre_params, "sr", 44100)
        Bsave = min(8, reverb_ri.shape[0]) # max 8 files
        for i in range(Bsave):
            val_dir = os.path.join(out_root, f"val_{i}")
            os.makedirs(val_dir, exist_ok=True)
            inp_wav = self.get_signal_from_RI_stft(reverb_ri[i:i+1]).squeeze(0).permute(1,0).cpu().numpy()  # (T,2)
            tar_wav = self.get_signal_from_RI_stft(clean_ri[i:i+1]).squeeze(0).permute(1,0).cpu().numpy()
            sf.write(os.path.join(val_dir, "input.wav"),  inp_wav, sr)
            sf.write(os.path.join(val_dir, "target.wav"), tar_wav, sr)
            for t, pred in enumerate(preds):
                pred_wav = self.get_signal_from_RI_stft(pred[i:i+1]).squeeze(0).permute(1,0).cpu().numpy()
                sf.write(os.path.join(val_dir, f"diffused_{t}.wav"), pred_wav, sr)

    def train(self, start_epoch: int = 0, best_loss: float | None = None):
        train_size = len(self.train_loader)
        val_size   = len(self.val_loader)
        print(f"Dataset with {train_size} training and {val_size} validation batches")

        patience = 0
        if best_loss is None:
            best_loss = float("inf")


        # roughly continue global step counter (for TB)
        gstep = start_epoch * len(self.train_loader)

        for epoch in range(start_epoch, self.train_params.epochs):
            print(f"\nStart of epoch {epoch}")
            t0 = time.time()

            # ---- Train ----
            self.model.train(True)

            for b, batch in enumerate(self.train_loader):
                out = self._step(batch, train=True, global_step=gstep)
                if (b % 300) == 0:
                    print(f"Batch {b:5d} | Noise {out['noise'].item():.4f} "
                        f"| Audio {out['audio'].item():.4f} ")
                # TB per-step
                self.tb_train.add_scalar("loss/noise", out["noise"].item(), gstep)
                self.tb_train.add_scalar("loss/audio", out["audio"].item(), gstep)
                gstep += 1

            # ---- Validate with EMA weights ----
            self.ema.apply_shadow()
            self.model.eval()
            noise_sum = audio_sum = 0.0
            n_batches = 0
            self.sisdr.reset(); 
            self.sisdri.reset() 

            with torch.no_grad():
                for batch in self.val_loader:
                    out = self._step(batch, train=False)
                    noise_sum += out["noise"].item()
                    audio_sum += out["audio"].item()
                    n_batches += 1
                    # SI metrics (stubs)
                    self.sisdr.update(out["clean_wav"], out["est_wav"])
                    self.sisdri.update(out["clean_wav"], out["est_wav"], out["inp_wav"]) 
                    
            self.ema.restore()

            noise_avg = noise_sum / max(n_batches,1)
            audio_avg = audio_sum / max(n_batches,1)
            val_loss  = noise_avg + audio_avg 

            # TB per-epoch
            self.tb_val.add_scalar("loss/noise", noise_avg, epoch)
            self.tb_val.add_scalar("loss/audio", audio_avg, epoch)
            self.tb_val.add_scalar("metrics/si_sdr", self.sisdr.result(), epoch)
            self.tb_val.add_scalar("metrics/si_sdri", self.sisdri.result(), epoch)


            print("----")
            print(f"Total Noise MAE Loss {noise_avg:.4f}")
            print(f"Total Audio MAE     {audio_avg:.4f}")
            print(f"Overall Val Loss    {val_loss:.4f}")
            print("----")
            print(f"SISDR {self.sisdr.result():.4f} | SISDRi {self.sisdri.result():.4f}")

            # early stopping + checkpoint
            if val_loss < best_loss:
                torch.save({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "ema": self.ema.state_dict(), 
                    "best_loss": val_loss,
                }, self.ckpt_path)
                print("Checkpoint saved.")
                best_loss = val_loss
                patience = 0

                if self.train_params.gen_val_batch:
                    self.generate_random_batch(epoch)
            else:
                print("No validation loss improvement.")
                patience += 1

            print(f"Time taken for this epoch: {time.time()-t0:.2f} secs")
            print("*******************************")

            if patience > self.train_params.patience:
                print("Terminating the training.")
                print("Best val loss stopped at", best_loss)
                break

def set_global_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---- YAML version ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)

    set_global_seed(cfg.experiment.seed, deterministic=False)
    set_runtime_flags(cfg.runtime)

    # choose device
    if torch.cuda.is_available():
        device = f"cuda:{cfg.runtime.gpu}"
        torch.cuda.set_device(cfg.runtime.gpu)
        print("Using GPU:", device)
    else:
        device = "cpu"
        print("No GPU, using CPU")


    # dataloaders
    train_loader, val_loader = build_dataloaders(cfg.data, cfg.runtime.data_dir)
    dataloaders = (train_loader, val_loader)

    # model
    model = build_model(cfg.model)

    # --- print parameter counts ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model Parameters: {trainable_params:,} trainable / {total_params:,} total "
        f"({trainable_params / 1e6:.2f} M params)"
    )

    # trainer
    out_dir = cfg.experiment.output_dir
    save_resolved_config(cfg, out_dir)

    trainer = ColdRITrainer(
        model=model,
        cfg=cfg,
        dataloaders=dataloaders,
        output_dir=out_dir,
        device=device,
    )

    # --- in case of resuming training ---
    start_epoch = 0
    best_loss = None

    if args.resume:
        if os.path.exists(trainer.ckpt_path):
            start_epoch, best_loss = trainer.load_checkpoint()
        else:
            print(f"WARNING: resume requested but checkpoint not found at {trainer.ckpt_path}. Starting from scratch.")

    # start training
    trainer.train(start_epoch=start_epoch, best_loss=best_loss)

if __name__ == "__main__":
    main()