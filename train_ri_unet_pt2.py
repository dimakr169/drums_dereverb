import os, time, math, random, argparse
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

# --- import your external libraries ---
from dataset.stereo_dataset import build_dataloaders  # PyTorch DataLoader
from config import Config
from backbones.unet_stereo import UNetRI # PyTorch Stereo UNet
from backbones.metrics_torch import SISDR, SISDRi, NormalizedMutualInformationLoss, NMILossConfig


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
    # Cosine restart (unchanged)
    # --------------------------------------------------
    if policy == "cosine_restart":
        first_decay = max(steps_per_epoch * max(1, restart_epochs), 1)
        sched = CosineAnnealingWarmRestarts(
            optimizer, T_0=first_decay, T_mult=1, eta_min=base_lr * cosine_floor_factor
        )
        return sched

    # --------------------------------------------------
    # Warmup + cosine (original)
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

# ---- Trainer Variable version ----
class ColdRIUNetTrainer: 
    def __init__(self, model, pre_params, train_params, dataloaders, output_dir, device="cuda"):
        self.model = model.to(device)
        self.pre_params = pre_params
        self.train_params = train_params
        self.train_loader, self.val_loader = dataloaders
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "samples"), exist_ok=True)

        self.device = device
        # cold diffusion params
        self.diffusion_steps = train_params.diffusions_steps
        self.diffusion_mode = train_params.diffusion_mode
        self.alpha_mode = train_params.alpha_mode
        self.residual_mode = train_params.residual_mode

        # optimizer Adam
        #self.optimizer = torch.optim.Adam(
        #    self.model.parameters(),
        #    lr=float(train_params.learning_rate),
        #    betas=(train_params.beta1, train_params.beta2),
        #    eps=train_params.eps,
        #)

        # optimizer AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_params.learning_rate),
            betas=(train_params.beta1, train_params.beta2),
            eps=train_params.eps,
            weight_decay = train_params.weight_decay,
            fused=True,   # PyTorch 2.9+ on CUDA
        )        

        # AMP
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))

        # EMA
        self.ema = EMAModel(self.model, decay=train_params.ema_decay)

        # losses & metrics
        self.l1 = nn.L1Loss()
        self.nmi_loss = NormalizedMutualInformationLoss(NMILossConfig(
            bins=384,
            window_length=1024,
            hop_length=384,
            use_db_scale=False,
        ))
        self.sisdr  = SISDR("si_sdr")
        self.sisdri = SISDRi("si_sdri")

        # TB writers
        log_root = os.path.join(self.output_dir, "logs")
        self.tb_train = SummaryWriter(os.path.join(log_root, "train"))
        self.tb_val   = SummaryWriter(os.path.join(log_root, "validation"))

        # ckpt path
        self.ckpt_path = os.path.join(self.output_dir, "checkpoints", "latest.pt")

        # window for ISTFT
        self.center = pre_params.center
        self.window = torch.hann_window(pre_params.win, periodic=True, device=self.device)

        # LR scheduler (per-step)
        steps_per_epoch = len(self.train_loader)
        self.scheduler = build_scheduler(
            self.optimizer,
            policy=train_params.lr_policy,
            base_lr=float(train_params.learning_rate),
            steps_per_epoch=steps_per_epoch,
            epochs=train_params.epochs,
            restart_epochs=train_params.restart_epochs, 
            warmup_epochs=train_params.warmup_epochs,
            warmup_initial_lr=train_params.warmup_initial_lr,
            cosine_floor_factor=train_params.cosine_floor_factor,
        )

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
        return istft_from_ri(ri_stft, n_fft=n_fft, hop=hop, win_length=win,
                             window=self.window, center=self.center, length=length)
    
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

    def alpha_continuous(self, t: torch.Tensor) -> torch.Tensor:
        """
        Continuous alpha(t) matching make_alpha_bar() shapes, for float t in [0, T_train].
        Returns a(t) in [0,1] with a(0)=1, a(T)=0.
        """
        T = float(self.diffusion_steps)
        x = (t / T).clamp(0.0, 1.0)

        kind = self.alpha_mode
        if kind == "poly":
            power = 3.0
            a = 1.0 - x.pow(power)
        elif kind == "cos2":
            a = torch.cos(0.5 * math.pi * x).pow(2)
        elif kind == "exp":
            beta = 5.0
            a = 1.0 - torch.exp(-beta * (1.0 - x))
        elif kind == "sigmoid":
            k = 8.0
            a = torch.sigmoid(k * (1.0 - 2.0 * x))
        else:
            raise ValueError(f"Unknown schedule: {kind}")

        return a.clamp(0.0, 1.0)


    def _random_t_and_dt(self, bsize: int,
                        dt_min: float = 0.25,
                        dt_max: float = 1.0):
        """
        Sample continuous t and a random step size dt.
        We keep the horizon [0..T_train] fixed; only the training *step size* varies.
        """
        T = float(self.diffusion_steps)
        # sample t in (0, T]
        t = torch.rand(bsize, device=self.device) * (T - 1e-3) + 1e-3  # (B,)
        # sample dt in [dt_min, dt_max]
        dt = torch.rand(bsize, device=self.device) * (dt_max - dt_min) + dt_min
        # next time (towards 0)
        t_next = (t - dt).clamp_min(0.0)
        return t, t_next    

    def _step(self, batch, train=True, global_step=0):
        reverb_ri, clean_ri = batch
        reverb_ri = reverb_ri.to(self.device, non_blocking=True)
        clean_ri  = clean_ri.to(self.device, non_blocking=True)

        bsize = reverb_ri.shape[0]

        # 1) continuous times + random dt
        t, t_next = self._random_t_and_dt(bsize, dt_min=0.25, dt_max=4.0)  # sampling 4-64 timesteps

        # 2) compute continuous alpha levels
        a_t    = self.alpha_continuous(t)       # (B,)
        a_next = self.alpha_continuous(t_next)  # (B,)

        # 3) forward diffusion at both levels (linear mix)
        x_t    = self.diffusion(reverb_ri, clean_ri, a_t)
        x_next = self.diffusion(reverb_ri, clean_ri, a_next)

        # 4) normalized update size in "alpha space"
        g = (a_next - a_t).clamp_min(1e-6).view(-1, 1, 1, 1)  # (B,1,1,1)

        self.model.train(train)
        with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
            # 5) predict velocity field v
            v_hat = self.model(x_t, t)  # NOTE: t is float (B,)

            # 6) one-step prediction for that random dt
            x_hat_next = x_t + g * v_hat

            # 7) target v is consistent for any dt:
            v_target = (x_next - x_t) / g

            # losses (keep your original spirit, just with continuous t/dt)
            delta_loss = self.l1(v_hat, v_target) * 35.0
            noise_loss = self.l1(x_hat_next, x_next) * 15.0
            noise_loss = noise_loss + delta_loss

            # Audio-domain MAE & NMI (compare to x_next, the true next point)
            est_wav = self.get_signal_from_RI_stft(x_hat_next)
            tar_wav = self.get_signal_from_RI_stft(x_next)
            audio_loss = self.l1(est_wav, tar_wav) * 400.0
            nmi_loss = self.nmi_loss(tar_wav, est_wav)

            loss = noise_loss + nmi_loss + audio_loss

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # optional
            self.scaler.step(self.optimizer)
            self.scaler.update()
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
            "nmi": nmi_loss.detach(),
            "est_wav": est_wav.detach(),
            "tar_wav": tar_wav.detach(),
            "inp_wav": inp_wav.detach(),  
            "clean_wav": clean_wav.detach(),
        }

    @torch.no_grad()
    def reverse_diffusion_variable(
        self,
        inp_ri: torch.Tensor,
        num_steps: int = 16,
        solver: str = "heun",          # "euler" or "heun"
        t_stop: float = 0.0,
    ):
        """
        Variable-step reverse diffusion for delta-norm mode.
        - num_steps can be anything (e.g., 4, 8, 16, 32, 64)
        - solver="heun" usually keeps quality with fewer steps.
        """
        #assert self.residual_mode == "next_delta_norm", \
        #    "This sampler is intended for next_delta_norm (velocity) mode."

        x = inp_ri
        B = x.shape[0]
        device = x.device

        # Continuous time grid: go from T_train -> 0 in num_steps
        ts = torch.linspace(
            float(self.diffusion_steps), float(t_stop),
            steps=num_steps + 1, device=device, dtype=torch.float32
        )

        xs = []
        for i in range(num_steps):
            t_curr = ts[i]
            t_next = ts[i + 1]

            t_curr_b = torch.full((B,), t_curr, device=device, dtype=torch.float32)
            t_next_b = torch.full((B,), t_next, device=device, dtype=torch.float32)

            a_curr = self.alpha_continuous(t_curr_b)
            a_next = self.alpha_continuous(t_next_b)

            g = (a_next - a_curr).clamp_min(1e-6).view(B, 1, 1, 1)

            # v_hat at current time
            v1 = self.model(x, t_curr_b)

            if solver == "euler":
                x = x + g * v1
            elif solver == "heun":
                # predictor
                x_e = x + g * v1
                # corrector
                v2 = self.model(x_e, t_next_b)
                x = x + g * 0.5 * (v1 + v2)
            else:
                raise ValueError("solver must be 'euler' or 'heun'")

            xs.append(x)

        return xs  # list of (B,4,F,T) like your current code

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
        preds = self.reverse_diffusion_variable(inp_ri, num_steps=self.diffusion_steps, solver="heun")
        # preds = self.reverse_diffusion_variable(inp_ri, num_steps=4, solver="heun")
        self.ema.restore()

        # save first N examples per batch
        sr = getattr(self.pre_params, "sr", 44100)
        Bsave = min(5, reverb_ri.shape[0])
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
                        f"| NMI {out['nmi'].item():.4f} | Audio {out['audio'].item():.4f} ")
                # TB per-step
                self.tb_train.add_scalar("loss/noise", out["noise"].item(), gstep)
                self.tb_train.add_scalar("loss/nmi",   out["nmi"].item(),   gstep)
                self.tb_train.add_scalar("loss/audio", out["audio"].item(), gstep)
                gstep += 1

            # ---- Validate with EMA weights ----
            self.ema.apply_shadow()
            self.model.eval()
            noise_sum = audio_sum = nmi_sum = 0.0
            n_batches = 0
            self.sisdr.reset(); 
            self.sisdri.reset() 

            with torch.no_grad():
                for batch in self.val_loader:
                    out = self._step(batch, train=False)
                    noise_sum += out["noise"].item()
                    audio_sum += out["audio"].item()
                    nmi_sum   += out["nmi"].item()
                    n_batches += 1
                    # SI metrics (stubs)
                    self.sisdr.update(out["clean_wav"], out["est_wav"])
                    self.sisdri.update(out["clean_wav"], out["est_wav"], out["inp_wav"]) 
                    
            self.ema.restore()

            noise_avg = noise_sum / max(n_batches,1)
            audio_avg = audio_sum / max(n_batches,1)
            nmi_avg   = nmi_sum   / max(n_batches,1)
            val_loss  = noise_avg + audio_avg + nmi_avg

            # TB per-epoch
            self.tb_val.add_scalar("loss/noise", noise_avg, epoch)
            self.tb_val.add_scalar("loss/nmi",   nmi_avg,   epoch)
            self.tb_val.add_scalar("loss/audio", audio_avg, epoch)
            self.tb_val.add_scalar("metrics/si_sdr", self.sisdr.result(), epoch)
            self.tb_val.add_scalar("metrics/si_sdri", self.sisdri.result(), epoch)


            print("----")
            print(f"Total Noise MAE Loss {noise_avg:.4f}")
            print(f"Total NMI Loss      {nmi_avg:.4f}")
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


# ---- CLI entry point (GPU selection etc.) ----
def main():
    set_global_seed(42, deterministic=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/out_combined_stereo")
    parser.add_argument("--model-name", default="CDiff_UNet_s_64ch_att_1248_delta-norm_var-adamW-emb_var")
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    args = parser.parse_args()

    # choose device
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)  
        print("Using GPU:", device)
    else:
        device = "cpu"; print("No GPU, using CPU")


    # Load config parameters
    params = Config()
    pre_params = params.data
    train_params = params.train
    model_params = params.model
 

    # dataloaders
    train_loader, val_loader = build_dataloaders(pre_params, args.data_dir)
    dataloaders = (train_loader, val_loader)

    # model
    model = UNetRI(model_params)

    # --- print parameter counts ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {trainable_params:,} trainable / {total_params:,} total "
        f"({trainable_params/1e6:.2f} M params)")

    # trainer
    out_dir = f"saved_models/{args.model_name}"
    trainer = ColdRIUNetTrainer(model, pre_params, train_params, 
                                dataloaders, out_dir, device=device)
    # --- RESUME LOGIC ---
    start_epoch = 0
    best_loss = None
    if args.resume:
        if os.path.exists(trainer.ckpt_path):
            start_epoch, best_loss = trainer.load_checkpoint()
        else:
            print(f"WARNING: resume flag set but checkpoint not found at {trainer.ckpt_path}. "
                  f"Starting from scratch.")

    trainer.train(start_epoch=start_epoch, best_loss=best_loss)
    #trainer.generate_random_batch(start_epoch)

if __name__ == "__main__":
    main()