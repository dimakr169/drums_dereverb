import os, time, math, random, argparse
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

from dataset.stereo_dataset import build_dataloaders  # as before
from config import Config
from backbones.unet_stereo import UNetRI
from backbones.metrics_torch import SISDR, SISDRi, NormalizedMutualInformationLoss, NMILossConfig

# ---- EMA wrapper ----
class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = model
        # list of EMA parameters (only for requires_grad params)
        self.shadow = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        self.backup = None

    @torch.no_grad()
    def update(self):
        i = 0
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            self.shadow[i].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
            i += 1

    @torch.no_grad()
    def apply_shadow(self):
        self.backup = [p.detach().clone() for p in self.model.parameters() if p.requires_grad]
        i = 0
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            p.data.copy_(self.shadow[i])
            i += 1

    @torch.no_grad()
    def restore(self):
        if self.backup is None:
            return
        i = 0
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            p.data.copy_(self.backup[i])
            i += 1
        self.backup = None

    # --------- NEW: for checkpointing ---------
    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {
            "shadow": [t.clone() for t in self.shadow],
            "decay": self.decay,
        }

    def load_state_dict(self, state):
        """Load EMA state from checkpoint.

        Accepts either:
        - a dict {"shadow": [...], "decay": ...}
        - or directly a list of tensors (for backwards compatibility
          with old checkpoints where we saved just [t.clone() for t in shadow]).
        """
        if isinstance(state, dict):
            shadow_list = state.get("shadow", [])
            self.decay = state.get("decay", self.decay)
        else:
            # old format: list of tensors
            shadow_list = state

        device = next(self.model.parameters()).device
        self.shadow = [t.to(device).detach().clone() for t in shadow_list]

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

class DDPMRIUNetTrainer:
    """
    Classic Gaussian DDPM baseline, conditional on reverberant input.
    - x0 = clean_ri
    - y = reverb_ri (condition, not diffused)
    - Forward: q(x_t|x0) = N(sqrt(ā_t) x0, (1-ā_t)I)
    - Model predicts ε given (x_t, y, t)
    """
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
        self.diffusion_steps = 200 #train_params.diffusions_steps

        # ===== CDiffuSE-like beta schedule (Large model) =====
        # CDiffuSE-Large uses T=200 with β ∈ [1e-4, 0.0095] (linear). 
        self.betas = torch.linspace(1e-4, 9.5e-3, self.diffusion_steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)   # (T,)

        # Convenience stuff
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

        # ===== CDiffuSE interpolation weights m_t and δ_t (Eq. 8–9) =====
        # Choose a valid interpolation schedule m_t in [0,1] with m_0 = 0, m_T ≈ 1.
        # A simple and safe choice: m_t = sqrt(1 - alpha_bar_t)
        m = torch.sqrt(torch.clamp(1.0 - self.alpha_bar, min=1e-8))  # (T,)

        # δ_t = (1 - ᾱ_t) - m_t^2 ᾱ_t   (Eq. 9)  -> simplifies to (1 - ᾱ_t)^2
        delta = (1.0 - self.alpha_bar) - (m ** 2) * self.alpha_bar
        delta = torch.clamp(delta, min=1e-12)  # avoid numerical issues

        # Pad t=0 so that index t in [0..T] corresponds to step t
        self.alpha_bar_full = torch.cat(
            [torch.ones(1, device=self.device), self.alpha_bar], dim=0
        )  # (T+1,)

        self.m_full = torch.cat(
            [torch.zeros(1, device=self.device), m], dim=0
        )  # (T+1,)

        self.delta_full = torch.cat(
            [torch.zeros(1, device=self.device), delta], dim=0
        )  # (T+1,)


        # optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(train_params.learning_rate),
            betas=(train_params.beta1, train_params.beta2),
            eps=train_params.eps,
        )

        # AMP, EMA
        self.scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))
        self.ema = EMAModel(self.model, decay=train_params.ema_decay)

        # losses & metrics
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
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

        # ISTFT params
        self.center = pre_params.center
        self.window = torch.hann_window(pre_params.win, periodic=True, device=self.device)

        # LR scheduler (per-step) – reuse your utility
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

    # ---- helpers ----
    def get_signal_from_RI_stft(self, ri_stft):
        n_fft = self.pre_params.fft
        hop = self.pre_params.hop
        win = self.pre_params.win
        length = getattr(self.pre_params, "wave_len", None)
        return istft_from_ri(ri_stft, n_fft=n_fft, hop=hop, win_length=win,
                             window=self.window, center=self.center, length=length)
    
    def load_checkpoint(self):
        """Load model, optimizer, scaler, EMA and return (next_epoch, best_loss)."""
        print(f"Loading checkpoint from: {self.ckpt_path}")
        ckpt = torch.load(self.ckpt_path, map_location=self.device)

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
    

    def _random_timesteps(self, bsize):
        # Uniform integers in [1, T]
        return torch.randint(low=1, high=self.diffusion_steps+1,
                             size=(bsize,), device=self.device)
    

    # ---- core training step (CDiffuse) ----
    def _step(self, batch, train=True, global_step=0):
        reverb_ri, clean_ri = batch  # (B,4,F,T)
        reverb_ri = reverb_ri.to(self.device, non_blocking=True)
        clean_ri  = clean_ri.to(self.device, non_blocking=True)

        bsize = reverb_ri.shape[0]
        timesteps = self._random_timesteps(bsize)  # (B,)
        # Index into full arrays (ᾱ_0..ᾱ_T etc.)
        ab_t   = self.alpha_bar_full.index_select(0, timesteps)       # (B,)
        m_t    = self.m_full.index_select(0, timesteps)               # (B,)
        delta_t = self.delta_full.index_select(0, timesteps)          # (B,)

        # Reshape to broadcast over (B,4,F,T)
        ab_t   = ab_t.view(-1, 1, 1, 1)
        m_t    = m_t.view(-1, 1, 1, 1)
        delta_t = delta_t.view(-1, 1, 1, 1)

        sqrt_ab_t   = torch.sqrt(ab_t)
        one_minus_ab_t = 1.0 - ab_t
        sqrt_1mab_t = torch.sqrt(torch.clamp(one_minus_ab_t, min=1e-8))
        sqrt_delta_t = torch.sqrt(torch.clamp(delta_t, min=1e-12))

        # ε ~ N(0, I)
        eps = torch.randn_like(clean_ri)

        # ----- Conditional diffusion process (Eq. 8) -----
        # x_t mean is interpolation between clean and reverb
        xt_mean = (1.0 - m_t) * sqrt_ab_t * clean_ri + m_t * sqrt_ab_t * reverb_ri
        x_t = xt_mean + sqrt_delta_t * eps

        # ----- Target combined noise η_t (Eq. 21) -----
        # η_t = sqrt(1/(1-ᾱ_t)) [ m_t sqrt(ᾱ_t) (y - x0) + sqrt(δ_t) ε ]
        num = m_t * sqrt_ab_t * (reverb_ri - clean_ri) + sqrt_delta_t * eps
        eta_t = num / sqrt_1mab_t  # (B,4,F,T)

        # Network input: x_t + reverb_ri as condition
        x_in = torch.cat([x_t, reverb_ri], dim=1)  # (B, 8, F, T)

        self.model.train(train)
        with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):

            eps_pred = self.model(x_in, timesteps)  # (B,4,F,T)

            # CDiffuSE training objective: MSE(η̂, η)
            eps_loss = self.mse(eps_pred, eta_t)

            # ---- Reconstruct x0_pred from x_t & η̂ for metrics/logging ----
            # x0_pred = (x_t - sqrt(1-ᾱ_t) * η̂_t) / sqrt(ᾱ_t)
            x0_pred = (x_t - sqrt_1mab_t * eps_pred) / (sqrt_ab_t + 1e-8)

            # waveforms for metrics
            est_wav = self.get_signal_from_RI_stft(x0_pred)   # (B,2,T)
            tar_wav = self.get_signal_from_RI_stft(clean_ri)  # (B,2,T)

            # audio MAE & NMI (LOGGING ONLY)
            audio_loss = self.l1(est_wav, tar_wav)
            nmi_loss   = self.nmi_loss(tar_wav, est_wav)

            # only eps_loss is used for backprop
            loss = eps_loss


        if train:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update()
            if self.scheduler is not None:
                self.scheduler.step()

        # also compute reverberant waveform for SI-SDRi
        with torch.no_grad():
            inp_wav = self.get_signal_from_RI_stft(reverb_ri)   # reverberant input

        return {
            "loss": loss.detach(),
            "eps": eps_loss.detach(),
            "audio": audio_loss.detach(),
            "nmi": nmi_loss.detach(),
            "est_wav": est_wav.detach(),
            "tar_wav": tar_wav.detach(),
            "inp_wav": inp_wav.detach(),
            "clean_wav": tar_wav.detach(),
        }

    def _cdiff_coeffs(self, t: int):
        """
        Compute c_x, c_y, c_eps, delta_tilde for a *scalar* t (1..T)
        according to Eqs. (15,17,18,19,20).
        Returns 0-dim tensors on self.device.
        """
        device = self.device
        T = self.diffusion_steps
        assert 1 <= t <= T

        # Scalars (0-dim tensors)
        alpha_t       = self.alphas[t-1]                 # α_t
        alpha_bar_t   = self.alpha_bar_full[t]           # ᾱ_t
        alpha_bar_tm1 = self.alpha_bar_full[t-1]         # ᾱ_{t-1}
        m_t           = self.m_full[t]
        m_tm1         = self.m_full[t-1]
        delta_t       = self.delta_full[t]
        delta_tm1     = self.delta_full[t-1]

        # δ_{t|t-1} (Eq. 15)
        if t > 1:
            delta_t_given_tm1 = delta_t - ((1.0 - m_t) / (1.0 - m_tm1))**2 * alpha_t * delta_tm1
        else:
            # For t=1, δ_0 = 0, so Eq. 15 collapses to δ_1
            delta_t_given_tm1 = delta_t

        # \tilde δ_t (Eq. 17)
        if t > 1 and delta_tm1 > 0:
            delta_tilde = delta_t_given_tm1 * delta_t / delta_tm1
        else:
            # deterministic at last step
            delta_tilde = torch.zeros_like(delta_t)

        sqrt_alpha_t       = torch.sqrt(alpha_t)
        inv_sqrt_alpha_t   = 1.0 / (sqrt_alpha_t + 1e-8)
        sqrt_alpha_bar_tm1 = torch.sqrt(alpha_bar_tm1)
        sqrt_one_minus_ab_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-8))

        # c_x (Eq. 18)
        c_x = ((1.0 - m_t) / (1.0 - m_tm1 + 1e-8)) * (delta_tm1 / (delta_t + 1e-8)) * sqrt_alpha_t \
              + (1.0 - m_tm1) * (delta_t_given_tm1 / (delta_t + 1e-8)) * inv_sqrt_alpha_t

        # c_y (Eq. 19)
        c_y = (m_tm1 * delta_t - m_t * (1.0 - m_t) / (1.0 - m_tm1 + 1e-8) * alpha_t * delta_tm1)
        c_y = c_y * (sqrt_alpha_bar_tm1 / (delta_t + 1e-8))

        # c_eps (Eq. 20)
        c_eps = (1.0 - m_tm1) * (delta_t_given_tm1 / (delta_t + 1e-8)) \
                * sqrt_one_minus_ab_t * sqrt_alpha_t

        return c_x.to(device), c_y.to(device), c_eps.to(device), delta_tilde.to(device)


    # ---- reverse diffusion (sampling) ----
    @torch.no_grad()
    def reverse_diffusion(self, reverb_ri, step_stop=0):
        """
        CDiffuSE sampling (Algorithm 2):
        - Start from x_T ~ N( sqrt(ᾱ_T) y, δ_T I )
        - Iterate t = T..1: x_{t-1} ~ N( c_x x_t + c_y y - c_eps eps_theta, \tilde δ_t I )
        If step_stop > 0, stop at that t (for debugging partial denoising).
        """
        self.model.eval()
        bsize = reverb_ri.shape[0]
        device = self.device
        T = self.diffusion_steps

        # ---- Initial x_T (Eq. 10) ----
        alpha_bar_T = self.alpha_bar_full[T]         # scalar tensor
        delta_T     = self.delta_full[T]

        sqrt_ab_T   = torch.sqrt(alpha_bar_T)
        sqrt_delta_T = torch.sqrt(torch.clamp(delta_T, min=1e-12))

        # x_T = sqrt(ᾱ_T) y + sqrt(δ_T) ε_T
        eps_T = torch.randn_like(reverb_ri)
        x = sqrt_ab_T * reverb_ri + sqrt_delta_T * eps_T

        xs = [x]

        # ---- Reverse steps (Algorithm 2) ----
        for t in range(T, step_stop, -1):
            T_batch = torch.full((bsize,), t, device=device, dtype=torch.long)

            # network prediction of combined noise
            x_in = torch.cat([x, reverb_ri], dim=1)
            eps_hat = self.model(x_in, T_batch)  # ε_θ(x_t,y,t) ~ η_t

            # coefficients
            c_x, c_y, c_eps, delta_tilde = self._cdiff_coeffs(t)

            # reshape coeffs to broadcast
            c_x   = c_x.view(1, 1, 1, 1)
            c_y   = c_y.view(1, 1, 1, 1)
            c_eps = c_eps.view(1, 1, 1, 1)

            mean = c_x * x + c_y * reverb_ri - c_eps * eps_hat

            if t > 1 and delta_tilde > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(torch.clamp(delta_tilde, min=1e-12)) * noise
            else:
                # last step: deterministic
                x = mean

            xs.append(x)

        return xs


    # ---- sample & save audio for a random validation batch ----
    @torch.no_grad()
    def generate_random_batch(self, epoch):
        out_root = os.path.join(self.output_dir, "samples", f"epoch_{epoch}")
        os.makedirs(out_root, exist_ok=True)

        try:
            batch = next(iter(self.val_loader))
        except StopIteration:
            return

        reverb_ri, clean_ri = [b.to(self.device) for b in batch]

        # Use EMA weights for sampling
        self.ema.apply_shadow()
        xs = self.reverse_diffusion(reverb_ri, step_stop=0)  # full T..1
        self.ema.restore()

        # xs[-1] is x_0 (final estimate)
        est_ri_final = xs[-1]                               # (B,4,F,T)
        est_wav = self.get_signal_from_RI_stft(est_ri_final)  # (B,2,T)
        inp_wav = self.get_signal_from_RI_stft(reverb_ri)
        tar_wav = self.get_signal_from_RI_stft(clean_ri)

        sr = getattr(self.pre_params, "sr", 44100)
        Bsave = min(4, reverb_ri.shape[0])

        for i in range(Bsave):
            val_dir = os.path.join(out_root, f"val_{i}")
            os.makedirs(val_dir, exist_ok=True)

            inp_i = inp_wav[i].permute(1,0).cpu().numpy()
            tar_i = tar_wav[i].permute(1,0).cpu().numpy()
            est_i = est_wav[i].permute(1,0).cpu().numpy()

            sf.write(os.path.join(val_dir, "input_reverb.wav"),  inp_i, sr)
            sf.write(os.path.join(val_dir, "target_clean.wav"),  tar_i, sr)
            sf.write(os.path.join(val_dir, "ddpm_est_clean.wav"), est_i, sr)

            # Also dump a few intermediate steps to hear the evolution
            T = self.diffusion_steps
            num_intermediate = min(8, len(xs))
            idxs = torch.linspace(0, len(xs)-1, steps=num_intermediate).long().tolist()

            for idx in idxs:
                ri_step = xs[idx][i:i+1]
                wav_step = self.get_signal_from_RI_stft(ri_step).squeeze(0).permute(1,0).cpu().numpy()
                t_label = T - idx   # rough t label
                sf.write(os.path.join(val_dir, f"ddpm_step_{t_label:03d}.wav"),
                         wav_step, sr)


    # ---- main training loop ----
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
                    print(f"Batch {b:5d} | eps {out['eps'].item():.4f} "
                          f"| NMI {out['nmi'].item():.4f} | Audio {out['audio'].item():.4f}")
                self.tb_train.add_scalar("loss/eps",   out["eps"].item(),   gstep)
                self.tb_train.add_scalar("loss/nmi",   out["nmi"].item(),   gstep)
                self.tb_train.add_scalar("loss/audio", out["audio"].item(), gstep)
                gstep += 1

            # ---- Validate with EMA ----
            self.ema.apply_shadow()
            self.model.eval()
            eps_sum = audio_sum = nmi_sum = 0.0
            n_batches = 0
            self.sisdr.reset()
            self.sisdri.reset()

            with torch.no_grad():
                for batch in self.val_loader:
                    out = self._step(batch, train=False)
                    eps_sum   += out["eps"].item()
                    audio_sum += out["audio"].item()
                    nmi_sum   += out["nmi"].item()
                    n_batches += 1
                    self.sisdr.update(out["clean_wav"], out["est_wav"])
                    self.sisdri.update(out["clean_wav"], out["est_wav"], out["inp_wav"])

            self.ema.restore()

            eps_avg   = eps_sum   / max(n_batches, 1)
            audio_avg = audio_sum / max(n_batches, 1)
            nmi_avg   = nmi_sum   / max(n_batches, 1)
            # val_loss  = eps_avg + audio_avg + nmi_avg
            val_loss = eps_avg 

            self.tb_val.add_scalar("loss/eps",   eps_avg,   epoch)
            self.tb_val.add_scalar("loss/nmi",   nmi_avg,   epoch)
            self.tb_val.add_scalar("loss/audio", audio_avg, epoch)
            self.tb_val.add_scalar("metrics/si_sdr",  self.sisdr.result(),  epoch)
            self.tb_val.add_scalar("metrics/si_sdri", self.sisdri.result(), epoch)

            print("----")
            print(f"Total epsilon Loss {eps_avg:.4f}")
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

def main():
    set_global_seed(42, deterministic=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/out_combined_stereo")
    parser.add_argument("--model-name", default="CDiffuse_UNet_s_64ch_att_1248")
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)
        print("Using GPU:", device)
    else:
        device = "cpu"; print("No GPU, using CPU")

    params = Config()
    pre_params = params.data
    train_params = params.train
    model_params = params.model

    # IMPORTANT for DDPM baseline:
    model_params.in_chans = 8
    model_params.residual_prediction = False

    train_loader, val_loader = build_dataloaders(pre_params, args.data_dir)
    dataloaders = (train_loader, val_loader)

    model = UNetRI(model_params)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {trainable_params:,} trainable / {total_params:,} total "
          f"({trainable_params/1e6:.2f} M params)")

    out_dir = f"saved_models/{args.model_name}"
    trainer = DDPMRIUNetTrainer(model, pre_params, train_params, dataloaders, out_dir, device=device)

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


if __name__ == "__main__":
    main()
