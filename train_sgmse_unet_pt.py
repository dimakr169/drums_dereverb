import os, time, math, random, argparse
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
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


# -------------------------------------------------------------------------
# SGMSE+ style OU-VE SDE (bridge between clean x0 and reverberant y)
# -------------------------------------------------------------------------
class SGMSESDE:
    """
    OU-VE SDE used in SGMSE+ (adapted from OUVESDE in sdes.py).

    Forward SDE (continuous time t ∈ [0, 1]):

        dX_t = theta * (y - X_t) dt + g(t) dW_t

    with log-linear diffusion schedule

        sigma(t) = sigma_min * (sigma_max / sigma_min) ** t

    and closed-form marginal

        X_t | X_0 = x0, y  ~  N( mean(t), std(t)^2 I )

        mean(t) = exp(-theta * t) * x0 + (1 - exp(-theta * t)) * y
        std(t)  given by the closed-form solution in _std().
    """

    def __init__(
        self,
        sigma_min: float = 0.05,
        sigma_max: float = 0.5,
        theta: float = 1.5,
        N: int = 30,
        t_eps: float = 1e-3,
        device: str = "cuda",
    ):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.theta = float(theta)
        self.N = int(N)
        self.t_eps = float(t_eps)
        self.device = device

        # logsig = log(sigma_max / sigma_min) as in OUVESDE
        self.logsig = math.log(self.sigma_max / self.sigma_min)
        # constant factor used in the diffusion g(t) = sigma(t) * sqrt(2 * logsig)
        self._diffusion_scale = math.sqrt(2.0 * self.logsig)

    @property
    def T(self) -> float:
        # final diffusion time
        return 1.0

    # ---------- helper: log-linear sigma schedule ----------
    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        sigma(t) = sigma_min * (sigma_max / sigma_min) ** t

        t : (B,)
        returns sigma(t) : (B,)
        """
        ratio = self.sigma_max / self.sigma_min
        return self.sigma_min * (ratio ** t)

    # ---------- forward SDE (drift + diffusion) ----------
    def sde(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward OU-VE SDE:

            dX_t = theta * (y - X_t) dt + sigma(t) * sqrt(2 * logsig) dW_t

        Returns
        -------
        drift     : same shape as x
        diffusion : (B,)  scalar per batch element
        """
        if t.dim() == 0:
            t = t[None]
        t = t.to(device=x.device, dtype=x.dtype)  # (B,)

        drift = self.theta * (y - x)                    # (B, C, F, T)
        sigma_t = self._sigma(t)                        # (B,)
        # IMPORTANT: use math.sqrt -> float, then multiply by tensor
        diffusion = sigma_t * self._diffusion_scale     # (B,)

        return drift, diffusion

    # ---------- closed-form OU-VE marginal ----------
    def _mean(self, x0: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        mean(t) = exp(-theta * t) * x0 + (1 - exp(-theta * t)) * y
        """
        theta = self.theta
        exp_interp = torch.exp(-theta * t)[:, None, None, None]
        return exp_interp * x0 + (1.0 - exp_interp) * y

    def _std(self, t: torch.Tensor) -> torch.Tensor:
        """
        std(t) from the closed-form solution in OUVESDE._std:

            P(t) = sigma_min^2 * exp(-2 θ t) *
                   (exp(2 (θ + logsig) t) - 1) * logsig / (θ + logsig)

            std(t) = sqrt(P(t))
        """
        sigma_min, theta, logsig = self.sigma_min, self.theta, self.logsig

        return torch.sqrt(
            (
                sigma_min**2
                * torch.exp(-2.0 * theta * t)
                * (torch.exp(2.0 * (theta + logsig) * t) - 1.0)
                * logsig
            )
            / (theta + logsig)
        )

    def marginal_prob(
        self,
        clean_ri: torch.Tensor,   # x0
        reverb_ri: torch.Tensor,  # y
        t: torch.Tensor,          # (B,) in [t_eps, T]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns mean and std of p(x_t | x0, y, t):

            x_t = mean + std * z,   z ~ N(0,I)

        mean : (B, C, F, T)
        std  : (B, 1, 1, 1)
        """
        if t.dim() == 0:
            t = t[None]

        B = clean_ri.size(0)
        t = t.to(device=clean_ri.device, dtype=clean_ri.dtype).view(B)

        mean = self._mean(clean_ri, reverb_ri, t)       # (B,C,F,T)
        std  = self._std(t).view(B, 1, 1, 1)            # (B,1,1,1)
        return mean, std

    # ---------- prior at time T: p_T(x | y) ----------
    def prior_sampling(
        self,
        shape: Tuple[int, int, int, int],
        reverb_ri: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample from p_T(x | y) = N( y, std(T)^2 I ) as in OUVESDE.prior_sampling.
        """
        B = shape[0]
        device = reverb_ri.device
        z = torch.randn(shape, device=device, dtype=reverb_ri.dtype)

        t_T = torch.ones((B,), device=device, dtype=reverb_ri.dtype)
        std_T = self._std(t_T)  # (B,)
        return reverb_ri + std_T.view(B, 1, 1, 1) * z

    # ---------- sample diffusion time t ~ Uniform[t_eps, T] ----------
    def sample_time(self, batch_size: int) -> torch.Tensor:
        """
        Sample t ~ Uniform[t_eps, T]. For the log-linear sigma(t) this is
        equivalent to sampling log σ approximately uniformly, as in SGMSE+.
        """
        t = torch.rand(batch_size, device=self.device)  # in [0,1]
        t = t * (self.T - self.t_eps) + self.t_eps      # in [t_eps, T]
        return t


# -------------------------------------------------------------------------
# Reverse-time SDE / probability-flow ODE wrapper (minimal version)
# -------------------------------------------------------------------------
class ReverseSDE:
    """
    Minimal reverse-time SDE wrapper, following Song et al. and SGMSE.

    If probability_flow = True we get the probability-flow ODE used in SGMSE+.
    """

    def __init__(self, sde: SGMSESDE, score_model, probability_flow: bool = True):
        self.sde = sde
        self.score_model = score_model  # callable(x, y, t) -> score
        self.probability_flow = probability_flow
        self.N = sde.N
        self.T = sde.T

    def drift_ode(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Probability-flow ODE drift:

            dx/dt = f(x, t) - 0.5 g(t)^2 score(x, t, y)

        where f,g come from the forward SDE and score is ∇_x log p_t(x | y).
        """
        drift, diffusion = self.sde.sde(x, y, t)          # drift: x-shape, diffusion: (B,)
        score = self.score_model(x, y, t)                # same shape as x
        g2 = (diffusion ** 2).view(x.size(0), 1, 1, 1)   # broadcast
        ode_drift = drift - 0.5 * g2 * score
        return ode_drift


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

# ------------------------------------------------------------------
# Trainer: SGMSE-style score matching on stereo RI-STFT
# ------------------------------------------------------------------
class SGMSEUNetTrainer:
    """
    SGMSE+-style score-based diffusion baseline using your UNetRI.

    - x0 = clean_ri
    - y  = reverb_ri (condition)
    - Forward marginal x_t = mean(x0,y,t) + sigma(t) * z
    - Model predicts score s_theta(x_t, t, y)
    - Loss: E[ || sigma(t) * s_theta(x_t,t,y) + z ||^2 ]
    """
    def __init__(self, model: UNetRI, pre_params, train_params, dataloaders, output_dir, device="cuda"):
        self.model = model.to(device)
        self.pre_params = pre_params
        self.train_params = train_params
        self.train_loader, self.val_loader = dataloaders
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "samples"), exist_ok=True)

        self.device = device
        self.diffusion_steps = 100 #for SGMSE+ Inference #train_params.diffusions_steps

        # SDE hyperparameters (roughly SGMSE+ / 48 kHz, tweak if needed)
        self.sde = SGMSESDE(
            sigma_min=0.1,
            sigma_max=1.0,
            theta=2.0,
            N=self.diffusion_steps,   # e.g. 30
            t_eps=1e-3,
            device=device,
        )


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
    
    

    # ---- core training step (SGMSE+) ----
    def _step(self, batch, train=True, global_step=0):
        reverb_ri, clean_ri = batch  # (B,4,F,T)
        reverb_ri = reverb_ri.to(self.device, non_blocking=True)
        clean_ri  = clean_ri.to(self.device, non_blocking=True)

        B = clean_ri.size(0)

        # Sample diffusion time
        t = self.sde.sample_time(B)  # (B,)

        # Forward marginal
        mean, std = self.sde.marginal_prob(clean_ri, reverb_ri, t)  # new SGMSESDE
        z = torch.randn_like(clean_ri)

        x_t = mean + std * z
        dnn_in = torch.cat([x_t, reverb_ri], dim=1)
        self.model.train(train)
        with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):

            score_hat = self.model(dnn_in, t)  # (B,4,F,T)
            # SGMSE / VE-style objective: E[ || sigma * s_theta + z ||^2 ]
            loss_per_elem = (std * score_hat + z) ** 2
            loss_per_sample = loss_per_elem.view(B, -1).mean(dim=-1)
            loss = loss_per_sample.mean()

            # Tweedie: x0 ≈ x_t + sigma^2 * s_theta(x_t)
            x0_est = x_t + (std ** 2) * score_hat

            # waveforms for metrics
            est_wav = self.get_signal_from_RI_stft(x0_est)   # (B,2,T)
            tar_wav = self.get_signal_from_RI_stft(clean_ri)  # (B,2,T)

            # audio MAE & NMI (LOGGING ONLY)
            audio_loss = self.l1(est_wav, tar_wav)
            nmi_loss   = self.nmi_loss(tar_wav, est_wav)



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
            "score": loss.detach(),
            "audio": audio_loss.detach(),
            "nmi": nmi_loss.detach(),
            "est_wav": est_wav.detach(),
            "tar_wav": tar_wav.detach(),
            "inp_wav": inp_wav.detach(),
            "clean_wav": tar_wav.detach(),
        }



    # ---- reverse diffusion (sampling, SGMSE⁺-style ODE) ----
    @torch.no_grad()
    def reverse_diffusion(
        self,
        reverb_ri: torch.Tensor,
        num_steps: int | None = None,
        return_all: bool = True,
    ):
        """
        SGMSE+ style probability-flow ODE sampler.

        Args
        ----
        reverb_ri : (B, 4, F, T) tensor with reverberant RI-STFT
        num_steps : optional number of ODE steps; defaults to self.sde.N
        return_all: if True, return list of states along the trajectory;
                    if False, return only final x_0 estimate.

        Returns
        -------
        xs or x0 :
            if return_all=True -> list of (B,4,F,T),
            else -> (B,4,F,T) final estimate.
        """
        self.model.eval()

        device = self.device
        B = reverb_ri.size(0)

        # Short aliases
        sde = self.sde
        T = sde.T
        N = num_steps if num_steps is not None else sde.N
        t_eps = sde.t_eps

        # ---------------- initial sample at time T ----------------
        x = sde.prior_sampling(reverb_ri.shape, reverb_ri).to(device)
        xs = [x]

        # ---------------- probability-flow ODE integration --------
        t_grid = torch.linspace(T, t_eps, steps=N + 1, device=device)

        # score model wrapper: UNet -> score(x, t, y)
        def score_model(x_t: torch.Tensor, y: torch.Tensor, t_scalar: torch.Tensor):
            if t_scalar.dim() == 0:
                t_scalar = t_scalar[None]
            dnn_in = torch.cat([x_t, y], dim=1)
            return self.model(dnn_in, t_scalar)

        rsde = ReverseSDE(sde, score_model, probability_flow=True)

        for i in range(N):
            t_i = t_grid[i].expand(B)        # (B,)
            t_next = t_grid[i + 1].expand(B) # (B,)
            dt = t_next - t_i                # negative

            drift_ode = rsde.drift_ode(x, reverb_ri, t_i)  # (B,4,F,T)
            x = x + drift_ode * dt.view(B, 1, 1, 1)
            xs.append(x)

        if return_all:
            return xs
        else:
            return xs[-1]

    @torch.no_grad()
    def reverse_diffusion_pc(
        self,
        reverb_ri: torch.Tensor,
        num_steps: int | None = None,
        snr: float = 0.5,
        n_corr_steps: int = 1,
        return_all: bool = True,
    ):
        """
        Predictor–Corrector sampler in the spirit of SGMSE+.

        Args
        ----
        reverb_ri : (B, 4, F, T) reverberant RI-STFT (conditioning)
        num_steps : number of predictor steps (time discretisation).
                    Defaults to self.sde.N (e.g. 30).
        snr       : target SNR for Langevin corrector (SGMSE+ ~0.33–0.5).
        n_corr_steps : Langevin steps per predictor step (usually 1).
        return_all: if True, return list of x_mean states along trajectory.

        Returns
        -------
        xs or x0:
            list of (B,4,F,T) if return_all=True,
            else final (B,4,F,T) estimate (denoised x_mean at t≈0).
        """
        self.model.eval()

        device = self.device
        B = reverb_ri.size(0)

        sde = self.sde
        T = sde.T
        N = num_steps if num_steps is not None else sde.N
        t_eps = sde.t_eps

        # ----- initial sample at time T: x_T ~ p_T(x|y) -----
        x = sde.prior_sampling(reverb_ri.shape, reverb_ri).to(device)

        # we will store the *denoised* states (x_mean) for inspection
        xs = []

        # time grid from T -> t_eps (SGMSE uses uniform grid)
        t_grid = torch.linspace(T, t_eps, steps=N, device=device)

        # score model wrapper: UNet -> score(x, t, y)
        def score_fn(x_t: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 0:
                t = t[None]
            dnn_in = torch.cat([x_t, y], dim=1)
            return self.model(dnn_in, t)

        for i in range(N):
            t_i = t_grid[i].expand(B)  # (B,)

            # -------- CORRECTOR: Langevin (LangevinCorrector) --------
            target_snr = snr
            for _ in range(n_corr_steps):
                grad = score_fn(x, reverb_ri, t_i)        # (B,4,F,T)
                noise = torch.randn_like(x)

                grad_norm = torch.norm(
                    grad.reshape(grad.shape[0], -1), dim=-1
                ).mean()
                noise_norm = torch.norm(
                    noise.reshape(noise.shape[0], -1), dim=-1
                ).mean()

                # step size as in SGMSE LangevinCorrector
                step_size = ((target_snr * noise_norm / (grad_norm + 1e-12)) ** 2 * 2.0)
                step_size = step_size.view(1, 1, 1, 1)

                x_mean = x + step_size * grad
                x = x_mean + torch.sqrt(2.0 * step_size) * noise

            # -------- PREDICTOR: Euler–Maruyama step --------
            # reverse SDE drift and diffusion
            f, g = sde.sde(x, reverb_ri, t_i)                # f: (B,4,F,T), g: (B,)
            g_b = g.view(B, 1, 1, 1)
            score = score_fn(x, reverb_ri, t_i)              # (B,4,F,T)
            drift = f - (g_b ** 2) * score                   # reverse drift

            dt = -T / float(N)                               # constant step, like SGMSE
            sqrt_dt = math.sqrt(-dt)

            x_mean = x + drift * dt                          # deterministic part
            x = x_mean + g_b * sqrt_dt * torch.randn_like(x) # stochastic part

            # store *denoised* version (no fresh noise)
            xs.append(x_mean.detach().clone())

        # final output: last denoised state
        x0 = xs[-1]

        if return_all:
            return xs
        else:
            return x0
 

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
        # xs = self.reverse_diffusion(reverb_ri, num_steps=self.diffusion_steps)
        xs = self.reverse_diffusion_pc(
            reverb_ri,
            num_steps= self.diffusion_steps,  # e.g. 30
            snr=0.5,
            n_corr_steps=1,
            return_all=True,
        )        
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
            sf.write(os.path.join(val_dir, "sgmse_est_clean.wav"), est_i, sr)

            # Also dump a few intermediate steps to hear the evolution
            T = self.diffusion_steps
            num_intermediate = min(15, len(xs))
            idxs = torch.linspace(0, len(xs)-1, steps=num_intermediate).long().tolist()

            for idx in idxs:
                ri_step = xs[idx][i:i+1]
                wav_step = self.get_signal_from_RI_stft(ri_step).squeeze(0).permute(1,0).cpu().numpy()
                t_label = T - idx   # rough t label
                sf.write(os.path.join(val_dir, f"sgmse_step_{t_label:03d}.wav"),
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
                    print(f"Batch {b:5d} | Score Loss {out['score'].item():.4f} "
                          f"| NMI {out['nmi'].item():.4f} | Audio {out['audio'].item():.4f}")
                self.tb_train.add_scalar("loss/score",   out["score"].item(),   gstep)
                self.tb_train.add_scalar("loss/nmi",   out["nmi"].item(),   gstep)
                self.tb_train.add_scalar("loss/audio", out["audio"].item(), gstep)
                gstep += 1

            # ---- Validate with EMA ----
            self.ema.apply_shadow()
            self.model.eval()
            score_sum = audio_sum = nmi_sum = 0.0
            n_batches = 0
            self.sisdr.reset()
            self.sisdri.reset()

            with torch.no_grad():
                for batch in self.val_loader:
                    out = self._step(batch, train=False)
                    score_sum += out["score"].item()
                    audio_sum += out["audio"].item()
                    nmi_sum   += out["nmi"].item()
                    n_batches += 1
                    self.sisdr.update(out["clean_wav"], out["est_wav"])
                    self.sisdri.update(out["clean_wav"], out["est_wav"], out["inp_wav"])

            self.ema.restore()

            score_avg = score_sum / max(n_batches, 1)
            audio_avg = audio_sum / max(n_batches, 1)
            nmi_avg   = nmi_sum   / max(n_batches, 1)
            # val_loss  = eps_avg + audio_avg + nmi_avg
            val_loss  = score_avg  

            self.tb_val.add_scalar("loss/score", score_avg,   epoch)
            self.tb_val.add_scalar("loss/nmi",   nmi_avg,   epoch)
            self.tb_val.add_scalar("loss/audio", audio_avg, epoch)
            self.tb_val.add_scalar("metrics/si_sdr",  self.sisdr.result(),  epoch)
            self.tb_val.add_scalar("metrics/si_sdri", self.sisdri.result(), epoch)

            print("----")
            print(f"Total Score Loss {score_avg:.4f}")
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
    parser.add_argument("--model-name", default="SGMSE+_UNet_s_64ch_att_1248")
    parser.add_argument("--gpu", default=1, type=int)
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

    # IMPORTANT for SGMSE+ baseline:
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
    trainer = SGMSEUNetTrainer(model, pre_params, train_params, dataloaders, out_dir, device=device)

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
    # trainer.generate_random_batch(start_epoch)


if __name__ == "__main__":
    main()
