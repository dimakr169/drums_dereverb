# inference_ri_unet_pt.py

import os, io, math
import librosa
import torch
import torch.nn as nn
import numpy as np
import pyloudnorm as pyln
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib.colors import PowerNorm
from scipy import signal
from typing import Tuple, Union, Optional
from types import SimpleNamespace
from backbones.unet_stereo import UNetRI
from backbones.dit_stereo import TransformerDiffuser
from backbones.dit_stereo2 import TransformerDiffuser as TransformerDiffuser2



# ========== Model builders using JSON entries ==========

def dict_to_cfg(d):
    """
    Recursively convert a dict into a SimpleNamespace so you can use cfg.foo
    instead of cfg['foo'].
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_cfg(v) for k, v in d.items()})
    return d

def build_model_from_entry(entry):
    """
    Instantiate a model (UNet or DiT) from a JSON model entry.
    entry keys:
      - type: 'unet' or 'dit'
      - model_config: dict with constructor kwargs
    """
    arch = entry["type"]
    cfg_dict  = entry["model_config"]
    scheme = entry["scheme"]  # "cold" (default) or "cdiffuse"

    # convert dict -> object with attributes
    cfg = dict_to_cfg(cfg_dict)

    if arch == "unet":
        model = UNetRI(cfg)
        #if scheme == "cold":
        #    model = UNetRI(cfg)
        #else:
        #    model = torch.compile(UNetRI(cfg), mode="reduce-overhead", fullgraph=False)
    elif arch == "dit":
        model = TransformerDiffuser(cfg)
        #model = torch.compile(TransformerDiffuser(cfg), mode="reduce-overhead", fullgraph=False)
    elif arch == "dit2":
        model = TransformerDiffuser2(cfg)
        #model = torch.compile(TransformerDiffuser(cfg), mode="reduce-overhead", fullgraph=False)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model


def load_ckpt_and_ema(model, entry, device):
    ckpt_path = entry["ckpt"]
    if not isinstance(ckpt_path, str):
        ckpt_path = ckpt_path["path"] if isinstance(ckpt_path, dict) else ckpt_path

    print(f"  - Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # --- 1) get state dict from ckpt ---
    if "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt  # in case you saved the raw state_dict

    model_state = model.state_dict()

    # --- 2) optionally filter extra keys (e.g. _relpos.*) ---
    filtered_state = {k: v for k, v in state.items() if k in model_state}

    # Not strictly needed, but nice for debug:
    unexpected_keys = [k for k in state.keys() if k not in model_state]
    if unexpected_keys:
        print("  - Ignoring unexpected keys in checkpoint (not present in current model):")
        for k in unexpected_keys:
            print(f"      {k}")

    # --- 3) load with strict=False so we don't crash on missing stuff ---
    missing, still_unexpected = model.load_state_dict(filtered_state, strict=False)
    if missing:
        print("  - Missing keys (present in model, absent in checkpoint):")
        for k in missing:
            print(f"      {k}")
    if still_unexpected:
        print("  - Still unexpected keys after filtering:")
        for k in still_unexpected:
            print(f"      {k}")

    model.to(device)

    # --- 4) EMA handling (if available and compatible) ---
    ema = None
    model_type = entry["type"]
    if "ema" in ckpt:
        ema_data = ckpt["ema"]
        #ema_decay = 0.995
        #ema_list = ema_data

        # New format from DDPMRIUNetTrainer: {"shadow": [...], "decay": ...}
        if isinstance(ema_data, dict):
            ema_list = ema_data.get("shadow", [])
            ema_decay = ema_data.get("decay", entry.get("ema_decay", 0.995))
        else:
        #    # Legacy format: directly a list of tensors
            ema_list = ema_data
            ema_decay = entry.get("ema_decay", 0.995)

        n_params = sum(1 for p in model.parameters() if p.requires_grad)

        if len(ema_list) == n_params:
            if model_type == "unet":
                # ema = EMAModel_UNet(model, decay=ema_decay)
                ema = EMAModel(model, decay=ema_decay)
            elif model_type == "dit" or model_type == "dit2":
                ema = EMAModel_DiT(model, decay=ema_decay)

            ema.load_state_dict(ema_list)
            # ema.shadow = [t.to(device) for t in ema_list]
            ema.apply_shadow()
            print(f"  - EMA applied ({len(ema_list)} params, decay={ema_decay}).")
        else:
            print(
                f"  - Skipping EMA: checkpoint ema length {len(ema_list)} "
                f"does not match model param count {n_params}"
            )
    else:
        print("  - No EMA in checkpoint.")

    return model, ema


def istft_from_ri(ri, n_fft, hop, win_length, window, center: bool, length: int | None):
    # ri: (B,4,F,T) [L_R, L_I, R_R, R_I]
    with torch.cuda.amp.autocast(enabled=False):
        L_real = ri[:, 0].float()
        L_imag = ri[:, 1].float()
        R_real = ri[:, 2].float()
        R_imag = ri[:, 3].float()

        L = torch.complex(L_real, L_imag)
        R = torch.complex(R_real, R_imag)

        recL = torch.istft(L, n_fft=int(n_fft), hop_length=int(hop),
                           win_length=int(win_length), window=window.float(),
                           center=center, length=length)
        recR = torch.istft(R, n_fft=int(n_fft), hop_length=int(hop),
                           win_length=int(win_length), window=window.float(),
                           center=center, length=length)
        out = torch.stack([recL, recR], dim=1)  # (B,2,T)
    return out

def make_alpha_bar(diffusion_steps: int, device, kind="poly",
                   power=3.0, beta=5.0, k=8.0):
    import math
    T = diffusion_steps
    t = torch.arange(T + 1, device=device, dtype=torch.float32)  # 0..T
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

    a[0] = 1.0   # clean
    a[-1] = 0.0  # fully reverberant
    return a


class EMAModel_DiT:
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


class EMAModel_UNet:
    def __init__(self, model: nn.Module, decay: float=0.995):
        self.decay = decay
        self.model = model
        self.shadow = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        self.backup = None

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


# ---- EMA wrapper ----
class EMAModel:
    def __init__(self, model: nn.Module, decay: float=0.995):
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


class ColdDiffInferencer:
    """
    Generic cold diffusion inferencer for RI spectrograms.
    Works for UNet or DiT, assuming model(x, t) is defined.
    """
    def __init__(self,
                 model: nn.Module,
                 model_type: str,
                 pre_params,
                 diffusion_steps: int,
                 alpha_mode: str,
                 cdiff_mode: str,
                 device: str):
        
        self.model = model.to(device)
        self.model_type = model_type #DiT or UNet
        self.pre_params = pre_params
        self.diffusion_steps = diffusion_steps
        self.alpha_mode = alpha_mode
        self.cdiff_mode = cdiff_mode # 'next_delta_norm', 'trust_region_delta', 'direct'
        self.device = device

        self.alpha_bar = make_alpha_bar(diffusion_steps, device=device, kind=alpha_mode)

        self.center = pre_params.center
        self.window = torch.hann_window(pre_params.win, periodic=True, device=device)

    def get_signal_from_RI_stft(self, ri_stft):
        n_fft = self.pre_params.fft
        hop = self.pre_params.hop
        win = self.pre_params.win
        length = getattr(self.pre_params, "wave_len", None)
        return istft_from_ri(ri_stft, n_fft=n_fft, hop=hop, win_length=win,
                             window=self.window, center=self.center, length=length)

    @torch.no_grad()
    def reverse_diffusion(self, inp_ri, step_stop=0):
        """
        inp_ri: (B,4,F,T) = x_T (reverberant).
        Returns list [x_T-1, ..., x_0].
        """
        x = inp_ri.to(self.device)
        B = x.shape[0]


        for t in range(self.diffusion_steps, step_stop, -1):
            T = torch.full((B,), t, device=self.device, dtype=torch.long)

            if self.cdiff_mode == "next_delta_norm":
                a_t = self.alpha_bar.index_select(0, T)
                a_tm1 = self.alpha_bar.index_select(0, T - 1)
                g = (a_tm1 - a_t).clamp_min(1e-6).view(-1, 1, 1, 1)
                if self.model_type == 'dit':
                    v, _ = self.model(x, T, sc=None)
                else:
                    v = self.model(x, T)
                x = x + g * v

            elif self.cdiff_mode == "trust_region_delta": #For DiT only
                a_t   = self.alpha_bar.index_select(0, T)          
                a_tm1 = self.alpha_bar.index_select(0, T - 1)      
                g = (a_tm1 - a_t).clamp_min(1e-6).view(-1,1,1,1) 

                v_hat, s = self.model(x, T, sc=None)
                s = s.view(-1,1,1,1)
                delta = (s * g) * v_hat

                # --- Trust-region bounding ---
                x_norm = x.norm(p=2, dim=(1,2,3), keepdim=True).clamp_min(1e-6)
                d_norm = delta.norm(p=2, dim=(1,2,3), keepdim=True).clamp_min(1e-6)
                scale  = (0.2 * x_norm / d_norm).clamp(max=1.0)
                delta = delta * scale
                x = x + delta


            elif self.cdiff_mode == "direct":
                # model directly predicts x_{t-1}
                x = self.model(x, T)

            else:
                raise ValueError(f"Unknown residual_mode: {self.residual_mode}")

        return x

    @torch.inference_mode()
    def dereverb_batch(self, reverb_ri):
        self.model.eval()
        use_amp = self.device.startswith("cuda")
        with torch.cuda.amp.autocast(enabled=use_amp):
            est_ri = self.reverse_diffusion(reverb_ri, step_stop=0)

        # bring back to float32 outside amp context for ISTFT + metrics
        est_ri = est_ri.float()
        est_wav = self.get_signal_from_RI_stft(est_ri)  # (B,2,T)
        return est_ri, est_wav
    

class ColdDiffInferencer_var:
    """
    Generic cold diffusion inferencer WITH VARIABLE DIFFUSION STEPS for RI spectrograms.
    Works for UNet or DiT, assuming model(x, t) is defined.
    """
    def __init__(self,
                 model: nn.Module,
                 model_type: str,
                 pre_params,
                 diffusion_steps: int,
                 reverse_steps: int,
                 solver: str,
                 alpha_mode: str,
                 cdiff_mode: str,
                 device: str):
        
        self.model = model.to(device)
        self.model_type = model_type #DiT or UNet
        self.pre_params = pre_params
        self.diffusion_steps = diffusion_steps 
        self.reverse_steps = reverse_steps #variable reverse steps
        self.solver = solver #euler or heun
        self.alpha_mode = alpha_mode
        self.cdiff_mode = cdiff_mode # ony works with 'next_delta_norm', 
        self.device = device


        self.center = pre_params.center
        self.window = torch.hann_window(pre_params.win, periodic=True, device=device)

    def get_signal_from_RI_stft(self, ri_stft):
        n_fft = self.pre_params.fft
        hop = self.pre_params.hop
        win = self.pre_params.win
        #length = getattr(self.pre_params, "wave_len", None)
        length = 88200
        return istft_from_ri(ri_stft, n_fft=n_fft, hop=hop, win_length=win,
                             window=self.window, center=self.center, length=length)
    
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
    
    def reverse_diffusion( #VARIABLE EDITION
        self,
        inp_ri: torch.Tensor,
        num_steps: int = 16,
        solver: str = "heun",          # "euler" or "heun"
        t_stop: float = 0.0,
        max_v: float = 4.0,          # clamp model update per step (start 3–6)
        max_x: float = 10.0,         # clamp state x (RI domain)
        rms_clamp_db: float = 6.0   # keep RI RMS within ±6 dB of initial per segment
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

        # Reference energy (per-segment) to prevent drift
        #rms0 = x.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)  # [B,1,1,1]
        #max_gain = 10 ** (rms_clamp_db / 20.0)
        #min_gain = 1.0 / max_gain

        # Continuous time grid: go from T_train -> 0 in num_steps
        ts = torch.linspace(
            float(self.diffusion_steps), float(t_stop),
            steps=num_steps + 1, device=device, dtype=torch.float32
        )

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
            #v1 = torch.nan_to_num(v1, nan=0.0, posinf=0.0, neginf=0.0)
            #if max_v is not None:
            #    v1 = v1.clamp(-max_v, max_v)

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
            
            # --- Guardrails on x ---
            #if max_x is not None:
            #    x = x.clamp(-max_x, max_x)

        # Keep RI RMS from drifting too far (prevents transient blow-ups)
        #rms = x.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)
        #scale = (rms0 / rms).clamp(min_gain, max_gain)
        #x = x * scale

        return x  

    @torch.inference_mode()
    def dereverb_batch(self, reverb_ri, reverse_steps: int = None):
        self.model.eval()
        use_amp = self.device.startswith("cuda")
        steps = self.reverse_steps if reverse_steps is None else int(reverse_steps)
        with torch.cuda.amp.autocast(enabled=use_amp):
            est_ri = self.reverse_diffusion(reverb_ri, num_steps=steps, solver=self.solver)

        # bring back to float32 outside amp context for ISTFT + metrics
        est_ri = est_ri.float()
        est_wav = self.get_signal_from_RI_stft(est_ri)  # (B,2,T)
        return est_ri, est_wav


class CDiffuseInferencer:
    """
    CDiffuSE-style Gaussian diffusion inferencer for RI spectrograms.

    Matches DDPMRIUNetTrainer settings:
      - linear β in [1e-4, 9.5e-3], T=diffusion_steps
      - UNet input: concat[x_t, reverb_ri] -> 8 channels
      - model predicts combined noise η_t
      - reverse diffusion = Algorithm 2 in CDiffuSE (see trainer).

      sampling_steps:
      - None or >= diffusion_steps  -> full sampling (evaluate model every step)
      - K < diffusion_steps         -> 'fast' sampling: model evaluated at K
                                      timesteps, ε̂ reused in between.
    """
    def __init__(self,
                 model: nn.Module,
                 pre_params,
                 diffusion_steps: int = 200,
                 sampling_steps: int | None = None,
                 device: str = "cuda"):
        self.model = model.to(device)
        self.pre_params = pre_params
        self.device = device
        self.diffusion_steps = diffusion_steps
        self.sampling_steps = sampling_steps  # None = full 200-step sampling

        # === same diffusion definitions as DDPMRIUNetTrainer === 
        self.betas = torch.linspace(1e-4, 9.5e-3, self.diffusion_steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)   # (T,)

        # CDiffuSE interpolation weights m_t and δ_t
        m = torch.sqrt(torch.clamp(1.0 - self.alpha_bar, min=1e-8))  # (T,)
        delta = (1.0 - self.alpha_bar) - (m ** 2) * self.alpha_bar
        delta = torch.clamp(delta, min=1e-12)

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

        # ISTFT setup
        self.center = pre_params.center
        self.window = torch.hann_window(pre_params.win, periodic=True, device=self.device)

    def get_signal_from_RI_stft(self, ri_stft):
        n_fft = self.pre_params.fft
        hop = self.pre_params.hop
        win = self.pre_params.win
        length = getattr(self.pre_params, "wave_len", None)
        return istft_from_ri(
            ri_stft, n_fft=n_fft, hop=hop, win_length=win,
            window=self.window, center=self.center, length=length
        )

    def _cdiff_coeffs(self, t: int):
        """
        Compute c_x, c_y, c_eps, delta_tilde for scalar t (1..T).
        Copied from DDPMRIUNetTrainer._cdiff_coeffs. 
        """
        device = self.device
        T = self.diffusion_steps
        assert 1 <= t <= T

        alpha_t       = self.alphas[t-1]                 # α_t
        alpha_bar_t   = self.alpha_bar_full[t]           # ᾱ_t
        alpha_bar_tm1 = self.alpha_bar_full[t-1]         # ᾱ_{t-1}
        m_t           = self.m_full[t]
        m_tm1         = self.m_full[t-1]
        delta_t       = self.delta_full[t]
        delta_tm1     = self.delta_full[t-1]

        if t > 1:
            delta_t_given_tm1 = delta_t - ((1.0 - m_t) / (1.0 - m_tm1))**2 * alpha_t * delta_tm1
        else:
            delta_t_given_tm1 = delta_t

        if t > 1 and delta_tm1 > 0:
            delta_tilde = delta_t_given_tm1 * delta_t / delta_tm1
        else:
            delta_tilde = torch.zeros_like(delta_t)

        sqrt_alpha_t       = torch.sqrt(alpha_t)
        inv_sqrt_alpha_t   = 1.0 / (sqrt_alpha_t + 1e-8)
        sqrt_alpha_bar_tm1 = torch.sqrt(alpha_bar_tm1)
        sqrt_one_minus_ab_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-8))

        c_x = ((1.0 - m_t) / (1.0 - m_tm1 + 1e-8)) * (delta_tm1 / (delta_t + 1e-8)) * sqrt_alpha_t \
              + (1.0 - m_tm1) * (delta_t_given_tm1 / (delta_t + 1e-8)) * inv_sqrt_alpha_t

        c_y = (m_tm1 * delta_t - m_t * (1.0 - m_t) / (1.0 - m_tm1 + 1e-8) * alpha_t * delta_tm1)
        c_y = c_y * (sqrt_alpha_bar_tm1 / (delta_t + 1e-8))

        c_eps = (1.0 - m_tm1) * (delta_t_given_tm1 / (delta_t + 1e-8)) \
                * sqrt_one_minus_ab_t * sqrt_alpha_t

        return c_x.to(device), c_y.to(device), c_eps.to(device), delta_tilde.to(device)

    @torch.no_grad()
    def reverse_diffusion(self, reverb_ri, step_stop=0):
        """
        CDiffuSE sampling (Algorithm 2):
          - Start from x_T ~ N(sqrt(ᾱ_T) y, δ_T I)
          - Iterate t = T..1: x_{t-1} ~ N( c_x x_t + c_y y - c_eps eps_theta, \tilde δ_t I )

        If self.sampling_steps is None or >= diffusion_steps:
            -> 'official' 200-step sampling (model evaluated at every t).

        If self.sampling_steps = K < diffusion_steps:
            -> fast sampling: choose K evaluation timesteps between T and 1,
               reuse ε̂ for intermediate t. This reduces UNet calls from T to K.
        """
        self.model.eval()
        y = reverb_ri.to(self.device)
        B = y.shape[0]
        T_steps = self.diffusion_steps

        # --- initial sample x_T ~ N(sqrt(ᾱ_T) y, δ_T I) ---
        alpha_bar_T = self.alpha_bar_full[T_steps]
        delta_T     = self.delta_full[T_steps]

        sqrt_ab_T    = torch.sqrt(alpha_bar_T)
        sqrt_delta_T = torch.sqrt(torch.clamp(delta_T, min=1e-12))

        eps_T = torch.randn_like(y)
        x = sqrt_ab_T * y + sqrt_delta_T * eps_T

        # --- choose at which timesteps we recompute eps_hat ---
        K = self.sampling_steps
        if (K is None) or (K >= T_steps):
            # full / official sampling: evaluate at every step
            eval_steps = set(range(T_steps, step_stop, -1))
        else:
            # pick K timesteps evenly spaced between T_steps and step_stop+1
            # (inclusive), rounded to integers
            eval_ts = torch.linspace(
                float(T_steps),
                float(max(step_stop + 1, 1)),
                steps=K,
            ).round().long().unique()
            eval_steps = set(int(t.item()) for t in eval_ts)
            # make sure we always evaluate at very first step
            eval_steps.add(T_steps)
            eval_steps.add(max(step_stop + 1, 1))

        eps_hat = None  # last predicted noise

        # --- reverse process ---
        for t in range(T_steps, step_stop, -1):
            T_batch = torch.full((B,), t, device=self.device, dtype=torch.long)

            # recompute eps_hat only at selected timesteps
            if (eps_hat is None) or (t in eval_steps):
                x_in = torch.cat([x, y], dim=1)  # (B,8,F,T)
                eps_hat = self.model(x_in, T_batch)

            c_x, c_y, c_eps, delta_tilde = self._cdiff_coeffs(t)

            c_x   = c_x.view(1, 1, 1, 1)
            c_y   = c_y.view(1, 1, 1, 1)
            c_eps = c_eps.view(1, 1, 1, 1)

            mean = c_x * x + c_y * y - c_eps * eps_hat

            if t > 1 and delta_tilde > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(torch.clamp(delta_tilde, min=1e-12)) * noise
            else:
                x = mean

        return x

    @torch.inference_mode()
    def dereverb_batch(self, reverb_ri):
        self.model.eval()

        use_amp = self.device.startswith("cuda")
        with torch.cuda.amp.autocast(enabled=use_amp):
            est_ri = self.reverse_diffusion(reverb_ri, step_stop=0)

        # bring back to float32 outside amp context for ISTFT + metrics
        est_ri = est_ri.float()
        est_wav = self.get_signal_from_RI_stft(est_ri)
        return est_ri, est_wav


def match_rms_to_ref(x: torch.Tensor,
                     ref: torch.Tensor,
                     eps: float = 1e-8) -> torch.Tensor:
    """
    Scale x so that its RMS matches ref RMS, per example.

    x, ref: (B, C, T)
    Returns: scaled x with same RMS as ref for each batch item.
    """
    # RMS over channels and time
    ref_rms = torch.sqrt((ref ** 2).mean(dim=(1, 2), keepdim=True) + eps)  # (B,1,1)
    x_rms   = torch.sqrt((x   ** 2).mean(dim=(1, 2), keepdim=True) + eps)

    # avoid division by zero
    gain = torch.where(ref_rms > 0, ref_rms / x_rms, torch.ones_like(ref_rms))
    return x * gain


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

# -------------------------------------------------------------------------
# SGMSE+ inference (UNetRI backbone, stereo RI-STFT)
# -------------------------------------------------------------------------

class SGMSEInferencer:
    """
    SGMSE+-style score-based inferencer for RI spectrograms.

    - Uses the same OU-VE SDE as in train_sgmse_unet_pt.SGMSESDE
    - Reverse-time sampler = predictor–corrector (Langevin + Euler–Maruyama)
    - num_steps can be < diffusion_steps for faster sampling (e.g. 16 vs 30)
    """
    def __init__(
        self,
        model: nn.Module,
        pre_params,
        device: str = "cuda",
        diffusion_steps: int = 30,
        sigma_min: float = 0.1,
        sigma_max: float = 1.0,
        theta: float = 2.0,
        t_eps: float = 1e-3,
        num_steps: int | None = None,
        snr: float = 0.5,
        n_corr_steps: int = 1,
    ):
        self.model = model.to(device)
        self.pre_params = pre_params
        self.device = device

        # SDE hyper-parameters must match training
        self.sde = SGMSESDE(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            theta=theta,
            N=diffusion_steps,
            t_eps=t_eps,
            device=device,
        )

        # effective number of reverse steps (≤ diffusion_steps)
        self.num_steps = num_steps if num_steps is not None else diffusion_steps
        self.snr = snr
        self.n_corr_steps = n_corr_steps

        # ISTFT params
        self.center = pre_params.center
        self.window = torch.hann_window(pre_params.win, periodic=True, device=self.device)

    # ---- ISTFT helper: same signature as other inferencers ----
    def get_signal_from_RI_stft(self, ri_stft: torch.Tensor) -> torch.Tensor:
        n_fft = self.pre_params.fft
        hop = self.pre_params.hop
        win = self.pre_params.win
        length = getattr(self.pre_params, "wave_len", None)
        return istft_from_ri(
            ri_stft,
            n_fft=n_fft,
            hop=hop,
            win_length=win,
            window=self.window,
            center=self.center,
            length=length,
        )

    # ---- reverse diffusion (sampling, SGMSE⁺-style ODE) ----
    @torch.no_grad()
    def reverse_diffusion(
        self,
        reverb_ri: torch.Tensor,
        num_steps: int | None = None,
        return_all: bool = False,
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

    # ---- core PC sampler (copied from trainer.reverse_diffusion_pc, but modular) ----
    @torch.no_grad()
    def reverse_diffusion_pc(
        self,
        reverb_ri: torch.Tensor,
        num_steps: int | None = None,
        snr: float | None = None,
        n_corr_steps: int | None = None,
        return_all: bool = False,
        use_tweedie_x0: bool = True,   #return x0 projection instead of x_t
        final_corrector: bool = True,  #do an extra corrector at t=t_eps
    ):
        """
        Predictor–corrector sampler:

          - x_T ~ N(y, std(T)^2 I)
          - N uniform time steps from T -> t_eps
          - At each step: n_corr_steps Langevin corrector + Euler–Maruyama predictor
        """
        device = self.device
        sde = self.sde
        B = reverb_ri.size(0)

        N = num_steps if num_steps is not None else self.num_steps
        n_corr = n_corr_steps if n_corr_steps is not None else self.n_corr_steps

        T = sde.T
        t_eps = sde.t_eps

        # initial sample x_T ~ p_T(x | y)
        x = sde.prior_sampling(reverb_ri.shape, reverb_ri)  # (B,4,F,T)

        # store denoised states (x_mean) for inspection
        xs = []

        # time grid with N intervals: T -> t_eps
        # (N+1 points => dt = t[i+1] - t[i] consistent)
        t_grid = torch.linspace(T, t_eps, steps=N + 1, device=device)

        # score model wrapper: UNet -> score(x_t, t, y)
        def score_fn(x_t: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 0:
                t = t[None]
            dnn_in = torch.cat([x_t, y], dim=1)  # concat[x_t, y] -> 8ch
            return self.model(dnn_in, t)
        
        def langevin_corrector_step(x_in: torch.Tensor, t_in: torch.Tensor):
            """
            Your current Langevin corrector (norm-based step size).
            Returns: (x, x_mean)
            """
            x_loc = x_in
            x_mean_loc = x_in
            target_snr = snr if snr is not None else self.snr

            for _ in range(n_corr):
                grad = score_fn(x_loc, reverb_ri, t_in)      # (B,4,F,T)
                noise = torch.randn_like(x_loc)

                # NOTE: you currently use batch-mean norms; keep identical behavior
                grad_norm = torch.norm(grad.reshape(B, -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(B, -1), dim=-1).mean()

                step_size = ((target_snr * noise_norm / (grad_norm + 1e-12)) ** 2) * 2.0
                step_size = step_size.view(1, 1, 1, 1)

                x_mean_loc = x_loc + step_size * grad
                x_loc = x_mean_loc + torch.sqrt(2.0 * step_size) * noise

            return x_loc, x_mean_loc

        # ---------------- main PC loop ----------------
        for i in range(N):
            t_i = t_grid[i].expand(B)         # (B,)
            t_next = t_grid[i + 1].expand(B)  # (B,)

            # -------- CORRECTOR: Langevin steps --------
            x, x_mean = langevin_corrector_step(x, t_i)

            # -------- PREDICTOR: Euler–Maruyama step --------
            f, g = sde.sde(x, reverb_ri, t_i)      # f: (B,4,F,T), g: (B,)
            g_b = g.view(B, 1, 1, 1)
            score = score_fn(x, reverb_ri, t_i)    # (B,4,F,T)
            # reverse SDE drift: f - g^2 * score
            drift = f - (g_b ** 2) * score

            # consistent dt for this interval (negative)
            dt = (t_next - t_i).view(B, 1, 1, 1)    # (B,1,1,1), dt < 0
            sqrt_dt = torch.sqrt(-dt)               # sqrt(|dt|)

            x_mean = x + drift * dt
            x = x_mean + g_b * sqrt_dt * torch.randn_like(x)

            xs.append(x_mean.detach().clone())

        # ---------------- (2) final return-state fix: extra corrector at t_eps ----------------
        if final_corrector:
            t_final = torch.full((B,), t_eps, device=device, dtype=reverb_ri.dtype)
            # Apply corrector once more on the last denoised state (x_mean is a good starting point)
            x, x_mean = langevin_corrector_step(x_mean, t_final)

        # ---------------- (3) OU Tweedie-style x0 projection ----------------
        if use_tweedie_x0:
            t_final = torch.full((B,), t_eps, device=device, dtype=reverb_ri.dtype)
            score_final = score_fn(x_mean, reverb_ri, t_final)  # s_theta(x_t,t,y)

            # OU bridge parameters:
            # alpha(t) = exp(-theta * t)
            alpha = torch.exp(-sde.theta * t_final).view(B, 1, 1, 1)  # (B,1,1,1)

            # sigma(t) from the closed-form OU-VE marginal std
            sigma = sde._std(t_final).view(B, 1, 1, 1)                # (B,1,1,1)
            sigma2 = sigma ** 2

            # x0_hat ≈ (x_t + sigma^2 * score - (1 - alpha) * y) / alpha
            denom = torch.clamp(alpha, min=1e-5)
            x0_hat = (x_mean + sigma2 * score_final - (1.0 - alpha) * reverb_ri) / denom
            final_out = x0_hat
        else:
            final_out = x_mean

        if return_all:
            xs.append(final_out.detach().clone())
            return xs
        else:
            return final_out

    @torch.no_grad()
    def reverse_diffusion_pc_predcorr(
        self,
        reverb_ri: torch.Tensor,
        num_steps: int | None = None,
        snr: float | None = None,
        n_corr_steps: int | None = None,
        return_all: bool = False,
        use_tweedie_x0: bool = True,   # return x0 projection instead of x_t
        final_corrector: bool = False, # NOTE: with Pred->Corr, last step already corrects at t_eps
    ):
        """
        Predictor–Corrector sampler (official order):
        - x_T ~ N(y, std(T)^2 I)
        - N uniform time steps from T -> t_eps
        - At each step: Predictor (Euler–Maruyama) then Corrector (Langevin)
        - Optional: extra final corrector at t_eps
        - Optional: final OU Tweedie x0 projection
        """
        device = self.device
        sde = self.sde
        B = reverb_ri.size(0)

        N = num_steps if num_steps is not None else self.num_steps
        n_corr = n_corr_steps if n_corr_steps is not None else self.n_corr_steps

        T = float(sde.T)
        t_eps = float(sde.t_eps)

        # initial sample x_T ~ p_T(x | y)
        x = sde.prior_sampling(reverb_ri.shape, reverb_ri).to(device)

        xs = []

        # time grid with N intervals: T -> t_eps
        t_grid = torch.linspace(T, t_eps, steps=N + 1, device=device, dtype=reverb_ri.dtype)

        # score model wrapper: UNet -> score(x_t, t, y)
        def score_fn(x_t: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 0:
                t = t[None]
            dnn_in = torch.cat([x_t, y], dim=1)  # concat[x_t, y] -> 8ch
            return self.model(dnn_in, t)

        def langevin_corrector_step(x_in: torch.Tensor, t_in: torch.Tensor):
            """
            Your current Langevin corrector (norm-based step size).
            Returns: (x, x_mean)
            """
            x_loc = x_in
            x_mean_loc = x_in
            target_snr = snr if snr is not None else self.snr

            for _ in range(n_corr):
                grad = score_fn(x_loc, reverb_ri, t_in)      # (B,4,F,T)
                noise = torch.randn_like(x_loc)

                grad_norm = torch.norm(grad.reshape(B, -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(B, -1), dim=-1).mean()

                step_size = ((target_snr * noise_norm / (grad_norm + 1e-12)) ** 2) * 2.0
                step_size = step_size.view(1, 1, 1, 1)

                x_mean_loc = x_loc + step_size * grad
                x_loc = x_mean_loc + torch.sqrt(2.0 * step_size) * noise

            return x_loc, x_mean_loc

        # ---------------- main PC loop (Predictor -> Corrector) ----------------
        for i in range(N):
            t_i = t_grid[i].expand(B)         # (B,)
            t_next = t_grid[i + 1].expand(B)  # (B,)

            # -------- PREDICTOR: Euler–Maruyama step (reverse SDE) --------
            f, g = sde.sde(x, reverb_ri, t_i)     # f: (B,4,F,T), g: (B,)
            g_b = g.view(B, 1, 1, 1)
            score = score_fn(x, reverb_ri, t_i)   # (B,4,F,T)

            # reverse drift: f - g^2 * score
            drift = f - (g_b ** 2) * score

            dt = (t_next - t_i).view(B, 1, 1, 1)   # dt < 0
            sqrt_dt = torch.sqrt(-dt)

            x_mean_pred = x + drift * dt
            x = x_mean_pred + g_b * sqrt_dt * torch.randn_like(x)

            # -------- CORRECTOR: Langevin steps at t_next --------
            # Standard PC corrects the *sample* x at the new time.
            # x, x_mean = langevin_corrector_step(x, t_next)
            x, x_mean = langevin_corrector_step(x_mean_pred, t_next)  #Variant B: correct the predictor mean (often even less hiss)

            xs.append(x_mean.detach().clone())

        # ---------------- optional extra final corrector at t_eps ----------------
        # With Pred->Corr loop, you already corrected at t_eps in the last iteration.
        if final_corrector:
            t_final = torch.full((B,), t_eps, device=device, dtype=reverb_ri.dtype)
            x, x_mean = langevin_corrector_step(x_mean, t_final)

        # ---------------- OU Tweedie-style x0 projection ----------------
        if use_tweedie_x0:
            t_final = torch.full((B,), t_eps, device=device, dtype=reverb_ri.dtype)
            score_final = score_fn(x_mean, reverb_ri, t_final)

            alpha = torch.exp(-sde.theta * t_final).view(B, 1, 1, 1)   # exp(-theta t)
            sigma = sde._std(t_final).view(B, 1, 1, 1)
            sigma2 = sigma ** 2

            denom = torch.clamp(alpha, min=1e-5)
            x0_hat = (x_mean + sigma2 * score_final - (1.0 - alpha) * reverb_ri) / denom
            final_out = x0_hat
        else:
            final_out = x_mean

        if return_all:
            xs.append(final_out.detach().clone())
            return xs
        else:
            return final_out


    @torch.inference_mode()
    def dereverb_batch(self, reverb_ri: torch.Tensor):
        """
        Entry point used by run_metrics:

          reverb_ri: (B,4,F,T)
          returns (est_ri, est_wav)
        """
        self.model.eval()
        use_amp = self.device.startswith("cuda")
        with torch.cuda.amp.autocast(enabled=use_amp):
            # est_ri = self.reverse_diffusion_pc(reverb_ri, num_steps=self.num_steps) #PC (Corrector Predictor)
            est_ri = self.reverse_diffusion_pc_predcorr(reverb_ri, num_steps=self.num_steps) #PC (Predictor Corrector)
            # est_ri = self.reverse_diffusion(reverb_ri, num_steps=50) #ODE
        est_ri = est_ri.float()
        est_wav = self.get_signal_from_RI_stft(est_ri)
        return est_ri, est_wav
    

# FOR GRADIO USE


def ensure_float_audio(x: np.ndarray) -> np.ndarray:
    """Convert int audio to float32 in [-1, 1]. Keep float as float32."""
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32)
        peak = np.max(np.abs(x)) if x.size else 0.0
        if peak > 0:
            x = x / peak
    else:
        x = x.astype(np.float32, copy=False)
        # if user uploaded float but outside [-1,1], normalize defensively
        peak = np.max(np.abs(x)) if x.size else 0.0
        if peak > 1.5:
            x = x / peak
    return x

def ensure_stereo(x: np.ndarray) -> np.ndarray:
    """
    Ensure output is shape [T, 2].
    - [T]      -> [T, 2] (duplicate)
    - [T, 1]   -> [T, 2] (duplicate)
    - [T, 2]   -> unchanged
    - [T, C>2] -> downmix to mono then duplicate
    """
    if x.ndim == 1:
        return np.stack([x, x], axis=-1)

    if x.ndim != 2:
        raise ValueError(f"Audio must be 1D or 2D, got shape {x.shape}")

    T, C = x.shape
    if C == 1:
        return np.concatenate([x, x], axis=1)
    if C == 2:
        return x
    # C > 2
    mono = np.mean(x, axis=1, dtype=np.float32)
    return np.stack([mono, mono], axis=-1)

def resample_audio(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Stereo-safe resampling along time axis using resample_poly."""
    if sr_in == sr_out:
        return x
    g = math.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    # axis=0 is time
    return signal.resample_poly(x, up, down, axis=0).astype(np.float32, copy=False)

def trim_or_pad_range(x: np.ndarray, sr: int, min_s: float, max_s: float) -> np.ndarray:
    """
    Enforce duration in [min_s, max_s].
    - if longer than max: trim
    - if shorter than min: pad zeros
    Expects shape [T] or [T, C].
    """
    min_len = int(round(min_s * sr))
    max_len = int(round(max_s * sr))

    T = x.shape[0]
    if T > max_len:
        x = x[:max_len, ...]
        T = x.shape[0]

    if T < min_len:
        pad = min_len - T
        if x.ndim == 1:
            x = np.pad(x, (0, pad))
        else:
            x = np.pad(x, ((0, pad), (0, 0)))
    return x


def set_loudness(data, rate, LUFS=-28.0):

    # measure the loudness first
    meter = pyln.Meter(rate)  # create BS.1770 meter
    loudness = meter.integrated_loudness(data)

    # loudness normalize audio to -28 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, LUFS)

    return loudness_normalized_audio

def segment_audio_torch(audio_np: np.ndarray,
                        sr: int,
                        ts_min: float = 2.0,
                        overlap: float = 0.5,
                        pad_end: bool = True,
                        device: str = "cpu"):

    # audio_np is [T, 2] float32
    x = torch.from_numpy(audio_np).to(device=device, dtype=torch.float32)  # [T, 2]
    x = x.transpose(0, 1).contiguous()  # [2, T]
    C, T = x.shape
    L = int(round(ts_min * sr))
    step = max(int(round((1.0 - overlap) * L)), 1)
    orig_len = T

    if not pad_end:
        if T < L:
            return x.new_zeros((0, C, L)), step, orig_len
        starts = list(range(0, T - L + 1, step))
        segs = [x[:, s:s+L] for s in starts]
        return torch.stack(segs, dim=0), step, orig_len  # [N, C, L]

    n_segs = max(1, int(np.ceil(max(T - L, 0) / step)) + 1)
    segs = []
    for i in range(n_segs):
        s = i * step
        e = s + L
        seg = x[:, s:min(e, T)]
        if seg.shape[1] < L:
            seg = torch.nn.functional.pad(seg, (0, L - seg.shape[1]))
        segs.append(seg)
    return torch.stack(segs, dim=0), step, orig_len  # [N, C, L]

def ola_reconstruct_torch(segs: torch.Tensor, step: int, orig_len: int):
    """
    segs: [N, C, L] time-domain segments
    Returns: [T, C] torch
    """
    N, C, L = segs.shape
    w = torch.hann_window(L, periodic=True, device=segs.device, dtype=segs.dtype)  # [L]
    w = (w ** 2)  # stronger edge suppression
    segs_w = segs * w.view(1, 1, L)

    out_len = step * (N - 1) + L
    y = segs.new_zeros((C, out_len))
    norm = segs.new_zeros((out_len,))

    for i in range(N):
        s = i * step
        e = s + L
        y[:, s:e] += segs_w[i]
        norm[s:e] += w

    nz = norm > 1e-12
    y[:, nz] = y[:, nz] / norm[nz].unsqueeze(0)

    y = y[:, :orig_len]          # [C, T]
    return y.transpose(0, 1)     # [T, C]

ArrayLike = Union[np.ndarray, torch.Tensor]

def audio_to_stereo_ri_stft(
    wav: ArrayLike,
    config=None,
    *,
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[torch.Tensor] = None,
    device: Optional[Union[str, torch.device]] = None,
    center: bool = True,
) -> torch.Tensor:
    """
    Convert stereo audio to RI STFT.

    Output packing:
      stereo RI -> channels (4, F, TT): [L_R, L_I, R_R, R_I]

    Accepted input shapes:
      - (T, 2) or (2, T)
      - (B, T, 2) or (B, 2, T)

    Returns:
      - if input is single segment -> (4, F, TT)
      - if input is batched        -> (B, 4, F, TT)
    """

    # ---- infer params from config if provided ----
    if config is not None:
        if n_fft is None:
            n_fft = int(config.fft)
        if hop_length is None:
            hop_length = int(config.hop)
        if win_length is None:
            win_length = int(config.win)

    if n_fft is None or hop_length is None or win_length is None:
        raise ValueError("Provide config with (fft, hop, win) or set n_fft/hop_length/win_length explicitly.")

    dev = torch.device(device) if device is not None else (
        wav.device if isinstance(wav, torch.Tensor) else torch.device("cpu")
    )

    # ---- to torch float32 ----
    x = torch.as_tensor(wav, dtype=torch.float32, device=dev)

    # ---- normalize shapes to (B, 2, T) ----
    single = False

    if x.ndim == 2:
        single = True
        # (T,2) -> (1,2,T)
        if x.shape[-1] == 2:
            x = x.transpose(0, 1).unsqueeze(0).contiguous()
        # (2,T) -> (1,2,T)
        elif x.shape[0] == 2:
            x = x.unsqueeze(0).contiguous()
        else:
            raise ValueError(f"2D audio must be (T,2) or (2,T). Got {tuple(x.shape)}")

    elif x.ndim == 3:
        # (B,T,2) -> (B,2,T)
        if x.shape[-1] == 2:
            x = x.permute(0, 2, 1).contiguous()
        # already (B,2,T)
        elif x.shape[1] == 2:
            x = x.contiguous()
        else:
            raise ValueError(f"3D audio must be (B,T,2) or (B,2,T). Got {tuple(x.shape)}")

    else:
        raise ValueError(f"Audio must be 2D or 3D. Got {tuple(x.shape)}")

    B, C, T = x.shape
    if C != 2:
        raise ValueError(f"Expected stereo with C=2 after formatting, got C={C}.")

    # ---- window (prefer config.window_tensor if available) ----
    if window is None:
        if config is not None and hasattr(config, "window_tensor") and callable(getattr(config, "window_tensor")):
            window = config.window_tensor(device=str(dev))
        else:
            window = torch.hann_window(win_length, periodic=True, device=dev, dtype=torch.float32)
    else:
        window = window.to(device=dev, dtype=torch.float32)

    # ---- batched STFT: reshape (B,2,T) -> (B*2, T) ----
    x2 = x.reshape(B * 2, T)

    X = torch.stft(
        x2,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        return_complex=True,
    )  # (B*2, F, TT) complex

    # back to (B,2,F,TT)
    F, TT = X.shape[-2], X.shape[-1]
    X = X.view(B, 2, F, TT)

    real = X.real
    imag = X.imag

    # pack as (B,4,F,TT): [L_R, L_I, R_R, R_I]
    ri = torch.stack(
        [real[:, 0], imag[:, 0], real[:, 1], imag[:, 1]],
        dim=1
    ).to(torch.float32)

    return ri[0] if single else ri


def center_stitch_from_segments(mid_chunks: list[torch.Tensor], step: int, orig_len: int) -> torch.Tensor:
    """
    mid_chunks: list of torch tensors [2, step] on CPU (recommended)
    step: hop size in samples
    orig_len: target output length in samples

    Returns: torch.Tensor [T, 2] on CPU
    """
    if len(mid_chunks) == 0:
        return torch.zeros((orig_len, 2), dtype=torch.float32)

    # mid_chunks are [2, step]
    y_len = step * len(mid_chunks)
    y = torch.zeros((2, y_len), dtype=mid_chunks[0].dtype)

    for i, c in enumerate(mid_chunks):
        s = i * step
        y[:, s:s + step] = c

    # Trim to original length
    y = y[:, :orig_len]                 # [2, T]
    return y.transpose(0, 1).contiguous()  # [T, 2]

def generate_spectrogram(
    audio_data,
    sr,
    *,
    plot_sr: int = 44100,
    n_fft: int = 4096,
    hop: int = 256,
    fmin: float = 20.0,
    fmax: float = 12000.0,
    db_range: float = 80.0,        # contrast control (60–100 typical)
    cmap: str = "magma",           # "inferno", "magma", "turbo", "viridis"
    gamma: float = 0.85,           # <1.0 reveals quiet details; 1.0 linear
    interpolation: str = "nearest",
):
    """
    audio_data: numpy [T] or [T,2] in [-1,1]
    Returns: numpy uint8 image (H,W,3)
    """

    x = audio_data
    if x.ndim == 2:  # stereo -> mono for visualization
        x = x.mean(axis=1)


    # downsample for speed (only for plotting)
    if sr != plot_sr:
        x = signal.resample_poly(x, plot_sr, sr).astype(np.float32, copy=False)
        sr = plot_sr

    # STFT (scipy)
    f, t, Zxx = signal.stft(
        x,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        window="hann",
        padded=False,
        boundary=None,
    )
    S = np.abs(Zxx) + 1e-8
    S_db = 20.0 * np.log10(S)
    S_db -= S_db.max()  # peak at 0 dB

    # clamp dB range for contrast
    S_db = np.maximum(S_db, -db_range)

    # freq range
    fmax_eff = min(fmax, sr / 2)
    mask = (f >= fmin) & (f <= fmax_eff)
    f = f[mask]
    S_db = S_db[mask, :]

    # render fast with imshow 
    fig, ax = plt.subplots(figsize=(10, 4), dpi=140)
    norm = PowerNorm(gamma=gamma, vmin=-db_range, vmax=0.0)
    im = ax.imshow(
        S_db,
        origin="lower",
        aspect="auto",
        extent=[t[0] if len(t) else 0, t[-1] if len(t) else 0, f[0] if len(f) else fmin, f[-1] if len(f) else fmax_eff],
        cmap=cmap,
        norm=norm,
        interpolation=interpolation,
    )
    ax.set_yscale("log")
    ax.set_ylim(max(fmin, 1.0), fmax_eff)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("dB (relative to peak)")
    fig.tight_layout()

    img = fig_to_rgb_numpy(fig)
    plt.close(fig)
    return img

def fig_to_rgb_numpy(fig) -> np.ndarray:
    """Return an (H,W,3) uint8 RGB image from a Matplotlib figure, robust across backends."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # Prefer RGB if available
    if hasattr(fig.canvas, "tostring_rgb"):
        buf = fig.canvas.tostring_rgb()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        return img

    # Fallback: ARGB (4 channels)
    if hasattr(fig.canvas, "tostring_argb"):
        buf = fig.canvas.tostring_argb()
        argb = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)  # A,R,G,B
        rgb = argb[:, :, 1:4]  # drop alpha -> R,G,B
        return rgb

    # Last-resort: use buffer_rgba if present
    if hasattr(fig.canvas, "buffer_rgba"):
        rgba = np.asarray(fig.canvas.buffer_rgba())  # (h,w,4)
        return rgba[:, :, :3].copy()

    raise RuntimeError("Matplotlib canvas does not support tostring_rgb/tostring_argb/buffer_rgba.")