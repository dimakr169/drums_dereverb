import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def sisdr_per_sample(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant SDR (dB), computed per sample.
      estimate, target: (B, C, T) or (B, T)
    Returns: (B,) tensor of SI-SDR in dB
    """
    if estimate.dim() == 2: estimate = estimate.unsqueeze(1)  # (B,1,T)
    if target.dim()   == 2: target   = target.unsqueeze(1)

    # projection of estimate onto target
    dot = torch.sum(estimate * target, dim=(-2, -1))                 # (B,)
    den = torch.sum(target * target,   dim=(-2, -1)) + eps           # (B,)
    scale = dot / den                                                 # (B,)

    s_target = scale.view(-1, 1, 1) * target                          # (B,C,T)
    e_noise  = estimate - s_target                                    # (B,C,T)

    num = torch.sum(s_target * s_target, dim=(-2, -1))               # (B,)
    den = torch.sum(e_noise  * e_noise,  dim=(-2, -1)) + eps         # (B,)

    ratio = num / den
    return 10.0 * torch.log10(torch.clamp(ratio, min=eps))           # (B,)

class SISDR(nn.Module):
    """Running mean SI-SDR (dB). Call update(...) per batch, result() at epoch end."""
    def __init__(self, name: str = "si_sdr"):
        super().__init__()
        self.name = name
        self.reset()

    @torch.no_grad()
    def update(self, target: torch.Tensor, estimate: torch.Tensor):
        vals = sisdr_per_sample(estimate, target)   # (B,)
        self.sum += float(vals.sum())
        self.n   += int(vals.numel())

    def result(self) -> float:
        return self.sum / max(1, self.n)

    def reset(self):
        self.sum = 0.0
        self.n   = 0


class SISDRi(nn.Module):
    """
    SI-SDR improvement (dB): SI-SDR(enhanced, target) − SI-SDR(input, target).
    Use update(target, enhanced, input_reverb).
    """
    def __init__(self, name: str = "si_sdri"):
        super().__init__()
        self.name = name
        self.reset()

    @torch.no_grad()
    def update(self, target: torch.Tensor, enhanced: torch.Tensor, input_reverb: torch.Tensor):
        sdr_enh = sisdr_per_sample(enhanced,     target)  # (B,)
        sdr_in  = sisdr_per_sample(input_reverb, target)  # (B,)
        vals = sdr_enh - sdr_in
        self.sum += float(vals.sum())
        self.n   += int(vals.numel())

    def result(self) -> float:
        return self.sum / max(1, self.n)

    def reset(self):
        self.sum = 0.0
        self.n   = 0


    # ==========================================
# 3) Normalized Mutual Information (NMI) loss
# ==========================================
@dataclass
class NMILossConfig:
    bins: int = 256
    window_length: int = 1024
    hop_length: int = 512
    use_db_scale: bool = False


class NormalizedMutualInformationLoss(nn.Module):
    def __init__(self, cfg: NMILossConfig = NMILossConfig()):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def _to_db(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return 20.0 * torch.log10(torch.clamp(x, min=eps))

    @staticmethod
    def _frame_unfold(x: torch.Tensor, win: int, hop: int) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        T = x.size(-1)
        pad = (0, (-(T - win) % hop) % hop) if T >= win else (0, win - T)
        if pad[1] > 0:
            x = F.pad(x, pad, mode="constant", value=0.0)
        return x.unfold(dimension=-1, size=win, step=hop)  # (B,C,n_frames,win)

    @staticmethod
    def _discretize_per_window(xw: torch.Tensor, bins: int, eps: float = 1e-7) -> torch.Tensor:
        x_min = xw.amin(dim=-1, keepdim=True)
        x_max = xw.amax(dim=-1, keepdim=True)
        denom = torch.clamp(x_max - x_min, min=eps)
        xn = (xw - x_min) / denom
        idx = torch.clamp((xn * bins).floor().long(), 0, bins - 1)
        return idx

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        # Force the whole computation to fp32 for numerical stability
        with torch.cuda.amp.autocast(enabled=False):
            y_true = y_true.float()
            y_pred = y_pred.float()

            if y_true.dim() == 2: y_true = y_true.unsqueeze(1)
            if y_pred.dim() == 2: y_pred = y_pred.unsqueeze(1)

            if self.cfg.use_db_scale:
                y_true = self._to_db(y_true)
                y_pred = self._to_db(y_pred)

            win  = int(self.cfg.window_length)
            hop  = int(self.cfg.hop_length)
            bins = int(self.cfg.bins)

            wt = self._frame_unfold(y_true, win, hop)  # (B,C,F,win)
            wp = self._frame_unfold(y_pred, win, hop)

            M = wt.shape[0] * wt.shape[1] * wt.shape[2]
            wt = wt.reshape(M, win)
            wp = wp.reshape(M, win)

            lt = self._discretize_per_window(wt, bins)  # (M, win)
            lp = self._discretize_per_window(wp, bins)

            lin = (lt * bins + lp).view(M, -1)  # (M, win)

            # Global contingency (simple & fast). Add tiny Laplace smoothing.
            counts = torch.bincount(lin.flatten(), minlength=bins * bins).float()
            counts = counts.view(bins, bins)
            counts = counts + 1e-3  # smoothing to avoid zeros

            total = counts.sum().clamp_min(1.0)
            P_ij = counts / total
            P_i  = P_ij.sum(dim=1, keepdim=True)  # (bins,1)
            P_j  = P_ij.sum(dim=0, keepdim=True)  # (1,bins)

            # Stable MI
            # Use eps that is representable in fp32 (>= 1e-7 is plenty)
            eps = 1e-7
            denom = (P_i @ P_j).clamp_min(eps)  # (bins,bins)
            ratio = (P_ij / denom).clamp_min(eps)  # avoid log(0)
            MI = torch.sum(P_ij * torch.log(ratio))

            # Entropies
            def _H(p):
                p = p.clamp_min(eps)
                return -torch.sum(p * torch.log(p))
            H_t = _H(P_i.squeeze(1))
            H_p = _H(P_j.squeeze(0))

            nmi = 2.0 * MI / (H_t + H_p + eps)
            return 1.0 - nmi
