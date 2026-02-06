# metrics_audio.py

from __future__ import annotations
import math
from typing import Dict, List, Tuple
from typing import Any
import librosa
import numpy as np
import torch
import torch.nn.functional as F


# ======================================
#  Helpers
# ======================================

def _l1_per_example(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, C, T) -> L1 averaged over C,T for each example: (B,)
    """
    return x.abs().mean(dim=(-1, -2))


def _flatten_per_example(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, ...) -> (B, N) flattened over all but batch.
    """
    return x.reshape(x.size(0), -1)


# ======================================
#  Time-domain metrics
# ======================================

def audio_mae(est: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    Mean absolute error in time domain per example.
    est, tgt: (B, C, T)
    Returns: (B,)
    """
    return _l1_per_example(est - tgt)


def esr(est: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Error-to-Signal Ratio (linear), per example:
        ESR = ||e||^2 / ||s||^2
    est, tgt: (B, C, T)
    Returns: (B,)
    """
    err = est - tgt
    num = (err ** 2).sum(dim=(-1, -2))
    den = (tgt ** 2).sum(dim=(-1, -2)).clamp_min(eps)
    return num / den


def si_sdr(est: torch.Tensor,
           tgt: torch.Tensor,
           eps: float = 1e-8) -> torch.Tensor:
    """
    Standard scale-invariant SDR (dB), averaged over channels.
    est, tgt: (B, C, T)
    Returns: (B,)
    """
    B, C, T = est.shape
    est_f = est.view(B * C, T)
    tgt_f = tgt.view(B * C, T)

    est_zm = est_f - est_f.mean(dim=-1, keepdim=True)
    tgt_zm = tgt_f - tgt_f.mean(dim=-1, keepdim=True)

    dot = (est_zm * tgt_zm).sum(dim=-1, keepdim=True)
    tgt_energy = (tgt_zm ** 2).sum(dim=-1, keepdim=True).clamp_min(eps)
    a = dot / tgt_energy

    s_target = a * tgt_zm
    e_noise = est_zm - s_target

    s_target_energy = (s_target ** 2).sum(dim=-1)
    noise_energy = (e_noise ** 2).sum(dim=-1).clamp_min(eps)

    sdr = 10.0 * torch.log10(s_target_energy / noise_energy)
    return sdr.view(B, C).mean(dim=-1)


def si_improvement(inp: torch.Tensor,
                   est: torch.Tensor,
                   tgt: torch.Tensor,
                   eps: float = 1e-8) -> torch.Tensor:
    """
    SI-SDR improvement per example (in dB):
        ΔSI-SDR = SI-SDR(est, tgt) - SI-SDR(inp, tgt)
    """
    sdr_in = si_sdr(inp, tgt, eps=eps)
    sdr_est = si_sdr(est, tgt, eps=eps)
    return sdr_est - sdr_in


def diff_signal_correlation(inp: torch.Tensor,
                            tgt: torch.Tensor,
                            est: torch.Tensor,
                            eps: float = 1e-8) -> torch.Tensor:
    """
    Difference Signal Correlation per example (higher is better):

      r_true    = inp - tgt      # true reverb
      r_removed = inp - est      # what the model removed

      corr = corr(r_true, r_removed) (Pearson), averaged over channels.

    inp, tgt, est: (B, C, T)
    Returns: (B,)
    """
    B, C, T = inp.shape

    r_true = (inp - tgt).view(B * C, T)
    r_rem  = (inp - est).view(B * C, T)

    # zero-mean
    r_true = r_true - r_true.mean(dim=-1, keepdim=True)
    r_rem  = r_rem - r_rem.mean(dim=-1, keepdim=True)

    num = (r_true * r_rem).sum(dim=-1)
    den = torch.sqrt(
        (r_true ** 2).sum(dim=-1).clamp_min(eps) *
        (r_rem  ** 2).sum(dim=-1).clamp_min(eps)
    )
    corr = num / den
    corr = corr.view(B, C).mean(dim=-1)  # average over channels
    return corr


# ======================================
#  STFT-based metrics
# ======================================

def _stft_mag_phase(x: torch.Tensor,
                    n_fft: int,
                    hop: int,
                    win_length: int | None = None,
                    window: torch.Tensor | None = None,
                    center: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: (B, C, T)
    Returns:
      mag:   (B, C, F, T_frames)
      phase: (B, C, F, T_frames)
    """
    B, C, T = x.shape
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(win_length, device=x.device)

    # compute STFT per channel
    x_flat = x.view(B * C, T)
    stft = torch.stft(x_flat,
                      n_fft=n_fft,
                      hop_length=hop,
                      win_length=win_length,
                      window=window,
                      center=center,
                      return_complex=True)
    # stft: (B*C, F, T_frames)
    mag = stft.abs().view(B, C, stft.size(-2), stft.size(-1))
    phase = stft.angle().view(B, C, stft.size(-2), stft.size(-1))
    return mag, phase


def multi_stft_mae(est: torch.Tensor,
                   tgt: torch.Tensor,
                   stft_configs: List[Tuple[int, int]],
                   center: bool = True) -> torch.Tensor:
    """
    Multi-STFT L1 loss over magnitudes, per example.
    stft_configs: list of (n_fft, hop)
    Returns: (B,)
    """
    B = est.size(0)
    total = torch.zeros(B, device=est.device)

    for n_fft, hop in stft_configs:
        mag_e, _ = _stft_mag_phase(est, n_fft, hop, center=center)
        mag_t, _ = _stft_mag_phase(tgt, n_fft, hop, center=center)
        total += (mag_e - mag_t).abs().mean(dim=(-1, -2, -3))  # avg over C,F,T

    return total / len(stft_configs)


def phase_mae(est: torch.Tensor,
              tgt: torch.Tensor,
              n_fft: int,
              hop: int,
              center: bool = True) -> torch.Tensor:
    """
    Phase-related loss (L1 on wrapped phase difference).
    est, tgt: (B, C, T)
    Returns: (B,)
    """
    _, phase_e = _stft_mag_phase(est, n_fft, hop, center=center)
    _, phase_t = _stft_mag_phase(tgt, n_fft, hop, center=center)

    # wrap difference to [-pi, pi]
    diff = phase_e - phase_t
    diff = (diff + math.pi) % (2 * math.pi) - math.pi
    return diff.abs().mean(dim=(-1, -2, -3))


# ======================================
#  Information-theoretic metrics
# ======================================

def _hist1d_probs(x: torch.Tensor,
                  bins: int = 64,
                  eps: float = 1e-8) -> torch.Tensor:
    """
    x: (N,) -> probabilities (bins,)
    """
    xmin = x.min()
    xmax = x.max()
    if xmax <= xmin:
        # degenerate case: single value
        hist = torch.zeros(bins, device=x.device)
        hist[0] = 1.0
        return hist

    x_norm = (x - xmin) / (xmax - xmin + eps)
    idx = (x_norm * (bins - 1)).long().clamp(0, bins - 1)
    hist = torch.bincount(idx, minlength=bins).float()
    p = hist / hist.sum().clamp_min(eps)
    return p


def _hist2d_probs(x: torch.Tensor,
                  y: torch.Tensor,
                  bins: int = 32,
                  eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    x, y: (N,) -> pxy (bins,bins), px (bins,), py (bins,)
    """
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    x_norm = (x - xmin) / (xmax - xmin + eps)
    y_norm = (y - ymin) / (ymax - ymin + eps)

    ix = (x_norm * (bins - 1)).long().clamp(0, bins - 1)
    iy = (y_norm * (bins - 1)).long().clamp(0, bins - 1)
    idx = ix * bins + iy

    hist = torch.bincount(idx, minlength=bins * bins).float()
    pxy = hist / hist.sum().clamp_min(eps)
    pxy = pxy.view(bins, bins)
    px = pxy.sum(dim=1)  # sum over y
    py = pxy.sum(dim=0)  # sum over x
    return pxy, px, py


def nmi(est: torch.Tensor,
        tgt: torch.Tensor,
        stft_cfg: Tuple[int, int],
        bins: int = 32,
        eps: float = 1e-8) -> torch.Tensor:
    """
    Normalized Mutual Information between magnitude spectra of est & tgt.
    est, tgt: (B, C, T)
    Returns: (B,)
    """
    n_fft, hop = stft_cfg
    B = est.size(0)
    vals = torch.zeros(B, device=est.device)

    mag_e, _ = _stft_mag_phase(est, n_fft, hop)
    mag_t, _ = _stft_mag_phase(tgt, n_fft, hop)

    # flatten per example
    for b in range(B):
        xe = mag_e[b].reshape(-1)
        xt = mag_t[b].reshape(-1)

        pxy, px, py = _hist2d_probs(xe, xt, bins=bins, eps=eps)
        # MI
        pxy_safe = pxy + eps
        px_safe = px + eps
        py_safe = py + eps

        # log pxy / (px*py)
        log_arg = pxy_safe / (px_safe[:, None] * py_safe[None, :])
        mi = (pxy_safe * log_arg.log()).sum()

        # entropies
        Hx = - (px_safe * px_safe.log()).sum()
        Hy = - (py_safe * py_safe.log()).sum()

        vals[b] = 2.0 * mi / (Hx + Hy + eps)

    return vals


def normalized_entropy(mag: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalized spectral entropy per example.
    mag: (B, C, F, T_frames) magnitude
    Returns: (B,) entropy normalized by log(K) with K = F*C
    """
    B, C, F, T = mag.shape
    # average over time
    power = (mag ** 2).mean(dim=-1)  # (B, C, F)
    power = power.view(B, -1)        # (B, K)
    power = power + eps
    power = power / power.sum(dim=-1, keepdim=True).clamp_min(eps)
    # entropy
    H = - (power * power.log()).sum(dim=-1)   # (B,)
    K = power.size(1)
    H_norm = H / math.log(K + eps)
    return H_norm


def normalized_entropy_diff(est: torch.Tensor,
                            tgt: torch.Tensor,
                            stft_cfg: Tuple[int, int]) -> torch.Tensor:
    """
    |H_norm(est) - H_norm(tgt)| per example.
    est, tgt: (B, C, T)
    """
    n_fft, hop = stft_cfg
    mag_e, _ = _stft_mag_phase(est, n_fft, hop)
    mag_t, _ = _stft_mag_phase(tgt, n_fft, hop)

    H_e = normalized_entropy(mag_e)
    H_t = normalized_entropy(mag_t)
    return (H_e - H_t).abs()


# ======================================
#  RMS Smoothness Difference
# ======================================

def rms_envelope(x: torch.Tensor,
                 frame_len: int,
                 hop: int,
                 eps: float = 1e-8) -> torch.Tensor:
    """
    x: (B, C, T)
    Returns RMS envelope: (B, C, N_frames)
    """
    B, C, T = x.shape
    x_unf = x.unfold(dimension=-1, size=frame_len, step=hop)  # (B, C, N, frame_len)
    rms = torch.sqrt((x_unf ** 2).mean(dim=-1) + eps)
    return rms  # (B, C, N)


def rms_smoothness(x: torch.Tensor,
                   frame_len: int,
                   hop: int) -> torch.Tensor:
    """
    Smoothness measure: std of frame-to-frame RMS differences.
    x: (B, C, T)
    Returns: (B,)
    """
    env = rms_envelope(x, frame_len, hop)  # (B, C, N)
    diff = env[:, :, 1:] - env[:, :, :-1]   # (B, C, N-1)
    return diff.std(dim=-1).mean(dim=-1)    # avg over C


def rms_smoothness_diff(est: torch.Tensor,
                        tgt: torch.Tensor,
                        frame_len: int,
                        hop: int) -> torch.Tensor:
    """
    |smoothness(est) - smoothness(tgt)| per example.
    """
    s_e = rms_smoothness(est, frame_len, hop)
    s_t = rms_smoothness(tgt, frame_len, hop)
    return (s_e - s_t).abs()

# ======================================
#  Perceptual Drum Metrics
# ======================================


def modulation_spectrum_distance(est: torch.Tensor,
                                 tgt: torch.Tensor,
                                 n_fft: int = 1024,
                                 hop: int = 256,
                                 center: bool = True,
                                 eps: float = 1e-8) -> torch.Tensor:
    """
    Modulation Spectrum Distance (log-domain L2), per example.
    est, tgt: (B, C, T)
    Returns: (B,) (lower is better)
    """
    # mono mix
    est_m = est.mean(dim=1)  # (B, T)
    tgt_m = tgt.mean(dim=1)  # (B, T)

    window = torch.hann_window(n_fft, device=est.device)

    # STFT
    Es = torch.stft(est_m, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                    window=window, center=center, return_complex=True)
    Ts = torch.stft(tgt_m, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                    window=window, center=center, return_complex=True)

    Emag = Es.abs() + eps   # (B,F,Tf)
    Tmag = Ts.abs() + eps

    # average over frequency to get broadband envelopes
    E_env = Emag.mean(dim=1)  # (B, Tf)
    T_env = Tmag.mean(dim=1)  # (B, Tf)

    # modulation spectra via FFT over time
    E_mod = torch.fft.rfft(E_env, dim=-1)
    T_mod = torch.fft.rfft(T_env, dim=-1)

    # log-magnitude modulation spectra
    E_log = (E_mod.abs() + eps).log10()
    T_log = (T_mod.abs() + eps).log10()

    # optional: zero-mean to remove global loudness bias
    E_log = E_log - E_log.mean(dim=-1, keepdim=True)
    T_log = T_log - T_log.mean(dim=-1, keepdim=True)

    # L2 distance over modulation bins
    dist = (E_log - T_log).pow(2).mean(dim=-1).sqrt()  # (B,)
    return dist


def envelope_correlation(est: torch.Tensor,
                         tgt: torch.Tensor,
                         sr: int,
                         frame_ms: float = 10.0,
                         hop_ms: float = 5.0,
                         eps: float = 1e-8) -> torch.Tensor:
    """
    Pearson correlation between RMS envelopes of est and tgt.
    est, tgt: (B, C, T)
    Returns: (B,) (higher is better)
    """
    frame_len = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    frame_len = max(1, frame_len)
    hop = max(1, hop)

    env_e = rms_envelope(est, frame_len, hop)  # (B,C,N)
    env_t = rms_envelope(tgt, frame_len, hop)

    # average over channels -> (B,N)
    env_e = env_e.mean(dim=1)
    env_t = env_t.mean(dim=1)

    # zero-mean
    env_e = env_e - env_e.mean(dim=-1, keepdim=True)
    env_t = env_t - env_t.mean(dim=-1, keepdim=True)

    num = (env_e * env_t).sum(dim=-1)
    den = torch.sqrt(
        (env_e ** 2).sum(dim=-1).clamp_min(eps) *
        (env_t ** 2).sum(dim=-1).clamp_min(eps)
    )
    corr = num / den
    return corr  # (B,)


def windowed_tter(x: torch.Tensor,
                  sr: int,
                  win_ms: float = 300.0,
                  hop_ms: float = 100.0,
                  attack_frac: float = 0.25,
                  eps: float = 1e-8) -> torch.Tensor:
    """
    Windowed Transient-to-Tail Energy Ratio (TTER) in dB, per example.

    x: (B, C, T)
    sr: sample rate
    win_ms: window length in ms (e.g. 300 ms)
    hop_ms: hop between windows in ms (e.g. 100 ms)
    attack_frac: fraction of window considered 'attack' (e.g. 0.25 -> first 25%)

    Steps:
      - Slide a window across the waveform.
      - In each window, split into [attack | tail].
      - Compute 10*log10(E_attack/E_tail) for that window.
      - Average TTER across windows and channels.

    Returns:
      tter: (B,) tensor
    """
    B, C, T = x.shape

    win_len = int(sr * win_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    win_len = max(2, min(win_len, T))    # at least 2 samples, not longer than signal
    hop_len = max(1, hop_len)

    # unfold along time -> (B, C, N_windows, win_len)
    if T < win_len:
        # pad at end if signal is shorter than one window
        pad = win_len - T
        x_pad = torch.nn.functional.pad(x, (0, pad))
    else:
        x_pad = x

    T_pad = x_pad.shape[-1]
    n_windows = 1 + (T_pad - win_len) // hop_len
    if n_windows <= 0:
        # fallback: single global window
        x_unf = x_pad.unsqueeze(-2)  # (B,C,1,T_pad)
        win_len = T_pad
    else:
        x_unf = x_pad.unfold(dimension=-1, size=win_len, step=hop_len)  # (B,C,N,win_len)

    # attack vs tail split inside each window
    N = x_unf.shape[-2]
    attack_len = max(1, int(attack_frac * win_len))
    tail_len = win_len - attack_len
    if tail_len <= 0:
        tail_len = 1
        attack_len = win_len - 1

    attack = x_unf[..., :attack_len]          # (B,C,N,attack_len)
    tail   = x_unf[..., attack_len:]         # (B,C,N,tail_len)

    E_att = (attack ** 2).sum(dim=-1).clamp_min(eps)   # (B,C,N)
    E_tail = (tail ** 2).sum(dim=-1).clamp_min(eps)    # (B,C,N)

    # tail-to-attack ratio in dB (higher = more reverb / longer tails)
    ratio_win = 10.0 * torch.log10(E_tail / E_att)     # (B,C,N)

    # average across windows and channels
    ratio = ratio_win.mean(dim=-1).mean(dim=-1)        # (B,)
    return ratio

def tter_window_metrics(inp: torch.Tensor,
                        est: torch.Tensor,
                        tgt: torch.Tensor,
                        sr: int,
                        win_ms: float = 300.0,
                        hop_ms: float = 100.0,
                        attack_frac: float = 0.25) -> Dict[str, torch.Tensor]:
    """
    Compute TTER-based metrics for continuous stems.

    Returns dict of (B,) tensors:
      - tter_inp, tter_est, tter_tgt     (for debugging/analysis)
      - tter_absdiff   = |TTER_est - TTER_tgt|   (lower is better)
      - tter_impr      = TTER_est - TTER_inp     (higher is better)
    """
    tter_inp = windowed_tter(inp, sr, win_ms, hop_ms, attack_frac)
    tter_est = windowed_tter(est, sr, win_ms, hop_ms, attack_frac)
    tter_tgt = windowed_tter(tgt, sr, win_ms, hop_ms, attack_frac)

    return {
        "tter_inp": tter_inp,
        "tter_est": tter_est,
        "tter_tgt": tter_tgt,
        "tter_absdiff": (tter_est - tter_tgt).abs(),
        "tter_impr": tter_inp - tter_est,
    }

def _per_hit_tter_single(x_inp: np.ndarray,
                         x_est: np.ndarray,
                         x_tgt: np.ndarray,
                         sr: int,
                         attack_ms: float,
                         tail_ms: float,
                         onset_backtrack: bool = True,
                         onset_kwargs: dict | None = None,
                         eps: float = 1e-8):
    """
    Compute per-hit tail/attack ratios (in dB) for inp/est/tgt.

    x_*: mono numpy arrays (T,)
    Returns:
        ratios_inp, ratios_est, ratios_tgt: 1D numpy arrays (n_hits,)
    """
    if onset_kwargs is None:
        onset_kwargs = {}

    # Onset detection on dry target
    onsets = librosa.onset.onset_detect(
        y=x_tgt,
        sr=sr,
        units="samples",
        backtrack=onset_backtrack,
        **onset_kwargs,
    )
    if len(onsets) == 0:
        return np.array([]), np.array([]), np.array([])

    T = len(x_tgt)
    att_len = int(sr * attack_ms / 1000.0)
    tail_len = int(sr * tail_ms / 1000.0)
    min_len = max(8, att_len // 4)  # minimum useful samples per region

    ratios_inp = []
    ratios_est = []
    ratios_tgt = []

    for o in onsets:
        start_a = int(o)
        end_a = min(start_a + att_len, T)
        start_t = end_a
        end_t = min(start_t + tail_len, T)

        if end_t - start_t < min_len or end_a - start_a < min_len:
            continue  # skip truncated / tiny segments

        segs = []
        for x in (x_inp, x_est, x_tgt):
            attack = x[start_a:end_a]
            tail = x[start_t:end_t]
            E_att = np.sum(attack ** 2)
            E_tail = np.sum(tail ** 2)
            if E_att < eps or E_tail < eps:
                segs = None
                break
            r_db = 10.0 * np.log10((E_tail + eps) / (E_att + eps))
            segs.append(r_db)

        if segs is None:
            continue

        ri, re, rt = segs
        ratios_inp.append(ri)
        ratios_est.append(re)
        ratios_tgt.append(rt)

    return (np.array(ratios_inp),
            np.array(ratios_est),
            np.array(ratios_tgt))


def per_hit_tter_metrics(inp: torch.Tensor,
                         est: torch.Tensor,
                         tgt: torch.Tensor,
                         sr: int,
                         attack_ms: float = 40.0,
                         tail_ms: float = 200.0,
                         onset_backtrack: bool = True,
                         onset_kwargs: dict | None = None) -> Dict[str, torch.Tensor]:
    """
    Per-hit TTER metrics (using onset detection on dry target).

    inp, est, tgt: (B, C, T) tensors
    Returns dict of (B,) tensors:
      - hit_tter_absdiff: mean over hits of |r_est - r_tgt| (↓ better)
      - hit_tter_impr:    mean over hits of (r_inp - r_est) (↑ better)
      - hit_tter_hits:    number of hits used per example (for diagnostics)
    """
    B, C, T = inp.shape
    hit_absdiff = torch.zeros(B)
    hit_impr = torch.zeros(B)
    hit_count = torch.zeros(B)

    # work on CPU numpy (librosa)
    inp_np = inp.detach().cpu().numpy()
    est_np = est.detach().cpu().numpy()
    tgt_np = tgt.detach().cpu().numpy()

    for b in range(B):
        # mono mix
        x_inp = inp_np[b].mean(axis=0)
        x_est = est_np[b].mean(axis=0)
        x_tgt = tgt_np[b].mean(axis=0)

        ri, re, rt = _per_hit_tter_single(
            x_inp, x_est, x_tgt, sr,
            attack_ms=attack_ms,
            tail_ms=tail_ms,
            onset_backtrack=onset_backtrack,
            onset_kwargs=onset_kwargs,
        )

        if ri.size == 0:
            # no valid hits; leave zeros
            continue

        # per-hit metrics
        absdiff_hits = np.abs(re - rt)
        impr_hits = (ri - re)

        hit_absdiff[b] = float(absdiff_hits.mean())
        hit_impr[b] = float(impr_hits.mean())
        hit_count[b] = float(len(ri))

    return {
        "hit_tter_absdiff": hit_absdiff,
        "hit_tter_impr": hit_impr,
        "hit_tter_hits": hit_count,
    }

def _onset_times(y: np.ndarray, sr: int,
                 hop_length: int = 512,
                 backtrack: bool = True,
                 onset_kwargs: dict | None = None) -> np.ndarray:
    if onset_kwargs is None:
        onset_kwargs = {}
    onsets = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        units="time",
        hop_length=hop_length,
        backtrack=backtrack,
        **onset_kwargs,
    )
    return onsets


def _onset_f1(ref: np.ndarray,
              cand: np.ndarray,
              tol: float = 0.03) -> float:
    """
    F1 between reference and candidate onset times.
    ref, cand: 1D arrays in seconds
    tol: match tolerance in seconds
    """
    if len(ref) == 0 and len(cand) == 0:
        return 1.0
    if len(ref) == 0 or len(cand) == 0:
        return 0.0

    ref_used = np.zeros(len(ref), dtype=bool)
    tp = 0
    for c in cand:
        # find nearest ref onset
        idx = np.argmin(np.abs(ref - c))
        if not ref_used[idx] and abs(ref[idx] - c) <= tol:
            tp += 1
            ref_used[idx] = True

    fp = len(cand) - tp
    fn = len(ref) - tp

    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def onset_f_metrics(inp: torch.Tensor,
                    est: torch.Tensor,
                    tgt: torch.Tensor,
                    sr: int,
                    hop_length: int = 512,
                    tol: float = 0.03) -> Dict[str, torch.Tensor]:
    """
    Onset F-measure metrics per example.

    inp, est, tgt: (B, C, T)
    Returns dict of (B,) tensors:
      - onset_F_inp
      - onset_F_est
      - onset_F_impr = onset_F_est - onset_F_inp
    """
    B, C, T = inp.shape
    F_inp = torch.zeros(B)
    F_est = torch.zeros(B)

    inp_np = inp.detach().cpu().numpy()
    est_np = est.detach().cpu().numpy()
    tgt_np = tgt.detach().cpu().numpy()

    for b in range(B):
        # mono downmix
        y_inp = inp_np[b].mean(axis=0)
        y_est = est_np[b].mean(axis=0)
        y_tgt = tgt_np[b].mean(axis=0)

        ref = _onset_times(y_tgt, sr, hop_length=hop_length)
        c_inp = _onset_times(y_inp, sr, hop_length=hop_length)
        c_est = _onset_times(y_est, sr, hop_length=hop_length)

        F_inp[b] = _onset_f1(ref, c_inp, tol=tol)
        F_est[b] = _onset_f1(ref, c_est, tol=tol)

    return {
        "onset_F_inp": F_inp,
        "onset_F_est": F_est,
        "onset_F_impr": F_est - F_inp,
    }


# ======================================
#  MetricComputer & MetricAggregator
# ======================================

class MetricComputer:
    """
    Modular metric computer: choose which metrics you want via flags/config.
    """

    def __init__(self,
                 sr: int,
                 use_audio_mae: bool = True,
                 use_multi_stft: bool = True,
                 use_phase_mae: bool = True,
                 use_si_metrics: bool = True,
                 use_si_metrics_impr: bool = True, 
                 use_nmi: bool = True,
                 use_norm_entropy_diff: bool = False,
                 use_rms_smoothness_diff: bool = False,
                 use_diff_signal_corr: bool = True,
                 use_esr: bool = True,
                 # perceptual drum metrics
                 use_modulation_dist: bool = True,
                 use_tter: bool = True,
                 use_env_corr: bool = True,
                 use_hit_tter = True,
                 use_onset_f = True,
                 # metric-specific configs:
                 mstft_cfgs: List[Tuple[int, int]] | None = None,
                 phase_fft_cfg: Tuple[int, int] | None = None,
                 nmi_fft_cfg: Tuple[int, int] | None = None,
                 rms_frame_len: int | None = None,
                 rms_hop: int | None = None,
                 nmi_bins: int = 32):
        self.sr = sr
        self.use_audio_mae = use_audio_mae
        self.use_multi_stft = use_multi_stft
        self.use_phase_mae = use_phase_mae
        self.use_si_metrics = use_si_metrics
        self.use_si_metrics_impr = use_si_metrics_impr
        self.use_nmi = use_nmi
        self.use_norm_entropy_diff = use_norm_entropy_diff
        self.use_rms_smoothness_diff = use_rms_smoothness_diff
        self.use_diff_signal_corr = use_diff_signal_corr
        self.use_esr = use_esr
        self.use_modulation_dist = use_modulation_dist
        self.use_env_corr = use_env_corr
        self.use_tter = use_tter
        self.use_hit_tter = use_hit_tter
        self.use_onset_f = use_onset_f


        # defaults
        if mstft_cfgs is None:
            # (n_fft, hop)
            mstft_cfgs = [
                (512, 128),
                (1024, 256),
                (2048, 512),
            ]
        if phase_fft_cfg is None:
            phase_fft_cfg = (1024, 256)
        if nmi_fft_cfg is None:
            nmi_fft_cfg = (1024, 256)
        if rms_frame_len is None:
            rms_frame_len = int(0.03 * sr)  # 30 ms
        if rms_hop is None:
            rms_hop = rms_frame_len // 2

        self.mstft_cfgs = mstft_cfgs
        self.phase_fft_cfg = phase_fft_cfg
        self.nmi_fft_cfg = nmi_fft_cfg
        self.rms_frame_len = rms_frame_len
        self.rms_hop = rms_hop
        self.nmi_bins = nmi_bins

    @torch.no_grad()
    def compute_batch(self,
                      inp_wav: torch.Tensor,
                      tgt_wav: torch.Tensor,
                      est_wav: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        All tensors: (B, C, T)
        Return dict: metric_name -> (B,) tensor
        """
        metrics = {}

        if self.use_audio_mae:
            metrics["audio_mae"] = audio_mae(est_wav, tgt_wav)

        if self.use_multi_stft:
            metrics["multi_stft_mae"] = multi_stft_mae(
                est_wav, tgt_wav, self.mstft_cfgs
            )

        if self.use_phase_mae:
            fft, hop = self.phase_fft_cfg
            metrics["phase_mae"] = phase_mae(est_wav, tgt_wav, fft, hop)

        if self.use_si_metrics:
            metrics["si_sdr"] = si_sdr(est_wav, tgt_wav)

        if self.use_si_metrics_impr:
            metrics["si_sdr_impr"] = si_improvement(inp_wav, est_wav, tgt_wav)

        if self.use_nmi:
            metrics["nmi"] = nmi(
                est_wav, tgt_wav, self.nmi_fft_cfg, bins=self.nmi_bins
            )

        if self.use_norm_entropy_diff:
            metrics["norm_entropy_diff"] = normalized_entropy_diff(
                est_wav, tgt_wav, self.nmi_fft_cfg
            )

        if self.use_rms_smoothness_diff:
            metrics["rms_smoothness_diff"] = rms_smoothness_diff(
                est_wav, tgt_wav, self.rms_frame_len, self.rms_hop
            )

        if self.use_diff_signal_corr:
            metrics["diff_signal_corr"] = diff_signal_correlation(
                inp_wav, tgt_wav, est_wav
            )

        if self.use_esr:
            metrics["esr"] = esr(est_wav, tgt_wav)

        if self.use_modulation_dist:
            metrics["mod_spec_dist"] = modulation_spectrum_distance(
                est_wav, tgt_wav
            )  
        if self.use_tter:
            tter_dict = tter_window_metrics(
                inp_wav, est_wav, tgt_wav, self.sr
            )
            # pick which ones you want to aggregate globally
            metrics["tter_absdiff"] = tter_dict["tter_absdiff"]
            metrics["tter_impr"] = tter_dict["tter_impr"]
            # if you want debug stats:
            # metrics["tter_est"] = tter_dict["tter_est"]
            # metrics["tter_tgt"] = tter_dict["tter_tgt"]

        if self.use_env_corr:
            metrics["env_corr"] = envelope_correlation(
                est_wav, tgt_wav, self.sr
            )   

        if self.use_hit_tter:
            hit_tter = per_hit_tter_metrics(
                inp_wav, est_wav, tgt_wav, self.sr
            )
            # we usually aggregate absdiff and improvement; hits count is diagnostic
            metrics["hit_tter_absdiff"] = hit_tter["hit_tter_absdiff"]
            metrics["hit_tter_impr"] = hit_tter["hit_tter_impr"]
            # you can also log hit_tter_hits if you want   

        if self.use_onset_f:
            onset_dict = onset_f_metrics(inp_wav, est_wav, tgt_wav, self.sr)
            metrics["onset_F_impr"] = onset_dict["onset_F_impr"]


        return metrics



class MetricAggregator:
    """
    Online mean/std aggregator. Keeps per-example lists (optional) and
    running sums for mean/std.

    Use:
        agg = MetricAggregator(param_count=n_params)
        for batch:
            m = metric_computer.compute_batch(...)
            agg.update(m)

        summary = agg.summary()
        # summary["param_count"]
        # summary["metrics"][name] = {mean, std, count}
    """

    def __init__(self, keep_per_example: bool = False, param_count: int | None = None):
        self.keep_per_example = keep_per_example
        self.param_count = param_count

        self._sum = {}
        self._sum_sq = {}
        self._count = {}
        self._values = {}  # optional per-example storage

    def set_param_count(self, n: int):
        self.param_count = int(n)

    def update(self, batch_metrics: Dict[str, torch.Tensor]):
        """
        batch_metrics: name -> (B,) tensor
        """
        for name, vals in batch_metrics.items():
            v = vals.detach().cpu()
            if name not in self._sum:
                self._sum[name] = 0.0
                self._sum_sq[name] = 0.0
                self._count[name] = 0
                if self.keep_per_example:
                    self._values[name] = []

            self._sum[name] += float(v.sum())
            self._sum_sq[name] += float((v ** 2).sum())
            self._count[name] += int(v.numel())

            if self.keep_per_example:
                self._values[name].append(v.numpy())

    def summary(self) -> Dict[str, Dict]:
        """
        Returns:
          {
            "param_count": int or None,
            "metrics": {
              metric_name: { "mean": float, "std": float, "count": int }
            }
          }
        """
        out_metrics = {}
        for name in self._sum.keys():
            c = self._count[name]
            if c == 0:
                continue
            mean = self._sum[name] / c
            mean_sq = self._sum_sq[name] / c
            var = max(mean_sq - mean ** 2, 0.0)
            std = math.sqrt(var)
            out_metrics[name] = {
                "mean": mean,
                "std": std,
                "count": c,
            }

        return {
            "param_count": self.param_count,
            "metrics": out_metrics,
        }


def write_html_summary(
    all_model_stats: Dict[str, Dict[str, Any]],
    filepath: str,
    higher_is_better: Dict[str, bool],
    metric_groups: Dict[str, List[str]],
):
    """
    all_model_stats:
      model_name -> {
        "param_count": int,
        "metrics": {
          metric_name: {"mean": float, "std": float, "count": int}
        }
      }

    higher_is_better:
      metric_name -> True if higher mean is better, False if lower is better.

    metric_groups:
      group_name -> list of metric names to show in that group (table)
    """
    html = []
    html.append("<html><head><meta charset='utf-8'><title>Dereverb Metrics</title>")
    html.append(
        "<style>"
        "body { font-family: sans-serif; }"
        "table { border-collapse: collapse; margin-bottom: 24px; }"
        "th, td { border: 1px solid #999; padding: 4px 8px; text-align: center; }"
        "th { background-color: #eee; }"
        ".best { background-color: #c8e6c9; font-weight: bold; }"
        ".group-title { margin-top: 24px; margin-bottom: 8px; }"
        "</style>"
    )
    html.append("</head><body>")
    html.append("<h2>Drums Dereverberation: Model Comparison</h2>")

    # For each group (low-level / high-level) we build a separate table
    first_group = True
    for group_name, metric_list in metric_groups.items():
        # Keep only metrics actually present in at least one model
        group_metrics = []
        for m in metric_list:
            if any(m in stats["metrics"] for stats in all_model_stats.values()):
                group_metrics.append(m)
        if not group_metrics:
            continue  # nothing to show for this group

        # Compute best value for each metric within this group
        best_values = {}
        for m in group_metrics:
            vals = []
            for stats in all_model_stats.values():
                if m in stats["metrics"]:
                    vals.append(stats["metrics"][m]["mean"])
            if not vals:
                continue
            if higher_is_better.get(m, False):
                best_values[m] = max(vals)
            else:
                best_values[m] = min(vals)

        # Group title
        html.append(f"<h3 class='group-title'>{group_name}</h3>")
        html.append("<table>")

        # Header row
        html.append("<tr>")
        html.append("<th>Model</th>")
        # Show param count in every table (or only in first if you prefer)
        html.append("<th>Params (M)</th>")
        for m in group_metrics:
            suffix = " (↑)" if higher_is_better.get(m, False) else " (↓)"
            html.append(f"<th>{m}{suffix}</th>")
        html.append("</tr>")

        # Rows per model
        for model_name, stats in all_model_stats.items():
            html.append("<tr>")
            html.append(f"<td>{model_name}</td>")

            p = stats.get("param_count", None)
            if p is None:
                html.append("<td>–</td>")
            else:
                html.append(f"<td>{p/1e6:.2f}</td>")  # in millions

            for m in group_metrics:
                if m not in stats["metrics"]:
                    html.append("<td>–</td>")
                    continue
                mean = stats["metrics"][m]["mean"]
                std = stats["metrics"][m]["std"]
                best = best_values.get(m, None)

                # Decide if this is best (within tiny tolerance)
                is_best = (best is not None) and (abs(mean - best) < 1e-9)
                cls = "best" if is_best else ""
                html.append(
                    f"<td class='{cls}'>{mean:.3f} ± {std:.3f}</td>"
                )
            html.append("</tr>")

        html.append("</table>")

        first_group = False

    html.append("</body></html>")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"[HTML] Wrote summary to {filepath}")