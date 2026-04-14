import math
import torch
import numpy as np
import pyloudnorm as pyln
from scipy import signal
from typing import Union, Optional


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

def compute_working_gain(audio: np.ndarray, sr: int,
                         target_lufs: float = -24.0,
                         peak_limit: float = 0.99) -> float:
    if audio.size == 0:
        return 1.0

    peak = float(np.max(np.abs(audio)))
    if not np.isfinite(peak) or peak < 1e-8:
        return 1.0

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)

    g = float(10.0 ** ((target_lufs - loudness) / 20.0))

    # same spirit as dataset prep: keep a peak safeguard
    peak_after = float(np.max(np.abs(audio * g)))
    if peak_after > peak_limit and peak_after > 0:
        g *= peak_limit / peak_after

    return g


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

'''
def ola_reconstruct_torch(segs: torch.Tensor, step: int, orig_len: int):
    """
    segs: [N, C, L] time-domain segments
    Returns: [T, C] torch
    """
    N, C, L = segs.shape
    # w = torch.hann_window(L, periodic=True, device=segs.device, dtype=segs.dtype)  # [L]
    # w = (w ** 2)  # stronger edge suppression
    w = torch.hann_window(L, periodic=False, device=segs.device, dtype=segs.dtype)
    w = torch.sqrt(torch.clamp(w, min=1e-8))  # constant-power style for 50% overlap
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
'''

def ola_reconstruct_torch(segs: torch.Tensor, step: int, orig_len: int):
    """
    segs: [N, C, L]
    returns: [T, C]
    """
    N, C, L = segs.shape

    # stronger boundary suppression than sqrt(Hann)
    w = torch.hann_window(L, periodic=True, device=segs.device, dtype=segs.dtype)
    w = torch.clamp(w, min=1e-8)

    out_len = step * (N - 1) + L
    y = segs.new_zeros((C, out_len))
    norm = segs.new_zeros((out_len,))

    for i in range(N):
        s = i * step
        e = s + L
        y[:, s:e] += segs[i] * w.view(1, L)
        norm[s:e] += w

    y = y / norm.clamp_min(1e-8).unsqueeze(0)
    y = y[:, :orig_len]
    return y.transpose(0, 1).contiguous()


def center_crop_stitch_torch(segs: torch.Tensor, step: int, orig_len: int) -> torch.Tensor:
    """
    Stitch overlapping segments by keeping only the reliable center region.

    segs: [N, C, L]
    step: hop size in samples
    orig_len: target output length in samples

    Returns: [T, C]
    """
    N, C, L = segs.shape

    if N == 0:
        return torch.zeros((orig_len, C), dtype=segs.dtype, device=segs.device)

    if step <= 0 or step > L:
        raise ValueError(f"Invalid step={step} for segment length L={L}")

    # context discarded from left/right for middle segments
    total_extra = L - step
    left_ctx = total_extra // 2
    right_ctx = total_extra - left_ctx

    pieces = []

    for i in range(N):
        seg = segs[i]  # [C, L]

        if N == 1:
            # only one segment: just trim to original length later
            keep = seg

        elif i == 0:
            # first segment: keep from start until right crop
            keep = seg[:, : L - right_ctx]

        elif i == N - 1:
            # last segment: keep from left crop until end
            keep = seg[:, left_ctx:]

        else:
            # middle segments: keep only center "step" samples
            keep = seg[:, left_ctx : left_ctx + step]

        pieces.append(keep)

    y = torch.cat(pieces, dim=1)   # [C, T_total]
    y = y[:, :orig_len]            # trim to original target length
    return y.transpose(0, 1).contiguous()  # [T, C]

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