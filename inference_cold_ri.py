import argparse
import os
from contextlib import nullcontext
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from configs.io import load_config
from backbones.unet_stereo import UNetRI
from backbones.dit_stereo import TransformerDiffuser
from backbones.utils_inference import (
    ensure_float_audio,
    ensure_stereo,
    resample_audio,
    trim_or_pad_range,
    set_loudness,
    segment_audio_torch,
    ola_reconstruct_torch,
    audio_to_stereo_ri_stft,
)


VALID_EXTS = {".wav", ".mp3"}


def make_alpha_bar(diffusion_steps: int, device, kind="poly", power=3.0, beta=5.0, k=8.0):
    import math

    T = diffusion_steps
    t = torch.arange(T + 1, device=device, dtype=torch.float32)
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

    a[0] = 1.0
    a[-1] = 0.0
    return a


def build_model(model_cfg):
    if model_cfg.backbone == "unet":
        return UNetRI(model_cfg)
    if model_cfg.backbone == "dit":
        return TransformerDiffuser(model_cfg)
    raise ValueError(f"Unsupported backbone: {model_cfg.backbone}")


def apply_ema_to_model(model: nn.Module, ema_state) -> bool:
    """
    Supports both:
      - dict[name -> tensor]
      - legacy list[tensor] in trainable-parameter order
    """
    if ema_state is None:
        return False

    if isinstance(ema_state, dict):
        shadow = ema_state.get("shadow", ema_state) if "shadow" in ema_state else ema_state
        if not isinstance(shadow, dict):
            return False

        applied = 0
        named_params = dict(model.named_parameters())
        with torch.no_grad():
            for name, param in named_params.items():
                if not param.requires_grad:
                    continue
                if name not in shadow:
                    continue
                tensor = shadow[name]
                if tensor.shape != param.shape:
                    continue
                param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
                applied += 1
        return applied > 0

    if isinstance(ema_state, list):
        trainable = [p for p in model.parameters() if p.requires_grad]
        if len(ema_state) != len(trainable):
            return False
        with torch.no_grad():
            for param, tensor in zip(trainable, ema_state):
                if tensor.shape != param.shape:
                    return False
                param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
        return True

    return False


def load_checkpoint(model: nn.Module, ckpt_path: str, device: str, prefer_ema: bool = True):
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model_state = model.state_dict()

    filtered = {}
    skipped = []
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    model.to(device)

    if skipped:
        print(f"[WARN] Skipped {len(skipped)} checkpoint keys due to mismatch.")
    if missing:
        print(f"[WARN] Missing {len(missing)} model keys after loading.")
    if unexpected:
        print(f"[WARN] Unexpected {len(unexpected)} keys after loading.")

    ema_applied = False
    if prefer_ema and isinstance(ckpt, dict) and "ema" in ckpt:
        ema_applied = apply_ema_to_model(model, ckpt["ema"])
        if ema_applied:
            print("[INFO] EMA weights applied.")
        else:
            print("[WARN] EMA found but not applied (format/shape mismatch).")

    return model.eval()


def resolve_checkpoint(cfg, checkpoint_arg: str | None) -> str:
    candidates = []

    if checkpoint_arg:
        candidates.append(checkpoint_arg)

    out_dir = getattr(cfg.experiment, "output_dir", None)
    if out_dir:
        candidates.extend(
            [
                os.path.join(out_dir, "checkpoints", "best.pt"),
                os.path.join(out_dir, "checkpoints", "latest.pt"),
            ]
        )

    for path in candidates:
        if path and os.path.exists(path):
            return path

    raise FileNotFoundError(
        "No checkpoint found. Pass --checkpoint explicitly or ensure one exists under "
        f"{getattr(cfg.experiment, 'output_dir', '<missing output_dir>')}/checkpoints/."
    )


def list_input_files(input_path: str) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        if p.suffix.lower() not in VALID_EXTS:
            raise ValueError(f"Unsupported input file extension: {p.suffix}")
        return [p]

    if not p.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = []
    for ext in VALID_EXTS:
        files.extend(p.rglob(f"*{ext}"))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(f"No wav/mp3 files found under: {input_path}")

    return files


def load_audio_any(path: Path) -> tuple[np.ndarray, int]:
    """
    Returns audio as [T, 2] float32 and original sample rate.
    librosa.load(..., mono=False) returns:
      - mono: [T]
      - stereo/multi: [C, T]
    """
    audio, sr = librosa.load(path.as_posix(), sr=None, mono=False)

    if audio.ndim == 2:
        audio = audio.T  # [T, C]
    audio = ensure_float_audio(audio)
    audio = ensure_stereo(audio)
    return audio.astype(np.float32, copy=False), int(sr)


def safe_set_loudness(audio: np.ndarray, sr: int, lufs: float) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)

    peak = float(np.max(np.abs(audio)))
    if not np.isfinite(peak) or peak < 1e-8:
        return audio.astype(np.float32, copy=False)

    try:
        out = set_loudness(audio, sr, LUFS=lufs)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.astype(np.float32, copy=False)
    except Exception as exc:
        print(f"[WARN] Loudness normalization skipped: {exc}")
        return audio.astype(np.float32, copy=False)


class ColdRIInferencer:
    def __init__(self, model: nn.Module, cfg, device: str):
        self.model = model.to(device).eval()
        self.cfg = cfg
        self.pre_params = cfg.data
        self.train_params = cfg.train
        self.model_params = cfg.model
        self.device = device

        if int(self.train_params.diffusion_steps) != 16:
            print(
                f"[WARN] YAML train.diffusion_steps={self.train_params.diffusion_steps}, "
                "The provided saved models are using fixed 16 reverse steps."
            )
        self.diffusion_steps = 16
        self.alpha_mode = self.train_params.alpha_mode
        self.residual_mode = self.train_params.residual_mode

        self.alpha_bar = make_alpha_bar(
            self.diffusion_steps,
            device=device,
            kind=self.alpha_mode,
        )

        self.center = self.pre_params.center
        self.segment_length = int(round(self.pre_params.sr * self.pre_params.dur))
        self._windows = {}

    def _get_window_for_device(self, device):
        device = torch.device(device)
        key = (device.type, device.index)
        if key not in self._windows:
            if self.pre_params.win_fn == "hann":
                self._windows[key] = torch.hann_window(
                    self.pre_params.win,
                    periodic=True,
                    device=device,
                    dtype=torch.float32,
                )
            elif self.pre_params.win_fn == "hamming":
                self._windows[key] = torch.hamming_window(
                    self.pre_params.win,
                    periodic=True,
                    device=device,
                    dtype=torch.float32,
                )
            else:
                raise ValueError(f"Unsupported window type: {self.pre_params.win_fn}")
        return self._windows[key]

    def istft_from_ri(self, ri_stft: torch.Tensor) -> torch.Tensor:
        """
        ri_stft: (B,4,F,T) packed as [L_R, L_I, R_R, R_I]
        returns: (B,2,T)
        """
        n_fft = self.pre_params.fft
        hop = self.pre_params.hop
        win_length = self.pre_params.win
        window = self._get_window_for_device(ri_stft.device)

        L = torch.complex(ri_stft[:, 0].float(), ri_stft[:, 1].float())
        R = torch.complex(ri_stft[:, 2].float(), ri_stft[:, 3].float())

        recL = torch.istft(
            L,
            n_fft=int(n_fft),
            hop_length=int(hop),
            win_length=int(win_length),
            window=window,
            center=self.center,
            length=self.segment_length,
        )
        recR = torch.istft(
            R,
            n_fft=int(n_fft),
            hop_length=int(hop),
            win_length=int(win_length),
            window=window,
            center=self.center,
            length=self.segment_length,
        )
        return torch.stack([recL, recR], dim=1)

    @torch.no_grad()
    def reverse_diffusion(self, inp_ri: torch.Tensor) -> torch.Tensor:
        x = inp_ri.to(self.device)
        bsize = x.shape[0]

        for t in range(self.diffusion_steps, 0, -1):
            T = torch.full((bsize,), t, device=self.device, dtype=torch.long)

            if self.residual_mode == "next_delta_norm":
                a_t = self.alpha_bar.index_select(0, T)
                a_tm1 = self.alpha_bar.index_select(0, T - 1)
                g = (a_tm1 - a_t).clamp_min(1e-6).view(-1, 1, 1, 1)

                pred = self.model(x, T)
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]
                x = x + g * pred

            elif self.residual_mode == "direct":
                pred = self.model(x, T)
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]
                x = pred

            else:
                raise ValueError(f"Unsupported residual mode: {self.residual_mode}")

        return x

    @torch.inference_mode()
    def dereverb_batch(self, reverb_ri: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        use_amp = self.device.startswith("cuda")
        ctx = (
            torch.amp.autocast(device_type="cuda", enabled=use_amp)
            if use_amp
            else nullcontext()
        )

        with ctx:
            est_ri = self.reverse_diffusion(reverb_ri)

        est_ri = est_ri.float()
        est_wav = self.istft_from_ri(est_ri)
        return est_ri, est_wav


def preprocess_audio(audio: np.ndarray, sr: int, target_sr: int, trim_seconds: float, target_lufs: float) -> np.ndarray:
    audio = ensure_float_audio(audio)
    audio = ensure_stereo(audio)

    if sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr

    audio = trim_or_pad_range(audio, sr, min_s=2.0, max_s=trim_seconds)
    audio = safe_set_loudness(audio, sr, target_lufs)
    return audio.astype(np.float32, copy=False)


def reconstruct_from_segments(
    segs: torch.Tensor,
    est_segments: list[torch.Tensor],
    step: int,
    orig_len: int,
    residual_mode: str,
    input_audio_np: np.ndarray,
) -> torch.Tensor:
    out_segs = torch.stack(est_segments, dim=0)  # [N,2,L]

    if residual_mode == "direct":
        return ola_reconstruct_torch(out_segs, step, orig_len)

    if residual_mode == "next_delta_norm":
        delta_segs = []
        for i in range(out_segs.shape[0]):
            delta_segs.append(out_segs[i] - segs[i].detach().cpu())
        delta_segs = torch.stack(delta_segs, dim=0)
        delta_full = ola_reconstruct_torch(delta_segs, step, orig_len)
        inp_t = torch.from_numpy(input_audio_np[:orig_len]).to(delta_full.dtype)
        out = inp_t + delta_full
        return torch.clamp(out, -1.0, 1.0)

    raise ValueError(f"Unsupported residual mode: {residual_mode}")


def process_one_file(
    input_file: Path,
    input_root: Path,
    output_root: Path,
    inferencer: ColdRIInferencer,
    cfg,
    trim_seconds: float,
    batch_size: int,
):
    print(f"[INFO] Processing: {input_file}")

    audio, sr = load_audio_any(input_file)
    audio = preprocess_audio(
        audio=audio,
        sr=sr,
        target_sr=int(cfg.data.sr),
        trim_seconds=trim_seconds,
        target_lufs=float(cfg.data.lufs),
    )

    segs, step, orig_len = segment_audio_torch(
        audio,
        int(cfg.data.sr),
        ts_min=float(cfg.data.dur),
        overlap=0.5,
        pad_end=True,
        device=inferencer.device,
    )

    ri_in = audio_to_stereo_ri_stft(segs, config=cfg.data, device=inferencer.device)

    est_segments = []
    batch_size = max(1, int(batch_size))

    with torch.inference_mode():
        for start in range(0, ri_in.shape[0], batch_size):
            batch_ri = ri_in[start:start + batch_size]          # [B,4,F,T]
            _, est_wav = inferencer.dereverb_batch(batch_ri)    # [B,2,L]

            for b in range(est_wav.shape[0]):
                est_segments.append(est_wav[b].detach().cpu())

    out_audio = reconstruct_from_segments(
        segs=segs,
        est_segments=est_segments,
        step=step,
        orig_len=orig_len,
        residual_mode=cfg.train.residual_mode,
        input_audio_np=audio,
    )

    out_np = np.clip(out_audio.detach().cpu().numpy(), -1.0, 1.0).astype(np.float32)

    if input_root.is_file():
        rel = Path(input_file.stem + "_dereverb.wav")
    else:
        rel = input_file.relative_to(input_root).with_suffix(".wav")
        rel = rel.with_name(rel.stem + "_dereverb.wav")

    out_path = output_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(out_path.as_posix(), out_np, int(cfg.data.sr))
    print(f"[INFO] Saved: {out_path}")

def resolve_device(gpu_arg: int, cfg_runtime):
    if gpu_arg == -1:
        print("[INFO] Forced CPU inference (--gpu -1).")
        return "cpu"

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if gpu_arg < 0 or gpu_arg >= n_gpus:
            raise ValueError(f"Requested --gpu {gpu_arg}, but available GPU indices are 0..{n_gpus-1}")

        device = f"cuda:{gpu_arg}"
        torch.cuda.set_device(gpu_arg)

        torch.backends.cudnn.benchmark = bool(getattr(cfg_runtime, "cudnn_benchmark", True))
        torch.backends.cuda.matmul.allow_tf32 = bool(getattr(cfg_runtime, "allow_tf32", False))
        torch.set_float32_matmul_precision(getattr(cfg_runtime, "matmul_precision", "high"))

        print(f"[INFO] Using device: {device}")
        return device

    print("[WARN] CUDA not available, falling back to CPU.")
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to model YAML config.")
    parser.add_argument(
        "--input-path",
        required=True,
        help="Single wav/mp3 file or directory containing wav/mp3 files.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Directory where dereverberated wav files will be saved.",
    )
    parser.add_argument(
        "--trim-seconds",
        type=float,
        default=30.0,
        help="Process at most this many seconds from the start. Clamped to [2, 30].",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. If omitted, tries output_dir/checkpoints/best.pt then latest.pt.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index for inference. Use -1 to force CPU. Default: 0.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of 2-second segments processed together during inference.",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Load raw model weights instead of applying EMA if available.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    trim_seconds = float(np.clip(args.trim_seconds, 2.0, 30.0))

    device = resolve_device(args.gpu, cfg.runtime)

    model = build_model(cfg.model)
    ckpt_path = resolve_checkpoint(cfg, args.checkpoint)
    model = load_checkpoint(model, ckpt_path, device=device, prefer_ema=not args.no_ema)

    inferencer = ColdRIInferencer(model=model, cfg=cfg, device=device)

    input_root = Path(args.input_path)
    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)

    files = list_input_files(args.input_path)
    print(f"[INFO] Found {len(files)} input file(s).")

    for file_path in files:
        process_one_file(
            input_file=file_path,
            input_root=input_root,
            output_root=output_root,
            inferencer=inferencer,
            cfg=cfg,
            trim_seconds=trim_seconds,
            batch_size=args.batch_size,
        )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()