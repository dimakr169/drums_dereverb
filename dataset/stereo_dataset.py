import os, glob, json, random
from typing import List

import numpy as np
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, random_split


def make_or_load_split(all_files, split_dir, val_ratio=0.2, seed=42):
    os.makedirs(split_dir, exist_ok=True)
    train_f = os.path.join(split_dir, "train_files.json")
    val_f   = os.path.join(split_dir, "val_files.json")

    if os.path.exists(train_f) and os.path.exists(val_f):
        with open(train_f, "r") as f: train_files = json.load(f)
        with open(val_f, "r") as f:   val_files   = json.load(f)
        return train_files, val_files

    # deterministic split
    files = sorted(all_files)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(files), generator=g).tolist()
    n_val = int(round(len(files) * val_ratio))
    val_idx = perm[:n_val]
    trn_idx = perm[n_val:]
    train_files = [files[i] for i in trn_idx]
    val_files   = [files[i] for i in val_idx]

    with open(train_f, "w") as f: json.dump(train_files, f, indent=2)
    with open(val_f, "w") as f:   json.dump(val_files,   f, indent=2)
    return train_files, val_files



def split_paths(all_files, val_ratio: float = 0.2, seed: int = 42):
    """
    Deterministic split of file paths using only PyTorch.
    - Sort first (for determinism across OS / glob orders)
    - Permute with a seeded generator
    - Split by ratio
    """
    files = sorted(all_files)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(files), generator=g)
    n_val = int(round(len(files) * val_ratio))
    val_idx = perm[:n_val].tolist()
    trn_idx = perm[n_val:].tolist()
    train_files = [files[i] for i in trn_idx]
    val_files   = [files[i] for i in val_idx]
    return train_files, val_files



class AD2Stereo(Dataset):
    """
      - Loads pairs (reverb, anechoic)
      - Builds RI STFTs
      - Upgraded to STEREO support:
          Output shapes per item:
            inp  : (4, F, T)  -> [L_real, L_imag, R_real, R_imag] for reverb
            target: (4, F, T)  -> same layout for anechoic
      - If a file is mono, we duplicate channel to keep stereo shape consistent.
    """

    def __init__(self, config, data_dir, file_list: List[str]):
        self.config = config
        self.data_dir = data_dir
        # keep exactly your mapping: list of (reverb_path, anechoic_path)
        self.pairs = [(p, p.replace("reverb", "anechoic")) for p in file_list]

        # Prepare window once (on CPU; moved to device inside __getitem__ as needed)
        self._window = None  # lazy; created on first __getitem__

    @staticmethod
    def _decode_audio(path: str, expected_type: str, sr: int) -> np.ndarray:
        """
          - wav: via soundfile
          - flac: via soundfile (supported if libsndfile is built with flac)
        Returns float32 in [-1, 1], shape (T,) or (T, C).
        """
        # soundfile returns float64 by default; always cast to float32
        audio, file_sr = sf.read(path, always_2d=False, dtype="float32")
        # shape handling: soundfile gives (T,) for mono or (T, C) for stereo
        # Resampling: assume dataset is at config.sr.
        if file_sr != sr:
            raise ValueError(f"Found sr={file_sr} in {path}, expected {sr}. Add resampling if needed.")
        return audio  # float32

    def _ensure_stereo(self, x: np.ndarray) -> np.ndarray:
        """
        Ensure shape (T, 2).
        - If mono (T,), duplicate channel.
        - If already stereo (T, 2), keep as-is.
        """
        if x.ndim == 1:  # mono
            x = np.stack([x, x], axis=-1)  # (T, 2)
        elif x.ndim == 2 and x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        elif x.ndim == 2 and x.shape[1] == 2:
            pass
        else:
            raise ValueError(f"Unsupported audio shape {x.shape}; expected (T,) or (T,2).")
        return x

    def _compute_ri(self, wav_np: np.ndarray, device="cpu") -> torch.Tensor:
        """
        RI STFT:
          - Return tensor shape: (4, F, T) for stereo -> [L_R, L_I, R_R, R_I]

        wav_np: (T, 2) float32 in [-1,1]
        """
        # lazy build window on the same device
        if self._window is None or self._window.device.type != device:
            self._window = self.config.window_tensor(device=device)

        wav = torch.from_numpy(wav_np).to(device)  # (T, 2)
        # torch.stft expects (T,) or (B,T); we’ll do per-channel then stack.
        n_fft = self.config.fft
        hop = self.config.hop
        win = self.config.win

        # Left / Right as 1D tensors
        L = wav[:, 0]
        R = wav[:, 1]

        stft_L = torch.stft(
            L,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=self._window,
            center=True,             
            return_complex=True
        )  # (F, T) complex64
        stft_R = torch.stft(
            R,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=self._window,
            center=True,
            return_complex=True
        )

        # Split to RI and pack as channels [L_R, L_I, R_R, R_I]
        L_real = stft_L.real
        L_imag = stft_L.imag
        R_real = stft_R.real
        R_imag = stft_R.imag

        # Stack to (C=4, F, T)
        ri = torch.stack([L_real, L_imag, R_real, R_imag], dim=0).to(torch.float32)
        return ri  # (4, F, T)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        reverb_path, anechoic_path = self.pairs[idx]

        # --- decode (float32), keep stereo ---
        x_rev = self._decode_audio(reverb_path, self.config.inp_type, self.config.sr)
        x_dry = self._decode_audio(anechoic_path, self.config.inp_type, self.config.sr)

        x_rev = self._ensure_stereo(x_rev)  # (T,2)
        x_dry = self._ensure_stereo(x_dry)  # (T,2)

        # (Optional) duration cropping like TF (if you used it later in pipeline)
        # samples = int(self.config.dur * self.config.sr)
        # if x_rev.shape[0] >= samples:
        #     x_rev = x_rev[:samples]
        #     x_dry = x_dry[:samples]

        # --- compute RI spectrograms (C=4, F, T) ---
        # Compute on CPU here; you can move to GPU in the training loop.
        inp_ri = self._compute_ri(x_rev, device="cpu")
        tar_ri = self._compute_ri(x_dry, device="cpu")

        return inp_ri, tar_ri



def build_dataloaders(config, data_dir, seed: int = 42, num_workers: int = 4, pin_memory: bool = True):
    """
      - Globs reverb/*.<ext>, maps to (reverb, anechoic) pairs
      - Splits train/val
      - reproducible worker seeding
      - Returns DataLoaders
    """
    pattern = os.path.join(data_dir, "reverb", f"*.{config.inp_type}")
    all_files = glob.glob(pattern)

    split_dir = os.path.join(data_dir, "_splits_ri_stereo")  # any folder you like
    train_files, val_files = make_or_load_split(all_files, split_dir, config.val_split, seed)

    train_ds = AD2Stereo(config, data_dir, train_files)
    val_ds   = AD2Stereo(config, data_dir, val_files)
    # val_ds   = AD2Stereo(config, data_dir, train_files[:10])

    # reproducible worker seeding
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=g,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=g,
        worker_init_fn=seed_worker,
    )
    return train_loader, val_loader