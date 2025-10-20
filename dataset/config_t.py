import torch
from dataclasses import dataclass

@dataclass
class ConfigTorch:
    # --- audio config ---
    inp_type: str = "wav"   # 'wav' or 'flac'
    sr: int = 44100
    dur: int = 2
    lufs: float = -28.0
    threshold: float = 1e-4

    # --- RIR params ---
    t60_r: tuple = (0.4, 1.5)
    room_dim_r: tuple = (5, 15, 5, 15, 2, 6)
    min_distance_to_wall: float = 1.0

    # --- augmentations  ---
    aug_factor: int = 3

    # --- STFT params  ---
    hop: int = 177
    win: int = 510
    fft: int = 510
    win_fn: str = "hann"  # 'hann' or 'hamming'

    # --- training/data ---
    rep_type: str = "ri"   # we’ll honor only 'ri' as requested
    val_split: float = 0.2
    batch_size: int = 4

    # window factory 
    def window_tensor(self, device="cpu"):
        if self.win_fn == "hann":
            # TF uses periodic=True by default. Match PyTorch periodic=True.
            return torch.hann_window(self.win, periodic=True, device=device, dtype=torch.float32)
        elif self.win_fn == "hamming":
            return torch.hamming_window(self.win, periodic=True, device=device, dtype=torch.float32)
        raise ValueError(f"invalid window function: {self.win_fn}")