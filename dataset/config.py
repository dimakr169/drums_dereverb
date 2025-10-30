# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:12:49 2024

@author: dimak
"""

# import tensorflow as tf
import torch


class Config:
    """Configuration for dataset construction."""

    def __init__(self):
        # audio config
        self.inp_type = "wav"  # 'wav' or 'flac'
        self.sr = 44100  # sample rate
        self.dur = 2  # duration in seconds
        self.lufs = -28.0  # for audio normalizing
        self.threshold = 0.0001  # for energy threshold

        # RIR parameters
        # following paper: https://arxiv.org/abs/2212.11851
        self.t60_r = [0.4, 1.5]  # Range for reverb time in seconds
        self.room_dim_r = [
            5,
            15,
            5,
            15,
            2,
            6,
        ]  # Range 5 to 15 meters length-width, 2 to 6 for height
        self.min_distance_to_wall = 1.0  # for mic and source positions

        # Augmentations
        self.aug_factor = 3 # apply augmentations for each slice

        # stft
        self.hop = 384
        self.win = 1024 
        self.fft = self.win
        self.win_fn = "hann"

        # window for ISTFT
        self.center = True

        # training
        self.rep_type = "ri"
        # Available representations
        # 'mag_phase': for cold diffusion UNet with magnitude only
        # 'ri_mag_phase': for GaGNet with Real and Imaginary Parts enchanced with magnitude and phase
        # 'ri': for cold diffusion UNet, DCUNet, DCCRN with Real and Imaginary Parts
        self.val_split = 0.2
        self.batch_size = 4  # 24

    # window factory 
    def window_tensor(self, device="cpu"):
        if self.win_fn == "hann":
            # TF uses periodic=True by default. Match PyTorch periodic=True.
            return torch.hann_window(self.win, periodic=True, device=device, dtype=torch.float32)
        elif self.win_fn == "hamming":
            return torch.hamming_window(self.win, periodic=True, device=device, dtype=torch.float32)
        raise ValueError(f"invalid window function: {self.win_fn}")
