# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:12:49 2024

@author: dimak
"""

import tensorflow as tf


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
        self.t60_r = [0.4, 1.2]  # Range for reverb time in seconds
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
        self.hop = (
            341  # for preventing shape mismatchin in UNet encoding-decoding block
        )
        self.win = 1022
        self.fft = self.win
        self.win_fn = "hann"

        # training
        self.rep_type = "ri"
        # Available representations
        # 'mag_phase': for cold diffusion UNet with magnitude only
        # 'ri_mag_phase': for GaGNet with Real and Imaginary Parts enchanced with magnitude and phase
        # 'ri': for cold diffusion UNet, DCUNet, DCCRN with Real and Imaginary Parts
        self.val_split = 0.2
        self.batch_size = 6  # 24

    def window_fn(self):
        """Return window generator.
        Returns:
            Callable, window function of tf.signal
                , which corresponds to self.win_fn.
        """
        mapper = {"hann": tf.signal.hann_window, "hamming": tf.signal.hamming_window}
        if self.win_fn in mapper:
            return mapper[self.win_fn]

        raise ValueError("invalid window function: " + self.win_fn)
