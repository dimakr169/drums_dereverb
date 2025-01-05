# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:47:47 2024

@author: dimak
"""

import glob
import os

import tensorflow as tf
import tensorflow_io as tfio
from sklearn.model_selection import train_test_split


class AD2:
    """AD2 dataset loader."""

    def __init__(self, config, data_dir):
        """Initializer.
        Args:
            config: Config, dataset configuration.
            data_dir: str, dataset directory
        """
        self.config = config
        self.data_dir = data_dir

    def split_data(self, data_dir):

        # load paths and split them
        all_files = glob.glob(
            os.path.join(data_dir, "reverb", "*." + self.config.inp_type)
        )  # flac or wav
        train_set, val_set = train_test_split(
            all_files, test_size=self.config.val_split
        )

        return train_set, val_set

    def get_paths(self, data_split):
        """Create paths for reverb and anechoic tracks.
        Args:
            data_split: str, dataset directory.
        Returns:
            tf.data.Dataset, data loader.
        """

        paths = [(inp, inp.replace("reverb", "anechoic")) for inp in data_split]
        return tf.data.Dataset.from_tensor_slices(paths)

    def decode_audio(self, file_path):
        """Load audio with tf apis.
        Args:
            path: str, wavfile or flac path to read.
        Returns:
            tf.Tensor, [T], mono audio in range (-1, 1).
        """
        if self.config.inp_type == "wav":
            audio_binary = tf.io.read_file(file_path)
            audio, _ = tf.audio.decode_wav(audio_binary)
        else:  # flac
            # audio_io = tfio.audio.AudioIOTensor(file_path, dtype=tf.int16)
            # audio = audio_io.to_tensor()
            audio_binary = tf.io.read_file(file_path)
            audio = tfio.audio.decode_flac(audio_binary, dtype=tf.int16)
            # convert to tf.float32
            audio = tf.cast(audio, tf.float32) / 32768.0

        return tf.squeeze(audio)

    def compute_mag_phase(self, signal):
        """Compute magnitude and phase from STFT for UNet."""
        signal_stft = tf.signal.stft(
            signal,
            frame_length=self.config.win,
            frame_step=self.config.hop,
            fft_length=self.config.fft,
            window_fn=self.config.window_fn(),
        )
        # get magnitude/phase
        mag = tf.abs(signal_stft)
        phase = tf.math.angle(signal_stft)

        # expand dims
        mag = tf.expand_dims(mag, axis=-1)
        phase = tf.expand_dims(phase, axis=-1)

        return tf.cast(mag, tf.float32), tf.cast(phase, tf.float32)

    def compute_ri_mag_phase(self, inp_signal, tar_signal):
        """Compute RI STFT with mag and phase for GaGNet."""

        # create stfts
        noisy_stft = tf.signal.stft(
            inp_signal,
            frame_length=self.config.win,
            frame_step=self.config.hop,
            fft_length=self.config.fft,
            window_fn=self.config.window_fn(),
        )

        noisy_stft_real = tf.math.real(noisy_stft)
        noisy_stft_imag = tf.math.imag(noisy_stft)
        # Concatenate real and imaginary parts along the last dimension
        noisy_stft = tf.stack([noisy_stft_real, noisy_stft_imag], axis=-1)

        target_stft = tf.signal.stft(
            tar_signal,
            frame_length=self.config.win,
            frame_step=self.config.hop,
            fft_length=self.config.fft,
            window_fn=self.config.window_fn(),
        )
        target_stft_real = tf.math.real(target_stft)
        target_stft_imag = tf.math.imag(target_stft)
        # Concatenate real and imaginary parts along the last dimension
        target_stft = tf.stack(
            [target_stft_real, target_stft_imag], axis=-1
        )  # [B,F,C,2]

        # permute inp[B,2,C,F] and tar[B,2,F,C]
        noisy_stft, target_stft = tf.transpose(
            noisy_stft, perm=[2, 0, 1]
        ), tf.transpose(target_stft, perm=[2, 1, 0])

        # Calculate magnitude (sqrt-compressed) and phase
        noisy_mag = tf.sqrt(tf.norm(noisy_stft, axis=0))
        noisy_phase = tf.atan2(noisy_stft[1, ...], noisy_stft[0, ...])

        target_mag = tf.sqrt(tf.norm(target_stft, axis=0))
        target_phase = tf.atan2(target_stft[1, ...], target_stft[0, ...])

        noisy_stft = tf.stack(
            [noisy_mag * tf.cos(noisy_phase), noisy_mag * tf.sin(noisy_phase)], axis=0
        )
        target_stft = tf.stack(
            [target_mag * tf.cos(target_phase), target_mag * tf.sin(target_phase)],
            axis=0,
        )

        return tf.cast(noisy_stft, tf.float32), tf.cast(target_stft, tf.float32)

    def compute_ri(self, signal):
        """Compute RI STFTs for UNet RI, DCCRN, DCUNet."""
        signal_stft = tf.signal.stft(
            signal,
            frame_length=self.config.win,
            frame_step=self.config.hop,
            fft_length=self.config.fft,
            window_fn=self.config.window_fn(),
        )

        signal_stft_real = tf.math.real(signal_stft)
        signal_stft_imag = tf.math.imag(signal_stft)

        # create a new dimension also for UNet
        signal_stft_ri = tf.stack([signal_stft_real, signal_stft_imag], axis=-1)

        return tf.cast(signal_stft_ri, tf.float32)

    def process_path(self, file_paths):
        """Preprocessing pipeline.

        3 different input representations:
            'mag_phase': for cold diffusion UNet with magnitude only
            'ri_mag_phase': for GaGNet with Real and Imaginary Parts enchanced with magnitude and phase
            'ri': for cold diffusion UNet, DCUNet, DCCRN with Real and Imaginary Parts

        """

        reverb_path = file_paths[0]
        anechoic_path = file_paths[1]
        reverb_audio = self.decode_audio(reverb_path)
        anechoic_audio = self.decode_audio(anechoic_path)
        if self.config.rep_type == "mag_phase":
            reverb_mag, reverb_phase = self.compute_mag_phase(reverb_audio)
            anechoic_mag, anechoic_phase = self.compute_mag_phase(anechoic_audio)

            return (reverb_mag, reverb_phase, anechoic_mag, anechoic_phase)

        elif self.config.rep_type == "ri_mag_phase":
            return self.compute_ri_mag_phase(reverb_audio, anechoic_audio)

        elif self.config.rep_type == "ri":
            reverb_ri_stft = self.compute_ri(reverb_audio)
            anechoic_ri_stft = self.compute_ri(anechoic_audio)

            return (reverb_ri_stft, anechoic_ri_stft)

    def prepare_dataset(self, dataset):
        """Mapping to tf.dataset."""

        dataset = dataset.map(self.process_path, num_parallel_calls=tf.data.AUTOTUNE)
        # dataset = dataset.cache() #not for large datasets
        dataset = dataset.batch(self.config.batch_size, drop_remainder=False)
        dataset = dataset.shuffle(10 * self.config.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def create_datasets(self):
        """Create train and validation datasets."""

        train_files, val_files = self.split_data(self.data_dir)
        train_dataset = self.prepare_dataset(self.get_paths(train_files))
        val_dataset = self.prepare_dataset(self.get_paths(val_files))

        return train_dataset, val_dataset
