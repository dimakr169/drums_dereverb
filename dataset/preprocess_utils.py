# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:17:22 2024

@author: dimak
"""

import statistics

import numpy as np
import pyloudnorm as pyln
import pyroomacoustics as pra


def trim_audio(data, rate, ts=2):

    if rate * ts < len(data):
        # cut it
        data = data[: rate * ts]
    else:
        # add silence
        diff = rate * ts - len(data)
        data = np.pad(data, (0, diff))

    return data


def set_loudness(data, rate, LUFS=-24.0):

    # measure the loudness first
    meter = pyln.Meter(rate)  # create BS.1770 meter
    loudness = meter.integrated_loudness(data)

    # loudness normalize audio to -24 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, LUFS)

    return loudness_normalized_audio


def detect_energy(data, threshold=0.01):

    data_abs = np.abs(data)

    if statistics.mean(data_abs) > threshold:
        is_ok = True
    else:
        is_ok = False

    return is_ok


def create_rir_conds(t60, room_dim, min_distance_to_wall, fs, audio_ex):

    # sample microphone position
    center_mic_position = np.array(
        [
            np.random.uniform(min_distance_to_wall, room_dim[n] - min_distance_to_wall)
            for n in range(3)
        ]
    )
    # sample source position
    source_position = np.array(
        [
            np.random.uniform(min_distance_to_wall, room_dim[n] - min_distance_to_wall)
            for n in range(3)
        ]
    )
    #
    # distance_source = 1/np.sqrt(center_mic_position.ndim)*np.linalg.norm(center_mic_position - source_position)
    mic_array_2d = pra.beamforming.circular_2D_array(
        center_mic_position[:-1], 1, phi0=0, radius=1.0
    )  # Compute microphone array
    mic_array = np.pad(
        mic_array_2d,
        ((0, 1), (0, 0)),
        mode="constant",
        constant_values=center_mic_position[-1],
    )

    # Reverberant Room
    e_absorption, max_order = pra.inverse_sabine(
        t60, room_dim
    )  # Compute absorption coeff
    reverberant_room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials=pra.Material(e_absorption),
        max_order=min(3, max_order),
        ray_tracing=True,
    )
    # Create room
    reverberant_room.set_ray_tracing()
    # Add microphone array
    reverberant_room.add_microphone_array(mic_array)
    # Generate reverberant room
    reverberant_room.add_source(source_position, signal=audio_ex.copy())
    reverberant_room.compute_rir()
    reverberant_room.simulate()
    # t60_real = np.mean(reverberant_room.measure_rt60()).squeeze()
    lossy_ex = np.squeeze(np.array(reverberant_room.mic_array.signals))

    # compute target
    e_absorption_dry = 0.99
    dry_room = pra.ShoeBox(
        room_dim, fs=fs, materials=pra.Material(e_absorption_dry), max_order=0
    )  # Create room
    dry_room.add_microphone_array(mic_array)  # Add microphone array

    # Generate dry room
    dry_room.add_source(source_position, signal=audio_ex.copy())
    dry_room.compute_rir()
    dry_room.simulate()
    # t60_real_dry = np.mean(dry_room.measure_rt60()).squeeze()
    speech = np.squeeze(np.array(dry_room.mic_array.signals))
    noise_floor_snr = 50
    noise_floor_power = (
        1 / speech.shape[0] * np.sum(speech**2) * np.power(10, -noise_floor_snr / 10)
    )
    noise_floor_signal = np.random.rand(int(0.5 * fs)) * np.sqrt(noise_floor_power)
    dry_ex = np.concatenate([speech, noise_floor_signal])

    min_length = min(lossy_ex.shape[0], dry_ex.shape[0])
    lossy_ex, dry_ex = lossy_ex[:min_length], dry_ex[:min_length]

    return lossy_ex, dry_ex
