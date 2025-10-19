# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:17:22 2024

@author: dimak
"""

import os, random
import soundfile as sf
import numpy as np
import pyloudnorm as pyln
import pyroomacoustics as pra
from rir_generator import generate
from scipy.signal import fftconvolve

'''
def trim_audio(data, rate, ts=2):

    if rate * ts < len(data):
        # cut it
        data = data[: rate * ts]
    else:
        # add silence
        diff = rate * ts - len(data)
        data = np.pad(data, (0, diff))

    return data
'''

def trim_audio(data, rate, ts=2):
    """
    Trim or pad the audio signal to a fixed length of rate * ts samples.

    Parameters:
        data (np.ndarray): Audio signal, either mono (1D) or stereo (2D with shape (samples, channels)).
        rate (int): Sampling rate (Hz).
        ts (int): Target duration in seconds.

    Returns:
        np.ndarray: The trimmed or padded audio signal.
    """
    target_length = rate * ts

    if data.shape[0] > target_length:
        # Trim the signal to the target length.
        data = data[:target_length]
    elif data.shape[0] < target_length:
        # Calculate the number of samples to pad.
        diff = target_length - data.shape[0]
        if data.ndim == 1:
            data = np.pad(data, (0, diff))
        elif data.ndim == 2:
            # Pad only along the time axis, leaving channel dimension unchanged.
            data = np.pad(data, ((0, diff), (0, 0)))
    return data

def set_loudness(data, rate, LUFS=-24.0):

    # measure the loudness first
    meter = pyln.Meter(rate)  # create BS.1770 meter
    loudness = meter.integrated_loudness(data)

    # loudness normalize audio to -24 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, LUFS)

    return loudness_normalized_audio

'''
def detect_energy(data, threshold=0.01):

    data_abs = np.abs(data)

    if statistics.mean(data_abs) > threshold:
        is_ok = True
    else:
        is_ok = False

    return is_ok
'''

# Works for both mono and stereo
def detect_energy(data, threshold=0.01):
    # Calculate the average absolute energy over all channels
    mean_energy = np.mean(np.abs(data))
    return mean_energy > threshold



def get_common_rir(mic_pos, source_pos, room_dim, fs, absorption, max_order, ray_tracing):
    """
    Compute the impulse response (RIR) for a single mic-source pair.
    This function returns the RIR computed by pyroomacoustics.
    """
    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials=pra.Material(absorption),
        max_order=max_order,
        ray_tracing=ray_tracing
    )
    room.add_microphone_array(mic_pos)
    # We use a dummy signal here; it won't be used in the convolution.
    dummy_signal = np.zeros(1)
    room.add_source(source_pos, signal=dummy_signal)
    room.compute_rir()
    # Get the impulse response from mic 0 for source 0.
    return room.rir[0][0]

def simulate_room_for_channel(mic_pos, source_pos, audio_mono, room_dim, fs, absorption, max_order, ray_tracing):
    """
    Simulate the room response for one channel (mono signal) using pyroomacoustics.
    This function returns the simulated (convolved) signal.
    """
    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials=pra.Material(absorption),
        max_order=max_order,
        ray_tracing=ray_tracing
    )
    room.add_microphone_array(mic_pos)
    room.add_source(source_pos, signal=audio_mono.copy())
    room.compute_rir()
    room.simulate()
    signal = np.array(room.mic_array.signals)[0]
    return np.squeeze(signal)



def create_rir_conds_stereo(t60, room_dim, min_distance_to_wall, fs, audio_ex, mic_spacing=0.2):
    """
    Create stereo reverberant and dry signals where reverberation is spatially *correlated*
    (same RIR applied to both channels), to avoid artificial stereo drift.
    """
    left_audio = audio_ex[:, 0]
    right_audio = audio_ex[:, 1]

    # --- Shared mic and source for reverb field ---
    mic_center = np.array([
        np.random.uniform(min_distance_to_wall, room_dim[n] - min_distance_to_wall)
        for n in range(3)
    ])
    source_pos = np.array([
        np.random.uniform(min_distance_to_wall, room_dim[n] - min_distance_to_wall)
        for n in range(3)
    ])

    # Use one RIR for both channels to ensure realistic stereo balance
    shared_mic = mic_center.reshape(3, 1)
    absorption, max_order = pra.inverse_sabine(t60, room_dim)
    shared_rir = get_common_rir(shared_mic, source_pos, room_dim, fs, absorption, max_order, ray_tracing=True)

    rev_left = fftconvolve(left_audio, shared_rir, mode="full")[:len(left_audio)]
    rev_right = fftconvolve(right_audio, shared_rir, mode="full")[:len(right_audio)]
    reverberant_stereo = np.vstack([rev_left, rev_right])

    # --- Fixed mic-source for dry version (no stereo drift) ---
    mic_center_dry = np.array([2.0, 1.5, 1.2])
    source_dry = np.array([2.0, 2.0, 1.2])
    left_mic_dry = mic_center_dry.copy(); left_mic_dry[0] -= mic_spacing / 2.0
    right_mic_dry = mic_center_dry.copy(); right_mic_dry[0] += mic_spacing / 2.0

    rir_l_dry = get_common_rir(left_mic_dry.reshape(3, 1), source_dry, room_dim, fs, 0.99, 0, ray_tracing=False)
    rir_r_dry = get_common_rir(right_mic_dry.reshape(3, 1), source_dry, room_dim, fs, 0.99, 0, ray_tracing=False)

    dry_left = fftconvolve(left_audio, rir_l_dry, mode="full")[:len(left_audio)]
    dry_right = fftconvolve(right_audio, rir_r_dry, mode="full")[:len(right_audio)]
    dry_stereo = np.vstack([dry_left, dry_right])

    # --- Match length ---
    min_len = min(reverberant_stereo.shape[1], dry_stereo.shape[1])
    return reverberant_stereo[:, :min_len], dry_stereo[:, :min_len]




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

def create_rir_generator_stereo(t60, room_dim, min_distance_to_wall, fs, audio_ex):
    """
    Create stereo reverberant and dry examples using rir-generator,
    with a shared RIR applied to both channels for realistic spatial behavior.

    Parameters:
        t60 (float): Reverberation time in seconds.
        room_dim (tuple): Room dimensions (LxWxH) in meters.
        min_distance_to_wall (float): Minimum distance of mic and source from walls.
        fs (int): Sampling rate in Hz.
        audio_ex (ndarray): Stereo input audio of shape (samples, 2).

    Returns:
        Tuple (reverberant_stereo, dry_stereo) each of shape (2, N).
    """
    if audio_ex.ndim != 2 or audio_ex.shape[1] != 2:
        raise ValueError("Input audio must be stereo with shape (samples, 2)")

    left_audio = audio_ex[:, 0]
    right_audio = audio_ex[:, 1]

    # --- Shared mic and source positions ---
    mic_pos = [
        np.random.uniform(min_distance_to_wall, room_dim[n] - min_distance_to_wall)
        for n in range(3)
    ]
    source_pos = [
        np.random.uniform(min_distance_to_wall, room_dim[n] - min_distance_to_wall)
        for n in range(3)
    ]

    # --- RIR parameters ---
    rir_len = min(int(fs * np.random.uniform(t60, 2 * t60)), int(fs * 0.5))

    # --- Reverberant RIR (shared) ---
    rir_rev = generate(c=340, fs=fs, r=mic_pos, s=source_pos, L=room_dim,
                       reverberation_time=t60, nsample=rir_len).squeeze()

    rev_left = fftconvolve(left_audio, rir_rev, mode="full")[:len(left_audio)]
    rev_right = fftconvolve(right_audio, rir_rev, mode="full")[:len(right_audio)]
    reverberant_stereo = np.vstack([rev_left, rev_right])

    # --- Dry RIR (shared, short, high absorption) ---
    rir_dry = generate(c=340, fs=fs, r=mic_pos, s=source_pos, L=room_dim,
                       reverberation_time=0.3, nsample=int(fs * 0.03)).squeeze()

    dry_left = fftconvolve(left_audio, rir_dry, mode="full")[:len(left_audio)]
    dry_right = fftconvolve(right_audio, rir_dry, mode="full")[:len(right_audio)]
    dry_stereo = np.vstack([dry_left, dry_right])

    # --- Match lengths ---
    min_len = min(reverberant_stereo.shape[1], dry_stereo.shape[1])
    return reverberant_stereo[:, :min_len], dry_stereo[:, :min_len]


def create_rir_conds_openair(fs, audio_ex, rir_folder, mix_range=(0.7, 1.0)):
    """
    Apply a random stereo OpenAIR RIR to a stereo audio input, with adjustable wet/dry mix.

    Parameters:
        fs (int): Sample rate (must match RIR)
        audio_ex (np.ndarray): Stereo input audio, shape (samples, 2)
        rir_folder (str): Path to processed OpenAIR RIRs (must be stereo, 44.1kHz)
        mix_range (tuple): Range of alpha (wet signal contribution), e.g., (0.7, 1.0)

    Returns:
        (reverberant_stereo, dry_stereo)
    """
    # Collect all valid .wav RIRs recursively
    rir_paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(rir_folder)
        for f in files
        if f.endswith(".wav")
    ]
    if not rir_paths:
        raise ValueError(f"No RIRs found in {rir_folder}")

    # Pick random RIR
    rir_path = random.choice(rir_paths)
    rir, rir_sr = sf.read(rir_path)
    if rir_sr != fs:
        raise ValueError(f"RIR sample rate {rir_sr} does not match expected {fs}")
    if rir.ndim != 2 or rir.shape[1] != 2:
        raise ValueError(f"Expected stereo RIR, got shape {rir.shape}")

    # Convolve both channels
    rev_left = fftconvolve(audio_ex[:, 0], rir[:, 0], mode="full")[:len(audio_ex)]
    rev_right = fftconvolve(audio_ex[:, 1], rir[:, 1], mode="full")[:len(audio_ex)]
    reverb = np.vstack([rev_left, rev_right])

    # Mix dry + reverb
    alpha = np.random.uniform(*mix_range)
    dry = np.swapaxes(audio_ex, 0, 1)
    output = alpha * reverb + (1 - alpha) * dry

    return output, dry




def create_rir_conds_rir_generator(t60, room_dim, min_distance_to_wall, fs, audio_ex):
    """
    Create reverberant and dry audio conditions using RIR-Generator.
    """
    # Sample microphone and source positions
    mic_position = [
        np.random.uniform(min_distance_to_wall, room_dim[n] - min_distance_to_wall)
        for n in range(3)
    ]
    source_position = [
        np.random.uniform(min_distance_to_wall, room_dim[n] - min_distance_to_wall)
        for n in range(3)
    ]

    # Randomize RIR length
    rir_length = int(
        fs * np.random.uniform(t60, t60 * 2)  # Randomize within [t60, t60 * 2]
    )
    rir_length = min(rir_length, int(fs * 0.5))  # Cap at 0.5 seconds max
    
    rir = generate(
        c=340, fs=fs, r=mic_position, s=source_position, L=room_dim, reverberation_time=t60, nsample=rir_length
    )
    rir = rir.squeeze()  # Ensure RIR is 1D
    assert rir.ndim == 1, f"RIR must be 1D, but got shape {rir.shape}."

    # Apply RIR
    lossy_ex = fftconvolve(audio_ex, rir, mode="full")[:len(audio_ex)]

    # Use the clean input for dry audio
    #dry_ex = audio_ex.copy()
    # Recreate anechoic input based on room dimensions
    dry_rir = generate(
        c=340,
        fs=fs,
        r=mic_position,
        s=source_position,
        L=room_dim,
        reverberation_time=0.3,  # Simulate a highly absorbing room
        nsample=int(fs * 0.03),  # Very short RIR length for anechoic conditions
    )
    dry_rir = dry_rir.squeeze()
    assert dry_rir.ndim == 1, f"Dry RIR must be 1D, but got shape {dry_rir.shape}."

    # Apply the dry RIR
    dry_ex = fftconvolve(audio_ex, dry_rir, mode="full")[:len(audio_ex)]

    # Add noise floor to the dry signal
    #noise_floor_snr = 50
    #noise_floor_power = (
    #    1 / dry_ex.shape[0] * np.sum(dry_ex**2) * np.power(10, -noise_floor_snr / 10)
    #)
    #noise_floor_signal = np.random.rand(int(0.5 * fs)) * np.sqrt(noise_floor_power)
    #dry_ex = np.concatenate([dry_ex, noise_floor_signal])

    # Ensure equal lengths
    min_length = min(len(lossy_ex), len(dry_ex))
    return lossy_ex[:min_length], dry_ex[:min_length]