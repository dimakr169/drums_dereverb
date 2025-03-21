# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:17:22 2024

@author: dimak
"""

import statistics

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

def create_rir_conds_stereo(t60, room_dim, min_distance_to_wall, fs, audio_ex,
                            stereo_ratio=0.7, mic_spacing=0.2):
    """
    Create stereo examples (reverberant and dry) using two modes:
      - 70% chance (common mode): apply the same RIR to each channel separately,
        so the original stereo differences in audio_ex are preserved.
      - 30% chance (different mode): simulate each channel separately and blend the right channel.
      
    Parameters:
      t60: Desired reverberation time for the reverberant simulation.
      room_dim: Room dimensions (length, width, height).
      min_distance_to_wall: Minimum distance from any wall for mic/source.
      fs: Sampling frequency.
      audio_ex: Stereo input signal of shape (samples, 2).
      stereo_ratio: In different mode, right channel = stereo_ratio * left + (1-stereo_ratio)*right.
      mic_spacing: Distance between the left and right microphones.
      
    Returns:
      Tuple (reverberant_stereo, dry_stereo) each of shape (2, N).
    """
    # --- Split the stereo input into left and right channels ---
    left_audio = audio_ex[:, 0]
    right_audio = audio_ex[:, 1]

    # --- Sample source and central microphone positions ---
    center_mic_position = np.array([
        np.random.uniform(min_distance_to_wall, room_dim[n] - min_distance_to_wall)
        for n in range(3)
    ])
    source_position = np.array([
        np.random.uniform(min_distance_to_wall, room_dim[n] - min_distance_to_wall)
        for n in range(3)
    ])

    # --- Define left and right microphone positions ---
    left_mic = center_mic_position.copy()
    right_mic = center_mic_position.copy()
    left_mic[0] -= mic_spacing / 2.0
    right_mic[0] += mic_spacing / 2.0

    # --- Reverberant Simulation ---
    e_absorption, max_order = pra.inverse_sabine(t60, room_dim)
    if np.random.rand() < 0.7:
        # 70% chance: use a common RIR.
        common_rir = get_common_rir(left_mic.reshape(3, 1), source_position,
                                    room_dim, fs, e_absorption, max_order, ray_tracing=True)
        # Convolve each channel separately so that the original stereo differences remain.
        left_rev = fftconvolve(left_audio, common_rir)[:len(left_audio)]
        right_rev = fftconvolve(right_audio, common_rir)[:len(right_audio)]
    else:
        # 30% chance: simulate each channel separately.
        left_rev = simulate_room_for_channel(left_mic.reshape(3, 1), source_position,
                                             left_audio, room_dim, fs, e_absorption, max_order, ray_tracing=True)
        right_rev = simulate_room_for_channel(right_mic.reshape(3, 1), source_position,
                                              right_audio, room_dim, fs, e_absorption, max_order, ray_tracing=True)
        # Ensure same length and blend right channel slightly.
        min_len = min(len(left_rev), len(right_rev))
        left_rev = left_rev[:min_len]
        right_rev = right_rev[:min_len]
        right_rev = stereo_ratio * left_rev + (1 - stereo_ratio) * right_rev

    reverberant_stereo = np.vstack([left_rev, right_rev])

    # --- Dry (Anechoic) Simulation ---
    # For the dry case, you might want to preserve the stereo image even more closely.
    # Here we follow the same common/different mode strategy.
    if np.random.rand() < 0.7:
        common_rir_dry = get_common_rir(left_mic.reshape(3, 1), source_position,
                                        room_dim, fs, 0.99, 0, ray_tracing=False)
        left_dry = fftconvolve(left_audio, common_rir_dry)[:len(left_audio)]
        right_dry = fftconvolve(right_audio, common_rir_dry)[:len(right_audio)]
    else:
        left_dry = simulate_room_for_channel(left_mic.reshape(3, 1), source_position,
                                             left_audio, room_dim, fs, 0.99, 0, ray_tracing=False)
        right_dry = simulate_room_for_channel(right_mic.reshape(3, 1), source_position,
                                              right_audio, room_dim, fs, 0.99, 0, ray_tracing=False)
        min_len = min(len(left_dry), len(right_dry))
        left_dry = left_dry[:min_len]
        right_dry = right_dry[:min_len]
        right_dry = stereo_ratio * left_dry + (1 - stereo_ratio) * right_dry

    dry_stereo = np.vstack([left_dry, right_dry])

    # --- Optional: Add Noise Floor to Dry Signal ---
    noise_floor_snr = 50  # dB
    noise_power_left = np.mean(left_dry**2) * np.power(10, -noise_floor_snr / 10)
    noise_left = np.random.rand(int(0.5 * fs)) * np.sqrt(noise_power_left)
    left_dry = np.concatenate([left_dry, noise_left])
    noise_power_right = np.mean(right_dry**2) * np.power(10, -noise_floor_snr / 10)
    noise_right = np.random.rand(int(0.5 * fs)) * np.sqrt(noise_power_right)
    right_dry = np.concatenate([right_dry, noise_right])
    dry_stereo = np.vstack([left_dry, right_dry])

    # --- Trim both outputs to the same length ---
    min_len_total = min(reverberant_stereo.shape[1], dry_stereo.shape[1])
    reverberant_stereo = reverberant_stereo[:, :min_len_total]
    dry_stereo = dry_stereo[:, :min_len_total]

    return reverberant_stereo, dry_stereo





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

def create_rir_generator_stereo(t60, room_dim, min_distance_to_wall, fs, audio_ex, stereo_offset=0.2):
    """
    Create stereo reverberant (lossy) and dry audio conditions using RIR-Generator.

    Parameters:
        t60: Reverberation time in seconds.
        room_dim: A list or tuple with 3 room dimensions (meters).
        min_distance_to_wall: Minimum distance from the walls (meters).
        fs: Sampling frequency (Hz).
        audio_ex: Stereo input audio signal as a NumPy array of shape (samples, 2).
        stereo_offset: Distance between left and right microphone positions (meters).
    
    Returns:
        A tuple (lossy_ex, dry_ex) where each is a stereo NumPy array with shape (samples, 2).
    """
    # Verify that the input is stereo.
    if audio_ex.ndim != 2 or audio_ex.shape[1] != 2:
        raise ValueError("Input audio must be a stereo signal with shape (samples, 2).")
    
    # Sample a base microphone position and a source position uniformly in the room.
    base_mic = [
        np.random.uniform(min_distance_to_wall, room_dim[n] - min_distance_to_wall)
        for n in range(3)
    ]
    source_position = [
        np.random.uniform(min_distance_to_wall, room_dim[n] - min_distance_to_wall)
        for n in range(3)
    ]
    
    # Create separate microphone positions for left and right channels by offsetting the base position.
    left_mic = base_mic.copy()
    right_mic = base_mic.copy()
    left_mic[0] -= stereo_offset / 2
    right_mic[0] += stereo_offset / 2
    
    # Determine a randomized RIR length within [t60, 2*t60] and cap at 0.5 seconds.
    rir_length = int(fs * np.random.uniform(t60, t60 * 2))
    rir_length = min(rir_length, int(fs * 0.5))
    
    # Generate impulse responses (RIRs) for each channel using the RIR-Generator's generate() function.
    left_rir = generate(
        c=340, fs=fs, r=left_mic, s=source_position,
        L=room_dim, reverberation_time=t60, nsample=rir_length
    ).squeeze()
    assert left_rir.ndim == 1, f"Left RIR must be 1D, but got shape {left_rir.shape}."
    
    right_rir = generate(
        c=340, fs=fs, r=right_mic, s=source_position,
        L=room_dim, reverberation_time=t60, nsample=rir_length
    ).squeeze()
    assert right_rir.ndim == 1, f"Right RIR must be 1D, but got shape {right_rir.shape}."
    
    # Apply the RIR to each channel separately.
    lossy_ex_left = fftconvolve(audio_ex[:, 0], left_rir, mode="full")[:audio_ex.shape[0]]
    lossy_ex_right = fftconvolve(audio_ex[:, 1], right_rir, mode="full")[:audio_ex.shape[0]]
    lossy_ex = np.stack((lossy_ex_left, lossy_ex_right), axis=1)
    
    # Generate short 'dry' RIRs to simulate anechoic (mostly absorption) conditions.
    left_dry_rir = generate(
        c=340, fs=fs, r=left_mic, s=source_position,
        L=room_dim, reverberation_time=0.3, nsample=int(fs * 0.03)
    ).squeeze()
    assert left_dry_rir.ndim == 1, f"Left dry RIR must be 1D, but got shape {left_dry_rir.shape}."
    
    right_dry_rir = generate(
        c=340, fs=fs, r=right_mic, s=source_position,
        L=room_dim, reverberation_time=0.3, nsample=int(fs * 0.03)
    ).squeeze()
    assert right_dry_rir.ndim == 1, f"Right dry RIR must be 1D, but got shape {right_dry_rir.shape}."
    
    # Convolve the dry RIRs with the corresponding channels.
    dry_ex_left = fftconvolve(audio_ex[:, 0], left_dry_rir, mode="full")[:audio_ex.shape[0]]
    dry_ex_right = fftconvolve(audio_ex[:, 1], right_dry_rir, mode="full")[:audio_ex.shape[0]]
    dry_ex = np.stack((dry_ex_left, dry_ex_right), axis=1)
    
    # Ensure both outputs have equal length (they should, but this is an extra safeguard).
    min_length = min(lossy_ex.shape[0], dry_ex.shape[0])
    return lossy_ex[:min_length, :], dry_ex[:min_length, :]




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