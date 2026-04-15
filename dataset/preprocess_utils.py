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
from scipy.signal import fftconvolve


def trim_audio(data, rate, ts=2):
    """
    Trim or pad the audio signal to a fixed length of rate * ts samples.
    """
    # ts may be float (e.g., 10.0s context windows)
    target_length = int(rate * ts)

    if data.shape[0] > target_length:
        data = data[:target_length]
    elif data.shape[0] < target_length:
        diff = target_length - data.shape[0]
        if data.ndim == 1:
            data = np.pad(data, (0, diff))
        elif data.ndim == 2:
            data = np.pad(data, ((0, diff), (0, 0)))
    return data

def set_loudness(data, rate, LUFS=-24.0):

    # measure the loudness first
    meter = pyln.Meter(rate)  # create BS.1770 meter
    loudness = meter.integrated_loudness(data)

    # loudness normalize audio to -24 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, LUFS)

    return loudness_normalized_audio

def _align_rir_to_direct_peak(rir: np.ndarray) -> np.ndarray:
    """Shift an RIR left so its largest-magnitude sample is at index 0."""
    if rir.ndim != 1 or rir.size == 0:
        return rir
    t0 = int(np.argmax(np.abs(rir)))
    if t0 <= 0:
        return rir
    out = np.zeros_like(rir)
    out[: rir.size - t0] = rir[t0:]
    return out

def _early_lr_diff_db(rir_stereo: np.ndarray, fs: int, early_ms: float = 20.0, eps: float = 1e-8) -> float:
    """
    Measure left/right early-energy mismatch in dB.
    Used to reject strongly panned / imbalanced measured RIRs.
    """
    n = max(1, int(fs * early_ms / 1000.0))
    early = rir_stereo[: min(n, rir_stereo.shape[0]), :]

    rms_l = np.sqrt(np.mean(early[:, 0] ** 2) + eps)
    rms_r = np.sqrt(np.mean(early[:, 1] ** 2) + eps)

    return float(abs(20.0 * np.log10(rms_l / rms_r)))

def _align_stereo_rir_per_channel(rir_stereo: np.ndarray) -> np.ndarray:
    """
    Align left/right channels independently to remove ITD.
    """
    if rir_stereo.ndim != 2 or rir_stereo.shape[1] != 2:
        return rir_stereo

    rir_l = _align_rir_to_direct_peak(rir_stereo[:, 0].astype(np.float32))
    rir_r = _align_rir_to_direct_peak(rir_stereo[:, 1].astype(np.float32))

    n = min(len(rir_l), len(rir_r))
    return np.stack([rir_l[:n], rir_r[:n]], axis=1).astype(np.float32)


def _align_stereo_rir_preserve_itd(rir_stereo: np.ndarray) -> np.ndarray:
    """Align stereo RIR by earliest direct-path peak (preserves ITD)."""
    if rir_stereo.ndim != 2 or rir_stereo.shape[1] != 2:
        return rir_stereo
    tL = int(np.argmax(np.abs(rir_stereo[:, 0])))
    tR = int(np.argmax(np.abs(rir_stereo[:, 1])))
    t0 = min(tL, tR)
    if t0 <= 0:
        return rir_stereo
    out = np.zeros_like(rir_stereo)
    out[: rir_stereo.shape[0] - t0, :] = rir_stereo[t0:, :]
    return out

def _normalize_stereo_rir_common(
    rir_stereo: np.ndarray,
    fs: int,
    early_ms: float = 50.0,
    mode: str = "rms",
    target: float = 0.1,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Apply ONE common scalar to both channels.
    Use only the early part to compute the normalization reference.
    """
    n = max(1, int(fs * early_ms / 1000.0))
    early = rir_stereo[: min(n, rir_stereo.shape[0]), :]

    if mode == "peak":
        ref = float(np.max(np.abs(early)))
    elif mode == "rms":
        ref = float(np.sqrt(np.mean(early ** 2) + eps))
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")

    if ref > eps:
        rir_stereo = rir_stereo * (target / ref)

    return rir_stereo.astype(np.float32, copy=False)


def _load_random_valid_openair_rir(
    rir_folder,
    fs: int,
    max_tries: int = 20,
    max_early_lr_diff_db: float = 4.0,
    remove_itd: bool = True,
    norm_mode: str = "rms",
    norm_target: float = 0.1,
    norm_early_ms: float = 50.0,
):
    """
    Load a random stereo RIR, reject extreme stereo imbalances,
    optionally remove ITD, and apply common normalization.
    """
    rir_paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(rir_folder)
        for f in files
        if f.endswith(".wav")
    ]
    if not rir_paths:
        raise ValueError(f"No RIRs found in {rir_folder}")

    last_error = None

    for _ in range(max_tries):
        rir_path = random.choice(rir_paths)

        try:
            rir, rir_sr = sf.read(rir_path)
            if rir_sr != fs:
                continue
            if rir.ndim != 2 or rir.shape[1] != 2:
                continue

            rir = rir.astype(np.float32, copy=False)

            if remove_itd:
                rir = _align_stereo_rir_per_channel(rir)

            lr_diff_db = _early_lr_diff_db(rir, fs=fs, early_ms=20.0)
            if lr_diff_db > max_early_lr_diff_db:
                continue

            rir = _normalize_stereo_rir_common(
                rir,
                fs=fs,
                early_ms=norm_early_ms,
                mode=norm_mode,
                target=norm_target,
            )
            return rir

        except Exception as e:
            last_error = e
            continue

    if last_error is not None:
        raise RuntimeError(f"Failed to load a valid stereo RIR from {rir_folder}: {last_error}")
    raise RuntimeError(f"Failed to find a valid stereo RIR in {rir_folder}")



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
    """Create (reverberant, dry) stereo pair using pyroomacoustics RIRs.

    - dry is the clean input
    - reverberant is generated by convolving mid with two RIRs (two mic positions)
    - aligns RIRs to direct peak to avoid learning time shifts
    - returns (2, N)
    """
    if audio_ex.ndim != 2 or audio_ex.shape[1] != 2:
        raise ValueError("Input audio must be stereo with shape (samples, 2)")

    mid = 0.5 * (audio_ex[:, 0] + audio_ex[:, 1])

    mic_center = np.array([
        np.random.uniform(min_distance_to_wall, room_dim[n] - min_distance_to_wall)
        for n in range(3)
    ])
    source_pos = np.array([
        np.random.uniform(min_distance_to_wall, room_dim[n] - min_distance_to_wall)
        for n in range(3)
    ])

    left_mic = mic_center.copy(); left_mic[0] -= mic_spacing / 2.0
    right_mic = mic_center.copy(); right_mic[0] += mic_spacing / 2.0

    absorption, max_order = pra.inverse_sabine(t60, room_dim)
    rir_l = get_common_rir(left_mic.reshape(3, 1), source_pos, room_dim, fs, absorption, max_order, ray_tracing=True)
    rir_r = get_common_rir(right_mic.reshape(3, 1), source_pos, room_dim, fs, absorption, max_order, ray_tracing=True)

    rir_l = _align_rir_to_direct_peak(np.asarray(rir_l).squeeze())
    rir_r = _align_rir_to_direct_peak(np.asarray(rir_r).squeeze())

    rev_left = fftconvolve(mid, rir_l, mode="full")[: len(mid)]
    rev_right = fftconvolve(mid, rir_r, mode="full")[: len(mid)]
    reverberant_stereo = np.vstack([rev_left, rev_right])

    dry_stereo = np.swapaxes(audio_ex, 0, 1)

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

'''
def create_rir_conds_openair(
    fs,
    audio_ex,
    rir_folder,
    mix_range=(0.7, 1.0),
    early_ms: float = 80.0,
    mode: str = "room",
    wet_gain_range=(0.0, 1.0),
):
    """Apply a random *measured* stereo RIR to a stereo audio input.

    - default mode="room": y = x * h (physical capture model)
    - mix_range scales the *late tail* (after early_ms) to create different “reverb levels”
    - mode="send": y = x + g*(x*h). Use ONLY if IRs are wet-only (no direct-path).
    """
    rir_paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(rir_folder)
        for f in files
        if f.endswith(".wav")
    ]
    if not rir_paths:
        raise ValueError(f"No RIRs found in {rir_folder}")

    rir_path = random.choice(rir_paths)
    rir, rir_sr = sf.read(rir_path)
    if rir_sr != fs:
        raise ValueError(f"RIR sample rate {rir_sr} does not match expected {fs}")
    if rir.ndim != 2 or rir.shape[1] != 2:
        raise ValueError(f"Expected stereo RIR, got shape {rir.shape}")

    rir = _align_stereo_rir_preserve_itd(rir)

    k = float(np.random.uniform(*mix_range))
    split = int((early_ms / 1000.0) * fs)
    if 0 < split < rir.shape[0]:
        rir_mod = rir.copy()
        rir_mod[split:, :] *= k
    else:
        rir_mod = rir

    mid = 0.5 * (audio_ex[:, 0] + audio_ex[:, 1])

    wet_left = fftconvolve(mid, rir_mod[:, 0], mode="full")[: len(audio_ex)]
    wet_right = fftconvolve(mid, rir_mod[:, 1], mode="full")[: len(audio_ex)]
    wet = np.stack([wet_left, wet_right], axis=1)

    dry = audio_ex.copy()

    if mode == "send":
        g = float(np.random.uniform(*wet_gain_range))
        out = dry + g * wet
    else:
        out = wet

    return out.T, dry.T
'''

def create_rir_conds_openair(
    fs,
    audio_ex,
    rir_folder,
    mix_range=(0.2, 1.0),
    early_ms: float = 80.0,
    mode: str = "room",
    wet_gain_range=(0.0, 1.0),
    max_tries: int = 20,
    max_early_lr_diff_db: float = 4.0,
    remove_itd: bool = True,
    rir_norm_mode: str = "rms",
    rir_norm_target: float = 0.1,
    rir_norm_early_ms: float = 50.0,
):
    """
    - remove ITD to match synthetic path
    - reject extreme stereo imbalance
    - apply common stereo normalization
    - keep room mode as the default physical model
    """
    if audio_ex.ndim != 2 or audio_ex.shape[1] != 2:
        raise ValueError("Input audio must be stereo with shape (samples, 2)")

    rir = _load_random_valid_openair_rir(
        rir_folder=rir_folder,
        fs=fs,
        max_tries=max_tries,
        max_early_lr_diff_db=max_early_lr_diff_db,
        remove_itd=remove_itd,
        norm_mode=rir_norm_mode,
        norm_target=rir_norm_target,
        norm_early_ms=rir_norm_early_ms,
    )

    split = int((early_ms / 1000.0) * fs)
    k = float(np.random.uniform(*mix_range))

    if 0 < split < rir.shape[0]:
        early = rir[:split, :].copy()
        late = rir[split:, :].copy()
        late *= k
        rir_mod = np.concatenate([early, late], axis=0)
    else:
        rir_mod = rir

    # Match current synthetic path: collapse source to mono-mid before room rendering
    mid = 0.5 * (audio_ex[:, 0] + audio_ex[:, 1])

    wet_left = fftconvolve(mid, rir_mod[:, 0], mode="full")[: len(audio_ex)]
    wet_right = fftconvolve(mid, rir_mod[:, 1], mode="full")[: len(audio_ex)]
    wet = np.stack([wet_left, wet_right], axis=1).astype(np.float32)

    dry = audio_ex.astype(np.float32, copy=True)

    if mode == "send":
        g = float(np.random.uniform(*wet_gain_range))
        out = dry + g * wet
    else:
        # room mode: return the room capture only
        out = wet

    return out.T, dry.T


def normalize_source_once(
    x: np.ndarray,
    rate: int,
    target_lufs: float = -24.0,
    peak_limit: float = 0.99,
    eps: float = 1e-8,
):
    """
    Normalize the dry source BEFORE RIR rendering.
    This is the only loudness normalization step.
    """
    meter = pyln.Meter(rate)
    try:
        L = float(meter.integrated_loudness(x))
    except Exception:
        rms = float(np.sqrt(np.mean(x ** 2) + eps))
        L = 20.0 * np.log10(rms + eps)

    g = float(10.0 ** ((target_lufs - L) / 20.0))
    x = x * g

    peak = float(np.max(np.abs(x)))
    if peak > peak_limit and peak > 0:
        x = x * (peak_limit / peak)

    return x.astype(np.float32, copy=False)


def calibrate_wet_full_context_relative_to_dry(
    dry: np.ndarray,
    wet: np.ndarray,
    rate: int,
    target_rms_ratio_range=(0.80, 1.00),
    max_lufs_delta_db=0.5,
    max_gain_db=12.0,
    eps: float = 1e-8,
):
    """
    Scale ONLY the wet signal so that its full-context RMS relative to dry
    lies in a bounded range, then apply a loudness guardrail.

    This is better than active-mask RMS when long late tails make the
    reverberant excerpt sound globally louder.
    """
    dry_rms = float(np.sqrt(np.mean(dry ** 2) + eps))
    wet_rms = float(np.sqrt(np.mean(wet ** 2) + eps))

    if dry_rms < eps or wet_rms < eps:
        return wet.astype(np.float32, copy=False)

    # 1) Main calibration: full-context RMS ratio
    target_ratio = float(np.random.uniform(*target_rms_ratio_range))
    current_ratio = wet_rms / dry_rms

    g = target_ratio / max(current_ratio, eps)
    g_db = 20.0 * np.log10(max(g, eps))
    g_db = float(np.clip(g_db, -max_gain_db, max_gain_db))
    g = float(10.0 ** (g_db / 20.0))

    wet = wet * g

    # 2) Guardrail: prevent wet from ending up clearly louder than dry in LUFS
    try:
        meter = pyln.Meter(rate)
        L_dry = float(meter.integrated_loudness(dry))
        L_wet = float(meter.integrated_loudness(wet))

        delta = L_wet - L_dry
        if delta > max_lufs_delta_db:
            g2_db = max_lufs_delta_db - delta
            g2 = float(10.0 ** (g2_db / 20.0))
            wet = wet * g2
    except Exception:
        pass

    return wet.astype(np.float32, copy=False)


def apply_final_peak_safety(
    dry: np.ndarray,
    wet: np.ndarray,
    peak_limit: float = 0.99,
):
    """
    Final shared peak safety only.
    No loudness normalization here.
    """
    peak = float(max(np.max(np.abs(dry)), np.max(np.abs(wet))))
    if peak > peak_limit and peak > 0:
        s = peak_limit / peak
        dry = dry * s
        wet = wet * s

    return dry.astype(np.float32, copy=False), wet.astype(np.float32, copy=False)