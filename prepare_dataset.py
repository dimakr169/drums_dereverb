import os
import argparse
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from audiomentations import Compose, PitchShift, TimeStretch, SevenBandParametricEQ

from dataset.data_config import DataConfig
from dataset.preprocess_utils import (
    create_rir_conds_stereo,
    create_rir_conds_openair,
    detect_energy,
    normalize_source_once,
    calibrate_wet_full_context_relative_to_dry,
    apply_final_peak_safety,
    trim_audio,
)

def load_drum_files(data_dir):
    """
    Recursively find all drum files in the given directory.
    """
    drum_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                drum_files.append(os.path.join(root, file))
    return drum_files

def process_item(file_path, pre_params, anechoic_path, reverb_path, rir_pools):
    # Load the audio file
    try:
        audio_ex, sr = sf.read(file_path)
        # Check if the audio is stereo
        if len(audio_ex.shape) != 2:
            raise ValueError("Audio file is not stereo.")

        audio_ex = audio_ex.astype(np.float32)
        if sr != pre_params.sr:
            raise ValueError(f"Sample rate mismatch. Expected {pre_params.sr}, got {sr}.")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    folder_name = os.path.basename(os.path.dirname(file_path))
    file_base = os.path.splitext(os.path.basename(file_path))[0]
    audio_filename = f"{folder_name}__{file_base}"

    # --- Context-first processing to preserve cross-boundary reverb tails ---
    # Slice into longer windows, apply augmentation + reverb, then slice into 2s segments.
    context_dur = float(pre_params.context_dur) # 10 seconds
    segment_dur = float(pre_params.dur)  # usually 2.0s
    context_samples = int(context_dur * pre_params.sr)
    segment_samples = int(segment_dur * pre_params.sr)

    num_context = len(audio_ex) // context_samples
    context_chunks = [
        audio_ex[i * context_samples : (i + 1) * context_samples]
        for i in range(num_context)
    ]

    for ctx_idx, chunk in enumerate(context_chunks):
        if not detect_energy(chunk, threshold=pre_params.threshold):
            print(f"Energy exception for {audio_filename}, context {ctx_idx}")
            continue

        # Create the augmentation pipeline once per context
        augment = Compose(
            [
                TimeStretch(min_rate=0.9, max_rate=1.1, p=0.33),
                PitchShift(min_semitones=-1, max_semitones=1, p=0.33),
                SevenBandParametricEQ(min_gain_db=-6.0, max_gain_db=6.0, p=0.5),
            ]
        )

        for cnt in range(pre_params.aug_factor):
            print(f"Processing file: {audio_filename}, context {ctx_idx}, augmentation {cnt}")
            try:
                # Audiomentations expects (channels, samples) for multichannel.
                chunk_cf = np.ascontiguousarray(np.swapaxes(chunk, 0, 1))
                chunk_aug_cf = augment(chunk_cf, sample_rate=pre_params.sr)
                dry_ctx = np.swapaxes(chunk_aug_cf, 0, 1).astype(np.float32)  # (N,2)

                # Ensure fixed context length after augmentation (time-stretch can change length)
                dry_ctx = trim_audio(dry_ctx, pre_params.sr, context_dur)

                # Normalize source before any RIR rendering
                dry_ctx = normalize_source_once(
                    dry_ctx,
                    pre_params.sr,
                    target_lufs=pre_params.lufs,
                    peak_limit=0.99,
                )

                t60 = np.random.uniform(pre_params.t60_r[0], pre_params.t60_r[1])
                room_dim = np.array(
                    [
                        np.random.uniform(
                            pre_params.room_dim_r[2 * n],
                            pre_params.room_dim_r[2 * n + 1],
                        )
                        for n in range(3)
                    ]
                )

                # Only real measured RIRs OR synthetic pyroomacoustics
                rir_method = random.choice(["real", "pyroom"])
                if rir_method == "pyroom":
                    lossy_cf, dry_cf = create_rir_conds_stereo(
                        t60,
                        room_dim,
                        pre_params.min_distance_to_wall,
                        pre_params.sr,
                        dry_ctx,
                    )
                else:
                    # select random RIR
                    pool = random.choice(rir_pools)
                    
                    lossy_cf, dry_cf = create_rir_conds_openair(
                        pre_params.sr,
                        dry_ctx,                      # (N,2)
                        rir_folder=pool["folder"],
                        mode=pool["mode"],            # <-- only change you need
                        mix_range=(0.2, 1.0),          # in room mode: late-tail multiplier
                        wet_gain_range=(0.1, 1.0),     # in send mode: α range
                        early_ms=80.0,
                        max_tries=20,
                        max_early_lr_diff_db=4.0,
                        remove_itd=True,
                        rir_norm_mode="rms",
                        rir_norm_target=0.1,
                        rir_norm_early_ms=50.0,
                    )

                # Convert to (samples,2)
                lossy_ctx = np.swapaxes(lossy_cf, 0, 1)
                dry_ctx_pair = np.swapaxes(dry_cf, 0, 1)

                # Calibrate the wet signal using active-sample RMS relative to dry

                lossy_ctx = calibrate_wet_full_context_relative_to_dry(
                    dry_ctx_pair,
                    lossy_ctx,
                    pre_params.sr,
                    target_rms_ratio_range=(0.80, 1.00),
                    max_lufs_delta_db=0.0,
                    max_gain_db=12.0,
                )

                # Final shared peak safety only
                dry_ctx_pair, lossy_ctx = apply_final_peak_safety(
                    dry_ctx_pair,
                    lossy_ctx,
                    peak_limit=0.99,
                )

                # Ensure fixed context length
                dry_ctx_pair = trim_audio(dry_ctx_pair, pre_params.sr, context_dur)
                lossy_ctx = trim_audio(lossy_ctx, pre_params.sr, context_dur)

                if (lossy_ctx is None or lossy_ctx.size == 0) or (dry_ctx_pair is None or dry_ctx_pair.size == 0):
                    print(f"Invalid audio for {audio_filename}, context {ctx_idx}, augmentation {cnt}")
                    continue

                # Slice into 2s segments AFTER reverb so tails cross boundaries
                total_len = dry_ctx_pair.shape[0]
                if total_len < segment_samples:
                    continue

                # number of full 2s segments we can extract from this context
                n_segs = total_len // segment_samples

                # allow a random offset ONLY within the "slack" that still keeps n_segs segments
                # (for context_dur=10s and segment_dur=2s, slack=0 -> offset=0 -> you get 5 segments)
                slack = total_len - n_segs * segment_samples  # 0..segment_samples-1
                offset = random.randint(0, slack) if slack > 0 else 0
                
                for seg_id in range(n_segs):
                    start = offset + seg_id * segment_samples
                    end = start + segment_samples

                    dry_seg = dry_ctx_pair[start:end]
                    lossy_seg = lossy_ctx[start:end]

                    if detect_energy(lossy_seg, threshold=pre_params.threshold) and detect_energy(
                        dry_seg, threshold=pre_params.threshold
                    ):
                        out_filename = f"{audio_filename}_ctx{ctx_idx}_aug{cnt}_seg{seg_id}.{pre_params.inp_type}"
                        sf.write(os.path.join(anechoic_path, out_filename), dry_seg, pre_params.sr)
                        sf.write(os.path.join(reverb_path, out_filename), lossy_seg, pre_params.sr)

            except Exception as e:
                print(f"Aborted processing {audio_filename}, context {ctx_idx}, augmentation {cnt} due to: {e}")

def process_batch(file_paths, pre_params, anechoic_path, reverb_path, rir_pools):
    for file_path in file_paths:
        process_item(file_path, pre_params, anechoic_path, reverb_path, rir_pools)

def main(args):
    pre_params = DataConfig()

    data_dir = Path(args.dataset_path)
    rir_pools = [{"folder": Path(p), "mode": args.rir_mode} for p in args.rir_path]

    drum_files = load_drum_files(data_dir)
    print(f"Found {len(drum_files)} drum files.")

    anechoic_path = os.path.join(args.out_path, "anechoic")
    reverb_path = os.path.join(args.out_path, "reverb")
    os.makedirs(anechoic_path, exist_ok=True)
    os.makedirs(reverb_path, exist_ok=True)

    batches = [
        drum_files[i:i + args.batch_size]
        for i in range(0, len(drum_files), args.batch_size)
    ]

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                process_batch,
                batch,
                pre_params,
                anechoic_path,
                reverb_path,
                rir_pools,
            )
            for batch in batches
        ]

        for future in as_completed(futures):
            future.result()

    print("Completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the input stereo drum dataset root."
    )
    parser.add_argument(
        "--rir-path",
        type=str,
        action="append",
        required=True,
        help="Path to a stereo RIR folder. Can be passed multiple times."
    )
    parser.add_argument(
        "--out-path",
        type=str,
        required=True,
        help="Output folder where anechoic/ and reverb/ will be created."
    )
    parser.add_argument(
        "--rir-mode",
        type=str,
        default="room", # send type is for future use
        choices=["room", "send"],
        help="How to interpret the provided RIR folders."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="How many source files to process per worker submission."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes."
    )

    args = parser.parse_args()
    main(args)
