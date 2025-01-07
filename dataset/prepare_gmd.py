import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile as sf

from audiomentations import Compose, PitchShift, TimeStretch
from config import Config

from iir import get_random_eq_values
from preprocess_utils import (
    create_rir_conds,
    detect_energy,
    set_loudness,
    trim_audio,
)


def load_drum_files(data_dir):
    """
    Recursively find all drum files in the new directory structure.
    Each file path is paired with its naming prefix.
    """
    drum_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                # Extract drummer and session information from the folder structure
                parts = os.path.normpath(root).split(os.sep)
                if len(parts) >= 2:  # Ensure we have at least drummerX/sessionY
                    drummer = parts[-2]
                    session = parts[-1]
                    naming_prefix = f"{drummer}_{session}"
                    drum_files.append((os.path.join(root, file), naming_prefix))
    return drum_files


def process_item(file_path, naming_prefix, pre_params, anechoic_path, reverb_path):
    # Load the audio file
    try:
        audio_ex, sr = sf.read(file_path)
        if len(audio_ex.shape) > 1:  # If the audio has more than one channel
            audio_ex = np.mean(audio_ex, axis=1)  # Average across channels to make mono

        audio_ex = audio_ex.astype(np.float32)
        if sr != pre_params.sr:
            raise ValueError(f"Sample rate mismatch. Expected {pre_params.sr}, got {sr}.")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    # Extract the base filename without the extension
    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    # Slice the audio into chunks of pre-defined duration
    duration_samples = int(pre_params.dur * pre_params.sr)
    num_chunks = len(audio_ex) // duration_samples
    audio_chunks = [
        audio_ex[i * duration_samples : (i + 1) * duration_samples]
        for i in range(num_chunks)
    ]

    # Process each chunk
    for idx, chunk in enumerate(audio_chunks):
        is_ok = detect_energy(chunk, threshold=pre_params.threshold)
        if is_ok:
            try:
                for cnt in range(pre_params.aug_factor):  # augmentation factor
                    print(f"Processing file: {naming_prefix}_{base_filename}, chunk {idx}, augmentation {cnt}")
                    
                    # Augment and process the chunk
                    augment = Compose(
                        [
                            TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5),
                            PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
                        ]
                    )
                    chunk_aug = augment(chunk, sample_rate=pre_params.sr)
                    chunk_aug = set_loudness(chunk_aug, pre_params.sr, LUFS=pre_params.lufs)
                    peq, _ = get_random_eq_values(70, pre_params.sr)
                    chunk_aug = peq.apply_eq(chunk_aug)
                    
                    # Reverberation
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
                    lossy_ex, dry_ex = create_rir_conds(
                        t60,
                        room_dim,
                        pre_params.min_distance_to_wall,
                        pre_params.sr,
                        chunk_aug,
                    )
                    
                    # Normalize and trim
                    lossy_ex = set_loudness(lossy_ex, pre_params.sr, LUFS=pre_params.lufs)
                    dry_ex = set_loudness(dry_ex, pre_params.sr, LUFS=pre_params.lufs)
                    lossy_ex = trim_audio(lossy_ex, pre_params.sr, pre_params.dur)
                    dry_ex = trim_audio(dry_ex, pre_params.sr, pre_params.dur)

                    if len(lossy_ex) > 0 and len(dry_ex) > 0:
                        try:
                            # Write to files with naming convention
                            sf.write(
                                os.path.join(
                                    anechoic_path,
                                    f"{naming_prefix}_{base_filename}_chunk{idx}_aug{cnt}.{pre_params.inp_type}",
                                ),
                                dry_ex,
                                pre_params.sr,
                            )
                            sf.write(
                                os.path.join(
                                    reverb_path,
                                    f"{naming_prefix}_{base_filename}_chunk{idx}_aug{cnt}.{pre_params.inp_type}",
                                ),
                                lossy_ex,
                                pre_params.sr,
                            )
                        except Exception as e:
                            print(
                                f"Error writing file {naming_prefix}_{base_filename}_chunk{idx}_aug{cnt}.{pre_params.inp_type}: {e}"
                            )
                    else:
                        print(
                            f"Skipping file {naming_prefix}_{base_filename}_chunk{idx}_aug{cnt}.{pre_params.inp_type} due to empty audio data"
                        )
            except Exception as e:
                print(f"Aborted due to: {e}")
        else:
            print(f"Energy exception for {naming_prefix}_{base_filename}, chunk {idx}")


def main(args):
    pre_params = Config()
    data_dir = "./data/gmd"  
    drum_files = load_drum_files(data_dir)
    print(f"Found {len(drum_files)} drum files.")

    anechoic_path = os.path.join(args.out_path, "anechoic")
    os.makedirs(anechoic_path, exist_ok=True)
    reverb_path = os.path.join(args.out_path, "reverb")
    os.makedirs(reverb_path, exist_ok=True)

    futures = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()*4) as executor:
        for file_path, naming_prefix in drum_files:
            futures.append(
                executor.submit(
                    process_item, file_path, naming_prefix, pre_params, anechoic_path, reverb_path
                )
            )

        for future in as_completed(futures):
            future.result()

    print("Completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-path", default="./data/out_gmd", type=str)

    args = parser.parse_args()
    main(args)
