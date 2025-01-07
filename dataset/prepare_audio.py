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
    Recursively find all drum files in the given directory.
    """
    drum_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav") and "drums" in file:
                drum_files.append(os.path.join(root, file))
    return drum_files


def process_item(file_path, pre_params, anechoic_path, reverb_path):
    # Load the audio file
    try:
        audio_ex, sr = sf.read(file_path)
        # Convert to mono if it's multi-channel
        if len(audio_ex.shape) > 1:  # If the audio has more than one channel
            audio_ex = np.mean(audio_ex, axis=1)  # Average across channels to make mono

        # Convert audio to np.float32
        audio_ex = audio_ex.astype(np.float32)
        
        if sr != pre_params.sr:
            raise ValueError(f"Sample rate mismatch. Expected {pre_params.sr}, got {sr}.")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    # Extract the name of the song from the path
    audio_filename = os.path.basename(os.path.dirname(file_path))

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
                    print(f"Processing file: {audio_filename}, chunk {idx}, augmentation {cnt}")
                    
                    # Augment with audiomentations
                    augment = Compose(
                        [
                            TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5),
                            PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
                        ]
                    )
                    chunk_aug = augment(chunk, sample_rate=pre_params.sr)
                    chunk_aug = set_loudness(chunk_aug, pre_params.sr, LUFS=pre_params.lufs)
                    
                    # Apply random 6-band PEQ
                    peq, _ = get_random_eq_values(70, pre_params.sr)
                    chunk_aug = peq.apply_eq(chunk_aug)
                    
                    # Sample T60 and room dimensions
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
                    
                    # Create clean and reverberant files
                    lossy_ex, dry_ex = create_rir_conds(
                        t60,
                        room_dim,
                        pre_params.min_distance_to_wall,
                        pre_params.sr,
                        chunk_aug,
                    )
                    
                    # Normalize
                    lossy_ex = set_loudness(lossy_ex, pre_params.sr, LUFS=pre_params.lufs)
                    dry_ex = set_loudness(dry_ex, pre_params.sr, LUFS=pre_params.lufs)
                    
                    # Trim to fixed duration
                    lossy_ex = trim_audio(lossy_ex, pre_params.sr, pre_params.dur)
                    dry_ex = trim_audio(dry_ex, pre_params.sr, pre_params.dur)
                    
                    # Check if the audio is not empty before writing to file
                    if len(lossy_ex) > 0 and len(dry_ex) > 0:
                        try:
                            # Write to files
                            sf.write(
                                os.path.join(
                                    anechoic_path,
                                    f"{audio_filename}_chunk{idx}_aug{cnt}.{pre_params.inp_type}",
                                ),
                                dry_ex,
                                pre_params.sr,
                            )
                            sf.write(
                                os.path.join(
                                    reverb_path,
                                    f"{audio_filename}_chunk{idx}_aug{cnt}.{pre_params.inp_type}",
                                ),
                                lossy_ex,
                                pre_params.sr,
                            )
                        except Exception as e:
                            print(
                                f"Error writing file {audio_filename}_chunk{idx}_aug{cnt}.{pre_params.inp_type}: {e}"
                            )
                    else:
                        print(
                            f"Skipping file {audio_filename}_chunk{idx}_aug{cnt}.{pre_params.inp_type} due to empty audio data"
                        )
            except Exception as e:
                print(f"Aborted due to: {e}")
        else:
            print(f"Energy exception for {audio_filename}, chunk {idx}")


def main(args):
    ## Load configuration
    pre_params = Config()

    # Find all drum files in the specified directory
    data_dir = "./data/musdb18hq_clean"  
    drum_files = load_drum_files(data_dir)
    print(f"Found {len(drum_files)} drum files.")

    # Create paths for anechoic and reverberant files to be stored
    anechoic_path = os.path.join(args.out_path, "anechoic")
    os.makedirs(anechoic_path, exist_ok=True)
    reverb_path = os.path.join(args.out_path, "reverb")
    os.makedirs(reverb_path, exist_ok=True)

    # Process dataset with multithreading
    futures = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for item in drum_files:
            futures.append(
                executor.submit(
                    process_item, item, pre_params, anechoic_path, reverb_path
                )
            )

        for future in as_completed(futures):
            future.result()

    print("Completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-path", default="./data/out_combined", type=str)

    args = parser.parse_args()
    main(args)
