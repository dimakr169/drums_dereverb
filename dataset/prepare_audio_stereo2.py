import os, argparse, random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

from audiomentations import Compose, PitchShift, TimeStretch, SevenBandParametricEQ
from config import Config

from preprocess_utils import (
    create_rir_conds_stereo,
    create_rir_generator_stereo,
    create_rir_conds_openair,
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
            if file.endswith(".wav"):
                drum_files.append(os.path.join(root, file))
    return drum_files

def process_item(file_path, pre_params, anechoic_path, reverb_path, rir_folder):
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
    duration_samples = int(pre_params.dur * pre_params.sr)
    num_chunks = len(audio_ex) // duration_samples
    audio_chunks = [
        audio_ex[i * duration_samples : (i + 1) * duration_samples]
        for i in range(num_chunks)
    ]

    for idx, chunk in enumerate(audio_chunks):
        if not detect_energy(chunk, threshold=pre_params.threshold):
            print(f"Energy exception for {audio_filename}, chunk {idx}")
            continue

        # Create the augmentation pipeline once per chunk
        augment = Compose(
            [
                TimeStretch(min_rate=0.9, max_rate=1.1, p=0.25),
                PitchShift(min_semitones=-1, max_semitones=1, p=0.25),
                SevenBandParametricEQ(min_gain_db=-6.0, max_gain_db=6.0, p=0.5),
            ]
        )
        for cnt in range(pre_params.aug_factor):
            print(f"Processing file: {audio_filename}, chunk {idx}, augmentation {cnt}")
            try:
                chunk_aug = augment(np.swapaxes(chunk, 0, 1), sample_rate=pre_params.sr)
                chunk_aug = set_loudness(np.swapaxes(chunk_aug, 0, 1), pre_params.sr, LUFS=pre_params.lufs)

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

                # Random Pick of generation method
                rir_method = random.choice(["real"]) #["pyroom", "rirgen", "real"]
                if rir_method == "pyroom":
                    lossy_ex, dry_ex = create_rir_conds_stereo(
                        t60,
                        room_dim,
                        pre_params.min_distance_to_wall,
                        pre_params.sr,
                        chunk_aug,
                    )
                elif rir_method == "rirgen":
                    lossy_ex, dry_ex = create_rir_generator_stereo(
                        t60,
                        room_dim,
                        pre_params.min_distance_to_wall,
                        pre_params.sr,
                        chunk_aug,
                    )
                else: # real
                    lossy_ex, dry_ex = create_rir_conds_openair(
                        pre_params.sr, 
                        chunk_aug, 
                        rir_folder=rir_folder, 
                        mix_range=(0.0, 1.0)
                    )

                lossy_ex = set_loudness(np.swapaxes(lossy_ex, 0, 1), pre_params.sr, LUFS=pre_params.lufs)
                dry_ex = set_loudness(np.swapaxes(dry_ex, 0, 1), pre_params.sr, LUFS=pre_params.lufs)

                lossy_ex = trim_audio(lossy_ex, pre_params.sr, pre_params.dur)
                dry_ex = trim_audio(dry_ex, pre_params.sr, pre_params.dur)

                if (lossy_ex is None or lossy_ex.size == 0 or np.allclose(lossy_ex, -1)) or \
                   (dry_ex is None or dry_ex.size == 0 or np.allclose(dry_ex, -1)):
                    print(f"Invalid audio for {audio_filename}, chunk {idx}, augmentation {cnt}")
                    continue

                if detect_energy(lossy_ex, threshold=pre_params.threshold) and \
                   detect_energy(dry_ex, threshold=pre_params.threshold):
                    out_filename = f"{audio_filename}_chunk{idx}_aug{cnt}.{pre_params.inp_type}"
                    sf.write(os.path.join(anechoic_path, out_filename), dry_ex, pre_params.sr)
                    sf.write(os.path.join(reverb_path, out_filename), lossy_ex, pre_params.sr)
                else:
                    print(f"Skipping file {audio_filename}_chunk{idx}_aug{cnt}.{pre_params.inp_type} due to empty audio data")
            except Exception as e:
                print(f"Aborted processing {audio_filename}, chunk {idx}, augmentation {cnt} due to: {e}")

def process_batch(file_paths, pre_params, anechoic_path, reverb_path, rir_folder):
    for file_path in file_paths:
        process_item(file_path, pre_params, anechoic_path, reverb_path, rir_folder)

def main(args):
    pre_params = Config()
    current_dir = Path.cwd()
    data_dir = current_dir.parent / "data/MoisesDB_test_clean"   #"data/gmd_musdb18hq_stereo" 
    rir_folder = current_dir.parent / "data/ReverbFX_ACE_RIRs_test"  #"data/OpenAir_RIRs_stereo" 
    drum_files = load_drum_files(data_dir)
    print(f"Found {len(drum_files)} drum files.")

    anechoic_path = os.path.join(args.out_path, "anechoic")
    os.makedirs(anechoic_path, exist_ok=True)
    reverb_path = os.path.join(args.out_path, "reverb")
    os.makedirs(reverb_path, exist_ok=True)

    # Define a batch size (adjust as needed)
    batch_size = 5
    batches = [drum_files[i:i + batch_size] for i in range(0, len(drum_files), batch_size)]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(process_batch, batch, pre_params, anechoic_path, reverb_path, rir_folder)
            for batch in batches
        ]
        for future in as_completed(futures):
            future.result()

    print("Completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-path", default=str(Path.cwd().parent / "data/moisesdb_test_rir-only"), type=str)
    args = parser.parse_args()
    main(args)
