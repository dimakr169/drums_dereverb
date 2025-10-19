# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:25:14 2024

@author: dimak
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile as sf
import tensorflow_datasets as tfds
import yaml
from audiomentations import Compose, PitchShift, TimeStretch

from iir import get_random_eq_values
from preprocess_utils import (
    create_rir_conds,
    detect_energy,
    set_loudness,
    trim_audio,
)


def load_params(params_file, environment):
    with open(params_file, "r") as file:
        params = yaml.safe_load(file)
    return params[environment]


def process_item(item, pre_params, anechoic_path, reverb_path):
    audio_ex = item["audio"][:, 0].numpy()
    # get name
    audio_filename = item["filename"].numpy().decode("utf-8").split("/")[-1][:-4]
    is_ok = detect_energy(audio_ex, threshold=pre_params["threshold"])
    if is_ok:
        try:
            for cnt in range(pre_params["aug_factor"]):  # augmentation factor
                print("Processing file:", audio_filename)
                # augment with audiomentations
                augment = Compose(
                    [
                        TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
                        PitchShift(min_semitones=-3, max_semitones=3, p=0.6),
                    ]
                )
                audio_ex_aug = augment(audio_ex, sample_rate=pre_params["sr"])
                audio_ex_aug = set_loudness(
                    audio_ex_aug, pre_params["sr"], LUFS=pre_params["lufs"]
                )
                # apply random 6 band PEQ
                peq, _ = get_random_eq_values(
                    70, pre_params["sr"]
                )  # prob=70% for each filter to activate
                audio_ex_aug = peq.apply_eq(audio_ex_aug)
                # sample t60
                t60 = np.random.uniform(pre_params["t60_r"][0], pre_params["t60_r"][1])
                # sample room dimensions
                room_dim = np.array(
                    [
                        np.random.uniform(
                            pre_params["room_dim_r"][2 * n],
                            pre_params["room_dim_r"][2 * n + 1],
                        )
                        for n in range(3)
                    ]
                )
                # create clean and reverberant files
                lossy_ex, dry_ex = create_rir_conds(
                    t60,
                    room_dim,
                    pre_params["min_distance_to_wall"],
                    pre_params["sr"],
                    audio_ex_aug,
                )
                # normalize
                lossy_ex = set_loudness(
                    lossy_ex, pre_params["sr"], LUFS=pre_params["lufs"]
                )
                dry_ex = set_loudness(dry_ex, pre_params["sr"], LUFS=pre_params["lufs"])
                # trim to fixed duration
                lossy_ex = trim_audio(
                    lossy_ex, pre_params["sr"], pre_params["dur"]
                )  # 2 seconds
                dry_ex = trim_audio(dry_ex, pre_params["sr"], pre_params["dur"])

                # Check if the audio is not empty before writing to file
                if len(lossy_ex) > 0 and len(dry_ex) > 0:
                    try:
                        # write to files
                        sf.write(
                            os.path.join(
                                anechoic_path,
                                audio_filename
                                + "_"
                                + str(cnt)
                                + "."
                                + pre_params["inp_type"],
                            ),
                            dry_ex,
                            pre_params["sr"],
                        )
                        sf.write(
                            os.path.join(
                                reverb_path,
                                audio_filename
                                + "_"
                                + str(cnt)
                                + "."
                                + pre_params["inp_type"],
                            ),
                            lossy_ex,
                            pre_params["sr"],
                        )
                    except Exception as e:
                        print(
                            f"Error writing file {audio_filename + '_' + str(cnt) + '.' + pre_params['inp_type']}: {e}"
                        )
                else:
                    print(
                        f"Skipping file {audio_filename + '_' + str(cnt) + '.' + pre_params['inp_type']} due to empty audio data"
                    )
            cnt += 1
        except Exception as e:
            print("Aborted to ", e)
    else:
        print("Energy exception")


def main(args):
    # Set GOOGLE_APPLICATION_CREDENTIALS environment variable in case it is needed
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.credentials_path

    # Load environment from env file
    with open("env", "r") as env_file:
        environment = env_file.read().strip()

    # Load parameters for the current environment
    params = load_params("params.yaml", environment)
    pre_params = params["preprocessing"]

    # Load dataset
    dataset, info = tfds.load(
        "addictive_drums", data_dir="gs://xln-datasets", with_info=True, split="train"
    )

    # List all channels
    print(info.features["channel"].names)
    # Get OH and CL (close mic) channels only
    dataset = dataset.filter(lambda x: x["channel"] == 0 or x["channel"] == 2)

    # Optionally reduce the dataset size for development
    if pre_params.get("reduced_data", False):
        dataset = dataset.take(100)

    # Create paths for anechoic and reverberant files to be stored
    anechoic_path = os.path.join(args.out_path, "anechoic")
    os.makedirs(anechoic_path, exist_ok=True)
    reverb_path = os.path.join(args.out_path, "reverb")
    os.makedirs(reverb_path, exist_ok=True)

    # Process dataset with multithreading
    futures = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for item in dataset:
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
    parser.add_argument("--out-path", default="./data", type=str)
    parser.add_argument(
        "--credentials-path",
        default="./cloud_credentials.json",
        type=str,
        help="Path to the Google Cloud credentials JSON file",
    )
    args = parser.parse_args()
    main(args)
