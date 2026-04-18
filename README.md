# A Cold Diffusion Approach for Percussive Dereverberation

Official codebase for **stereo drums dereverberation** using **cold diffusion** models operating on **real–imaginary (RI) stereo STFT representations**. 

This repository includes:
- **Dataset preparation** for paired dry / reverberant stereo drum mixtures
- **Training** script for cold diffusion **UNet** and **DiT** backbones through YAML configuration files
- **Inference** script using **pre-trained** provided checkpoints.

---


## 1. Installation

We recommend using **Conda** for the base environment and **pip** for PyTorch (2.9) and Python (3.11) dependencies.

```bash
conda env create -f environment.yml
conda activate drums_dereverb
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

In case you want to use it only for inference you may install PyTorch for CPU as well.
- `ffmpeg` is recommended for  MP3 decoding (for inference)
- `libsndfile` is required by `soundfile`


---

## 2. Dataset preparation

To generate your own dataset use `prepare_dataset.py` for creating paired **anechoic** and **reverberant** stereo drum files. The script expects a root folder containing clean stereo drum audio files (`.wav`) and one or more folders containing stereo RIRs. We provide the same selection from OpenAIR library (`dataset/OpenAir_stereo_RIRs`) that we used on our initial work. If you use them, please cite them as well.


#### Example usage

```bash
python prepare_dataset.py \
  --dataset-path /path/to/clean_stereo_drums \
  --rir-path dataset/OpenAir_stereo_RIRs \
  --out-path /path/to/dataset_stereo
```

#### Output structure

```text
dataset_stereo/
├── anechoic/
└── reverb/
```

Dataset creation uses the preprocessing settings stored in `dataset/data_config.py`. The script may use either:
- **synthetic RIRs** generated on the fly, or
- **real measured RIRs** from the provided RIR pools

---

## 3. Training

Training is controlled through YAML configuration files stored in `configs/`.

We provide the following configs to reproduce our models from the paper: 
- `configs/unet_direct.yaml`
- `configs/unet_delta.yaml`
- `configs/dit_delta.yaml`

Example usage:

```bash
python train_cold_ri.py --config configs/unet_delta.yaml 
```

In case resuming the training you can use the ```--resume``` argument.

Training outputs are written under the configured  directory, typically containing:

```text
saved_models/<experiment_name>/
├── checkpoints/
├── logs/
├── samples/
└── resolved_config.yaml
```

---

## 4. Inference

You can dereverb your own files using the `inference_cold_ri.py` script which: 
- supports `.wav` and `.mp3`.
- ensures stereo format
- resamples to **44.1 kHz** if needed
- trims or pads the signal to a requested duration (2-30 seconds)
- segments audio into **2-second windows** (following the paper) with **50% overlap**
- reconstructs the final waveform after segment-wise dereverberation


The script will use the model definition from the provided YAML config. We provide pre-trained checkpoints for UNet (`delta`, `direct`) and DiT (`delta`) backbones. You can download them from here (link will be added soon).

#### Example Usage
**UNet delta**

```bash
python inference_cold_ri.py \
  --config configs/delta_unet.yaml \
  --checkpoint /path/to/checkpoints/best.pt \
  --input-path /path/to/input_folder_or_file \
  --output-path /path/to/output_folder \
  --gpu 0 \
  --batch-size 8 \
  --trim-seconds 20
```

- `--trim-seconds N` will only use the first N seconds from the input files.
- `--batch-size` controls how many 2-second segments are processed together
- `--gpu -1` will force CPU inference

---


## 5. Examples
Coming soon

## 6. Reference

We kindly ask you to cite the paper in your publication when using any of our research or code:

```bibtex
@inproceedings{TBA,
  title     = {A Cold Diffusion Approach for Percussive Dereverberation},
  author    = {Makris, Dimos and Barjak, Andras and ́Kaliakatsos-Papakostas, Maximos},
  booktitle = {Proceedings of the IEEE World Congress on Computational Intelligence (WCCI)},
  year      = {2026}
}
```

