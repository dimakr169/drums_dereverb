import glob
import os
import json
import torch
import gradio as gr
import numpy as np
import soundfile as sf
from scipy import signal
from dataset.config import Config
from utils_inference import build_model_from_entry, load_ckpt_and_ema,\
        ColdDiffInferencer_var, ensure_float_audio, ensure_stereo, resample_audio,\
        trim_or_pad_range, set_loudness, segment_audio_torch, center_stitch_from_segments,\
        ola_reconstruct_torch, audio_to_stereo_ri_stft, generate_spectrogram

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


# Global Params
gpu_id = 2 # GPU for inference
models_config = 'models_gradio.json' #Load models for inference
save_path = './eval_out/gradio_out' #save diffused segs
ts_min = 2 # min num of seconds for input (model duration)
ts_max = 30  # max num of seconds for input
fs = 44100  # default sample rate
LUFS = -28.0  # default loudness

# Load dataset config
pre_params = Config()

# choose device
if torch.cuda.is_available():
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)  
    print("Using GPU:", device)
else:
    device = "cpu"; print("No GPU, using CPU")


# Set examples folder
example_files = [[file] for file in glob.glob("gradio_examples/*.mp3")]

# Add description
description_html = """
<span style="font-size: 24px; font-weight: bold;">Drums Dereverberation 0.7</span>
<br>
Upload your own input or use the preselected examples (in the bottom of the page). The uploaded input will be resampled to 44.1 KHz, ensuring to be stereo,
and normalized to -28 LUFS. Duration has to be from 2 up to 30 seconds, otherwise it will be trimmed. All models have been trained with initial diffusion schedule 
of 16 steps. However, you can change it from range 4 up to 64.
"""

# Model Names load from JSON ----
with open(models_config, "r") as f:
    model_zoo = json.load(f)

model_names = list(model_zoo.keys())
print("Models to evaluate:", model_names)

# Load Models (Inferencers)
models = {}
# ---- loop over models ----
for name in model_names:
    if name not in model_zoo:
        print(f"[WARN] Model '{name}' not found in JSON, skipping.")
        continue

    entry = model_zoo[name]
    # build model + load checkpoint
    model = build_model_from_entry(entry)
    model, ema  = load_ckpt_and_ema(model, entry, device)

    # Load info from JSON
    scheme = entry["scheme"]  # "cold" (default) or "cdiffuse"
    model_type = entry["type"]
    diffusion_steps = entry["diffusion_steps"]
    alpha_mode = entry["alpha_mode"]
    cdiff_mode = entry["mode"]
    reverse_steps = entry["reverse_steps"]
    solver = entry["solver"]

    inferencer = ColdDiffInferencer_var(
        model=model,
        model_type=model_type,
        pre_params=pre_params,
        diffusion_steps=diffusion_steps,
        reverse_steps = reverse_steps, #sampling steps
        solver = solver, #euler or heun
        alpha_mode=alpha_mode,
        cdiff_mode=cdiff_mode,
        device=device,
    )

    # Store Inference on dict
    models[name] = inferencer
    print("Model", name, "loaded succesfully")


# Dropdown for model selection
model_selector = gr.Dropdown(
    choices=model_names, label="Select Model", value=model_names[0]
)

steps_slider = gr.Slider(
    minimum=4,
    maximum=64,
    value=16,          # default
    step=1,
    label="Reverse Diffusion Steps",
)

overlap_slider = gr.Slider(
    minimum=0.25,
    maximum=0.75,
    value=0.5,          # default
    step=0.25,
    label="Segmentation Overlap Ratio",
)



def peak_ceiling(
    wav: torch.Tensor,
    ceiling: float = 0.98,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Light peak limiter via global gain per segment (no hard clipping).
    wav: [B,2,T]
    """
    peak = wav.abs().amax(dim=(1, 2), keepdim=True).clamp_min(eps)  # [B,1,1]
    scale = torch.minimum(torch.ones_like(peak), torch.tensor(ceiling, device=wav.device).view(1,1,1) / peak)
    return wav * scale


def rms_db_mono(wav: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    wav: [B, 2, T] or [B, 1, T]
    returns: [B] RMS level in dBFS (relative, assuming wav in [-1,1])
    """
    if wav.ndim != 3:
        raise ValueError(f"Expected [B,C,T], got {wav.shape}")
    mono = wav.mean(dim=1)  # [B,T]
    rms = mono.pow(2).mean(dim=-1).clamp_min(eps).sqrt()  # [B]
    return 20.0 * torch.log10(rms)

def match_segment_rms_db_threshold(
    est: torch.Tensor,
    ref: torch.Tensor,
    *,
    threshold_db: float = 2.5,
    max_gain_db: float = 6.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Apply RMS-dB loudness matching ONLY when |ref_db - est_db| > threshold_db.

    est, ref: [B,2,T]
    """
    est_db = rms_db_mono(est, eps=eps)   # [B]
    ref_db = rms_db_mono(ref, eps=eps)   # [B]
    delta_db = ref_db - est_db           # [B]

    # Only correct if outside threshold
    apply = delta_db.abs() > threshold_db

    # Cap the correction
    corr_db = delta_db.clamp(-max_gain_db, max_gain_db)

    # If not applying, set correction to 0 dB
    corr_db = torch.where(apply, corr_db, torch.zeros_like(corr_db))

    gain = torch.pow(10.0, corr_db / 20.0).view(-1, 1, 1)  # [B,1,1]
    return est * gain

# Function for processing
def process_audio(audio, model_name, reverse_steps, overlap_ratio):

    # 0. Load audio from user
    sr, data_inp = audio
    # 1. Ensure audio is in floating-point format and Stereo
    data_inp = ensure_float_audio(data_inp)
    data_inp = ensure_stereo(data_inp)          # [T,2]
    # 2. Resample if needed
    if sr != fs:
        data_inp = resample_audio(data_inp, sr, fs)
        sr = fs
    # 3. Trim file to 2 to 30 seconds and convert it to mono
    print(data_inp.shape)
    data_inp = trim_or_pad_range(data_inp, fs, min_s=ts_min, max_s=ts_max)
    print('orig_len', data_inp.shape)
    # 4. Set Loudness to desired LUFS
    data_inp = set_loudness(data_inp, fs, LUFS=LUFS)
    # 5. Segment audio and compute STFT
    segs, step, orig_len = segment_audio_torch(
        data_inp, fs, ts_min=ts_min, overlap=overlap_ratio, pad_end=True, device=device
    )
    print("segments:", tuple(segs.shape), "step:", step, "orig_len:", orig_len)
    # segs: [N, 2, L]
    # STFTs
    ri_in =  audio_to_stereo_ri_stft(segs, config=pre_params, device=device)  # [N,4,F,TT]
    # 6. Reverse Diffusion
    out_wavs = []
    in_wavs = []
    with torch.inference_mode():  # IMPORTANT
        for s in range(ri_in.shape[0]):
            buffer = ri_in[s:s+1]  # [1,4,F,T]
            est_ri, est_wav = models[model_name].dereverb_batch(buffer, reverse_steps=reverse_steps)  # est_wav: [1,2,L]
            #est_wav = peak_ceiling(est_wav, ceiling=0.9)
            seg_wav = segs[s:s+1]
            est_wav = est_wav = match_segment_rms_db_threshold(est_wav, seg_wav, threshold_db=2.5,max_gain_db=6.0)
            #save them optionally
            # sf.write(os.path.join(save_path , "out_"+str(s)+".wav"),  est_wav.squeeze(0).permute(1,0).cpu().numpy(), sr)
            #seg_wav = peak_ceiling(seg_wav, ceiling=0.9)
            # sf.write(os.path.join(save_path , "in_"+str(s)+".wav"),  seg_wav.squeeze(0).permute(1,0).cpu().numpy(), sr)
            out_wavs.append(est_wav.squeeze(0))  



    out_segs = torch.stack(out_wavs, dim=0)  # [N,2,L] on CPU

    # 7. Reconstruct and convert to audio
    data_out = ola_reconstruct_torch(out_segs, step, orig_len)  # [T,2]
    #print(data_out.shape)
    data_inp = ola_reconstruct_torch(segs, step, orig_len)  # [T,2] (sanity)


    # 8. Compute difference signal
    print('Reconstruction completed!')
    data_diff = data_inp - data_out

    # 9) to numpy for gradio
    data_inp_np = data_inp.detach().cpu().numpy()
    data_out_np = data_out.detach().cpu().numpy()
    data_diff_np = data_diff.detach().cpu().numpy()

    #10. Generate visual specs
    # TODO create spectrograms with plotly (is not working atm)
    spec_inp = generate_spectrogram(data_inp_np, fs)
    spec_out = generate_spectrogram(data_out_np, fs)
    print('Spectrograms completed!')

    # 11) int16 for gr.Audio (optional; gradio also accepts float)
    data_inp_i16 = np.int16(np.clip(data_inp_np, -1, 1) * 32767)
    data_out_i16 = np.int16(np.clip(data_out_np, -1, 1) * 32767)
    data_diff_i16 = np.int16(np.clip(data_diff_np, -1, 1) * 32767)

    return (fs, data_inp_i16), spec_inp, (fs, data_out_i16), spec_out, (fs, data_diff_i16)


# Initiliaze gradio components
input_audio = gr.Audio(label="Upload Audio", type="numpy")

output_audio_in = gr.Audio(label="Input Processed Audio")
output_audio_out = gr.Audio(label="Output Diffused Audio")
output_audio_diff = gr.Audio(label="Difference Audio (Reverb Removed)")
output_image_in = gr.Image(label="Input Processed Spectrogram")
output_image_out = gr.Image(label="Output Diffused Spectrogram")


demo = gr.Interface(
    fn=process_audio,
    inputs=[input_audio, model_selector, steps_slider, overlap_slider],
    outputs=[
        output_audio_in,
        output_image_in,
        output_audio_out,
        output_image_out,
        output_audio_diff,
    ],
    flagging_mode="never",
    examples=example_files,
    description=description_html,
)

demo.launch(server_name="0.0.0.0", share=True, server_port=7870) #share=True