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
        trim_or_pad_range, set_loudness, segment_audio_torch, ola_reconstruct_torch,\
        audio_to_stereo_ri_stft, generate_spectrogram, ColdDiffInferencer_var2

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


# Global Params
gpu_id = 0 # GPU for inference
models_config = 'models_gradio.json' #Load models for inference
save_path = './eval_out/gradio_out' #save diffused segs
ts_min = 2 # min num of seconds for input (model duration)
ts_max = 30  # max num of seconds for input
fs = 44100  # default sample rate
# LUFS = -28.0  # default loudness #-24

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

    if model_type == 'dit3':
        inferencer = ColdDiffInferencer_var2(
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
    else:
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
# Dropdown for OLA reconstruction
ola_modes = ['Direct', 'Residual']
ola_selector = gr.Dropdown(
    choices=ola_modes, label="Select OLA mode", value=ola_modes[0]
)
# Dropdown for solver selection
solver_modes = ['euler', 'heun']
solver_selector = gr.Dropdown(
    choices=solver_modes, label="Select Inference mode", value=solver_modes[0]
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

# Function for processing
def process_audio(audio, model_name, reverse_steps, solver, ola_mode, overlap_ratio):

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
    if model_name == "Cold Diffusion Transformer fixed 57M":
        LUFS = -24.0
    else:
        LUFS = -28.0
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
    delta_wavs = []
    
    with torch.inference_mode():  # IMPORTANT
        for s in range(ri_in.shape[0]):
            buffer = ri_in[s:s+1]  # [1,4,F,T]
            est_ri, est_wav = models[model_name].dereverb_batch(buffer, reverse_steps=reverse_steps, solver=solver)  # est_wav: [1,2,L]
            #est_wav = peak_ceiling(est_wav, ceiling=0.9)
            #seg_wav = segs[s:s+1]
            # est_wav = match_segment_rms_db_threshold(est_wav, seg_wav, threshold_db=2.5,max_gain_db=6.0)
            #save them optionally
            # sf.write(os.path.join(save_path , "out_"+str(s)+".wav"),  est_wav.squeeze(0).permute(1,0).cpu().numpy(), sr)
            #seg_wav = peak_ceiling(seg_wav, ceiling=0.9)
            # sf.write(os.path.join(save_path , "in_"+str(s)+".wav"),  seg_wav.squeeze(0).permute(1,0).cpu().numpy(), sr)
            out_wavs.append(est_wav.squeeze(0))  
            # OLA Residual IMPORTANT: delta = output - input for this segment
            seg_wav = segs[s:s+1]                 # [1,2,L]
            delta = est_wav - seg_wav             # [1,2,L]
            delta_wavs.append(delta.squeeze(0))   # [2,L]



    delta_segs = torch.stack(delta_wavs, dim=0)   # [N,2,L]
    out_segs = torch.stack(out_wavs, dim=0)  # [N,2,L] on CPU

    # 7. Reconstruct and convert to audio
    if ola_mode =='Direct':
        data_out = ola_reconstruct_torch(out_segs, step, orig_len)  # [T,2]
        data_inp = ola_reconstruct_torch(segs, step, orig_len)  # [T,2] (sanity)
    else:
        #Residual OLA
        # 7. Reconstruct delta with OLA
        delta_full = ola_reconstruct_torch(delta_segs, step, orig_len)  # [T,2]
        data_inp_t = torch.from_numpy(data_inp[:orig_len]).to(delta_full.device, dtype=delta_full.dtype)
        data_out = data_inp_t + delta_full

        # optional safety
        data_out = torch.clamp(data_out, -1.0, 1.0)    



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
    spec_diff = generate_spectrogram(data_diff_np, fs)
    print('Spectrograms completed!')

    # 11) int16 for gr.Audio (optional; gradio also accepts float)
    data_inp_i16 = np.int16(np.clip(data_inp_np, -1, 1) * 32767)
    data_out_i16 = np.int16(np.clip(data_out_np, -1, 1) * 32767)
    data_diff_i16 = np.int16(np.clip(data_diff_np, -1, 1) * 32767)

    return (fs, data_inp_i16), spec_inp, (fs, data_out_i16), spec_out, (fs, data_diff_i16), spec_diff


# Initiliaze gradio components
input_audio = gr.Audio(label="Upload Audio", type="numpy")

output_audio_in = gr.Audio(label="Input Processed Audio")
output_audio_out = gr.Audio(label="Output Diffused Audio")
output_audio_diff = gr.Audio(label="Reverb Removed Audio")
output_image_in = gr.Image(label="Input Processed Spectrogram")
output_image_out = gr.Image(label="Output Diffused Spectrogram")
output_image_diff = gr.Image(label="Reverb Removed Spectrogram")


demo = gr.Interface(
    fn=process_audio,
    inputs=[input_audio, model_selector, steps_slider, solver_selector, ola_selector, overlap_slider],
    outputs=[
        output_audio_in,
        output_image_in,
        output_audio_out,
        output_image_out,
        output_audio_diff,
        output_image_diff,
    ],
    flagging_mode="never",
    examples=example_files,
    description=description_html,
)

demo.launch(server_name="0.0.0.0", share=False, server_port=7870) #share=True