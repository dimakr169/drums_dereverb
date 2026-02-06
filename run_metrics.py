import os
import json
import argparse
import torch
import time
import soundfile as sf


from dataset.config import Config
from dataset.stereo_dataset import build_dataloaders
from utils_inference import build_model_from_entry, load_ckpt_and_ema,\
        ColdDiffInferencer, CDiffuseInferencer, SGMSEInferencer, match_rms_to_ref, \
        ColdDiffInferencer_var

from eval_metrics import MetricComputer, MetricAggregator, write_html_summary

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# ========== Main ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/moisesdb_test_stereo",  #data/out_combined_stereo moisesdb_test_rir-only
                        help="Root of precomputed RI dataset")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="(Optional) batch size for validation loader (if supported by your build_dataloaders)")
    parser.add_argument("--gpu", type=int, default=2) #-1 for CPU
    parser.add_argument("--models-config", default="models.json", 
                        help="Path to models.json (model zoo definition)")
    parser.add_argument("--max-batches", type=int, default=10,
                        help="Stop after this many batches (for quick test). 0 = all.")
    parser.add_argument("--save-wavs", action="store_true",
                        help="If set, saves input/target/estimate WAVs.")
    parser.add_argument("--out-root", type=str, default="eval_out",
                        help="Root folder for evaluation outputs.")
    args = parser.parse_args()

    # ---- device ----
    if args.gpu == 0 or args.gpu == 1 or args.gpu == 2:
        try:
            device = f"cuda:{args.gpu}"
            torch.cuda.set_device(args.gpu)
            print("Using GPU:", device)
        except:
            device = "cpu"
            print("No GPU found, using CPU")
    else:
        device = "cpu"
        print("No GPU found, using CPU")


    # ---- dataset config + dataloaders ----
    pre_params = Config()

    # if your Config/build_dataloaders supports changing val batch size via pre_params, you can do:
    # pre_params.valid_bs = args.batch_size
    # and modify build_dataloaders accordingly; for now we keep it as in training
    print("Building dataloaders...")
    _, val_loader = build_dataloaders(pre_params, args.data_dir, num_workers=8)
    print(f"Validation batches: {len(val_loader)}")

    sr = getattr(pre_params, "sr", 44100)

    # Initialize metrics
    metric_computer = MetricComputer(sr=sr)


    # ---- load model zoo from JSON ----
    with open(args.models_config, "r") as f:
        model_zoo = json.load(f)

    model_names = list(model_zoo.keys())
    print("Models to evaluate:", model_names)

    all_model_stats = {}

    # ---- loop over models ----
    for name in model_names:
        if name not in model_zoo:
            print(f"[WARN] Model '{name}' not found in JSON, skipping.")
            continue

        entry = model_zoo[name]
        print(f"\n=== Evaluating model: {name} (arch={entry['type']}) ===")

        # build model + load checkpoint
        model = build_model_from_entry(entry)
        model, ema  = load_ckpt_and_ema(model, entry, device)

        scheme = entry["scheme"]  # "cold" (default) or "cdiffuse"
        model_type = entry["type"]
        diffusion_steps = entry["diffusion_steps"]


        if scheme == "cold":
            alpha_mode = entry["alpha_mode"]
            cdiff_mode = entry["mode"]

            inferencer = ColdDiffInferencer(
                model=model,
                model_type=model_type,
                pre_params=pre_params,
                diffusion_steps=diffusion_steps,
                alpha_mode=alpha_mode,
                cdiff_mode=cdiff_mode,
                device=device,
            )

        elif scheme == "cold_var":
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


        elif scheme == "cdiffuse":
            sampling_steps = entry["sampling_steps"] # None => full 200-step CDiffuSE
            inferencer = CDiffuseInferencer(
                model=model,
                pre_params=pre_params,
                diffusion_steps=diffusion_steps,
                sampling_steps=sampling_steps,
                device=device,
            )

        elif scheme == "sgmse":
            sampling_steps = entry["sampling_steps"]
            inferencer = SGMSEInferencer(
                model=model,
                pre_params=pre_params,
                device=device,
                diffusion_steps=diffusion_steps ,
                num_steps=sampling_steps
            )

        else:
            raise ValueError(f"Unknown diffusion scheme '{scheme}' for model '{name}'")

        # ---- param count ----
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Trainable parameters: {n_params/1e6:.2f} M")

        # new aggregator per model
        metric_agg = MetricAggregator(keep_per_example=False, param_count=n_params)

        if args.save_wavs: #in case wavs will be saved
            out_dir = os.path.join(args.out_root, name)
            os.makedirs(out_dir, exist_ok=True)
            print("  - Saving outputs to:", out_dir)

        global_ex_idx = 0

        with torch.no_grad():
            for b_idx, batch in enumerate(val_loader):
                start = time.perf_counter()
                print('Batch:' , b_idx)
                reverb_ri, clean_ri = [b.to(device) for b in batch]  # (B,4,F,T)
                B = reverb_ri.size(0)

                _, est_wav = inferencer.dereverb_batch(reverb_ri)
                inp_wav = inferencer.get_signal_from_RI_stft(reverb_ri)
                clean_wav = inferencer.get_signal_from_RI_stft(clean_ri)

                # --- RMS normalization for fair comparison ---
                # Match input & estimate RMS to the clean target RMS, per example
                est_wav = match_rms_to_ref(est_wav, clean_wav)
                inp_wav = match_rms_to_ref(inp_wav, clean_wav)
                # clean_wav stays as reference

                # ---- metrics calculation ----
                try:
                    batch_metrics = metric_computer.compute_batch(
                        inp_wav, clean_wav, est_wav
                    )
                    metric_agg.update(batch_metrics)
                except Exception as e:
                    print("Batch exception due to:", e)
                torch.cuda.synchronize()
                print("Batch time:", time.perf_counter() - start)

                if args.save_wavs:
                    for i in range(B):
                        ex_dir = os.path.join(out_dir, f"val_{global_ex_idx:05d}")
                        os.makedirs(ex_dir, exist_ok=True)

                        inp_np = inp_wav[i].permute(1, 0).cpu().numpy()   # (T,2)
                        tgt_np = clean_wav[i].permute(1, 0).cpu().numpy()
                        est_np = est_wav[i].permute(1, 0).cpu().numpy()

                        sf.write(os.path.join(ex_dir, "input.wav"),   inp_np, sr)
                        sf.write(os.path.join(ex_dir, "target.wav"),  tgt_np, sr)
                        sf.write(os.path.join(ex_dir, "estimate.wav"), est_np, sr)

                        global_ex_idx += 1

                if args.max_batches > 0 and (b_idx + 1) >= args.max_batches:
                    print(f"  - Reached max_batches={args.max_batches}, stopping early.")
                    break

        summary = metric_agg.summary()
        all_model_stats[name] = summary

        # Readable printout
        lines = []
        lines.append(f"== Metrics for model {name} ==")
        lines.append(f"Params: {summary['param_count']/1e6:.2f} M")
        for k, v in summary["metrics"].items():
            lines.append(
                f"{k:24s}: mean={v['mean']:.4f}, std={v['std']:.4f} (N={v['count']})"
            )

        # print to stdout
        print("\n" + "\n".join(lines))

        # save to per-model txt file
        txt_path = os.path.join(args.out_root, f"{name}_metrics.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[INFO] Saved per-model metrics to {txt_path}")

        if ema is not None:
            ema.restore()

    # ====== after all models: HTML summary ======

    higher_is_better = {
        # low-level
        "audio_mae": False,
        "multi_stft_mae": False,
        "phase_mae": False,
        "esr": False,
        "si_sdr": True,
        "si_sdr_impr": True,
        "nmi": True,
        "diff_signal_corr": True,

        # high-level / perceptual
        "mod_spec_dist": False,
        "env_corr": True,
        "tter_absdiff": False,
        "tter_impr": True,
        "hit_tter_absdiff": False,
        "hit_tter_impr": True,
        "onset_F_impr": True,

    }

    metric_groups = {
        "Low-level metrics": [
            "audio_mae",
            "multi_stft_mae",
            "phase_mae",
            "esr",
            "si_sdr", 
            "si_sdr_impr", 
            "nmi",
            "diff_signal_corr",
        ],
        "High-level (perceptual) metrics": [
            "mod_spec_dist",
            "env_corr",
            "tter_absdiff",
            # "tter_impr",
            "hit_tter_absdiff",
            # "hit_tter_impr",
            "onset_F_impr",
        ],
    }

    html_out = os.path.join(args.out_root, "metrics_summary.html")
    write_html_summary(all_model_stats, html_out, higher_is_better, metric_groups)


    print("\nDone!")


if __name__ == "__main__":
    main()