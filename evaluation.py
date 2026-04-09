import argparse
import os
import re
import time
import json
import pandas as pd
import torch
import jiwer
from tqdm import tqdm
from praatio import textgrid
from pathlib import Path
import librosa
import numpy as np

from careless_whisper_stream import load_streaming_model
from careless_whisper_stream.streaming_transcribe import transcribe
from training_code.ds_dict import ds_paths

ckpt_root = f"{os.environ.get('HOME')}/ma/data/models/ckpts"


def _extract_epoch_from_name(path: Path) -> int:
    m = re.fullmatch(r"checkpoint-epoch=(-?\d+)\.ckpt", path.name)
    if m:
        return int(m.group(1))
    return -10**9


def _resolve_checkpoint_path(model_run_name: str, checkpoint: int | None) -> Path:
    """
    Resolve a checkpoint inside:
        {ckpt_root}/{model_run_name}/checkpoint/

    Accepted filename style ONLY:
    - checkpoint-epoch=XXXX.ckpt
    - checkpoint-epoch=-001.ckpt

    Rules:
    - if --checkpoint N is given -> use checkpoint-epoch=XXXX.ckpt
    - else prefer best_model_path from Lightning metadata
    - else fall back to highest epoch checkpoint
    """
    run_dir = Path(ckpt_root) / model_run_name
    checkpoint_dir = run_dir / "checkpoint"

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    ckpt_files = [
        p for p in checkpoint_dir.iterdir()
        if p.is_file() and re.fullmatch(r"checkpoint-epoch=-?\d+\.ckpt", p.name)
    ]

    if not ckpt_files:
        raise FileNotFoundError(
            f"No valid checkpoint files found in: {checkpoint_dir}"
        )

    if checkpoint is not None:
        target = checkpoint_dir / f"checkpoint-epoch={checkpoint:04d}.ckpt"

        if target.exists():
            return target

        available_epochs = sorted(_extract_epoch_from_name(p) for p in ckpt_files)
        raise FileNotFoundError(
            f"Requested checkpoint {checkpoint} not found.\n"
            f"Expected: {target}\n"
            f"Available epochs: {available_epochs}"
        )

    for ckpt_path in sorted(ckpt_files):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            callbacks = ckpt.get("callbacks", {})
            for _, cb_state in callbacks.items():
                if isinstance(cb_state, dict):
                    best_model_path = cb_state.get("best_model_path")
                    if best_model_path and os.path.exists(best_model_path):
                        return Path(best_model_path)
        except Exception:
            pass

    return max(ckpt_files, key=_extract_epoch_from_name)


def _resolve_csv_relative_path(csv_path: str, value: str) -> str:
    value = str(value)
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(csv_path)), value))


def extract_words_and_times_from_tg(tg_path):
    """Reconstructs the transcript and timestamps from a TextGrid using praatio."""
    try:
        tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=False)
        text_intervals = tg.getTier("words")

        words = [
            {"word": interval.label.strip(), "start": interval.start, "end": interval.end}
            for interval in text_intervals if interval.label.strip()
        ]
        return words
    except Exception as e:
        print(f"Error parsing TextGrid {tg_path}: {e}")
        return []


def get_gt_prefix_at_time(gt_words, current_time):
    """Returns the ground truth string spoken up to 'current_time'."""
    return " ".join([w["word"] for w in gt_words if w["start"] <= current_time])


def calculate_idsc(ref, hyp):
    """Calculates Insertions, Deletions, Substitutions, and Correct hits."""
    if not ref and not hyp:
        return 0, 0, 0, 0
    if not ref:
        return len(hyp.split()), 0, 0, 0
    if not hyp:
        return 0, len(ref.split()), 0, 0

    out = jiwer.process_words(ref, hyp)
    return out.insertions, out.deletions, out.substitutions, out.hits


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate CarelessWhisper WER on a dataset")

    # Model Setup
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model run name under ckpt_root, e.g. the subfolder inside ckpt_root",
    )
    parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint epoch number to evaluate, e.g. 7 -> checkpoint-0007")
    parser.add_argument("--chunk_size", type=int, default=300, help="Chunk size (gran)")
    parser.add_argument("--multilingual", action="store_true", help="Use multilingual model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset_fraction", type=float, default=1.0, help="Fraction of the dataset, that will be used. 1.0 (100%) by default.")
    parser.add_argument("--dataset_partition", type=str, default="test", help="The partition of the dataset that will be used for evaluation. 'test' by default.")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size during inference.")
    parser.add_argument("-sa_kv_cache", action="store_true", help="Use self-attention KV cache")
    parser.add_argument("-ca_kv_cache", action="store_true", help="Use cross-attention KV cache")
    parser.add_argument("-verbose", action="store_true", help="Prints additional info while evaluating")
    parser.add_argument("-cw", action="store_true", help="Uses a CW whisper base model instead of a local model.")

    # Dataset Setup
    parser.add_argument("--dataset_name", type=str, required=True, help="Key from ds_paths in ds_dict.py")

    args = parser.parse_args()

    if not args.cw:
        ckpt_path = _resolve_checkpoint_path(args.model, args.checkpoint)
        print(f"Using checkpoint: {ckpt_path}")

        # Infer actual model size from checkpoint metadata if available, otherwise fall back to run name
        checkpoint_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hparams = checkpoint_obj.get("hyper_parameters", checkpoint_obj.get("cfg", {}))
        base_model_name = hparams.get("size", args.model)
    else:
        ckpt_path = None

    # 1. Load Model
    model = load_streaming_model(
        name=args.model if args.cw else base_model_name,
        gran=args.chunk_size,
        multilingual=args.multilingual,
        device=args.device,
        local_ckpt_path=None if args.cw else str(ckpt_path),
    )
    model.eval()

    # 2. Load Dataset CSV
    if args.dataset_name not in ds_paths:
        raise ValueError(f"Dataset {args.dataset_name} not found in ds_dict.py")

    csv_path = ds_paths[args.dataset_name][str(args.dataset_partition)]
    print(f"Loading {args.dataset_partition} split from: {csv_path}")
    df = pd.read_csv(csv_path)

    if 0.0 < args.dataset_fraction < 1.0:
        df = df.sample(frac=args.dataset_fraction, random_state=42).reset_index(drop=True)
        print(f"Subsetting dataset to {args.dataset_fraction * 100:.1f}%. New size: {len(df)} samples.")
    elif args.dataset_fraction <= 0 or args.dataset_fraction > 1.0:
        print(f"Warning: dataset_fraction {args.dataset_fraction} is out of bounds. Using full dataset.")

    global_rwer_num, global_rwer_den = 0, 0
    global_arwer_num, global_arwer_den = 0, 0
    all_chunk_latencies = []
    total_audio_duration_sec = 0.0
    total_processing_time_sec = 0.0
    predictions, references = [], []

    # 3. Inference Loop
    print(f"Starting evaluation on {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        wav_path = _resolve_csv_relative_path(csv_path, row["wav_path"])
        tg_path = _resolve_csv_relative_path(csv_path, row["tg_path"])

        audio_duration = librosa.get_duration(path=wav_path)
        total_audio_duration_sec += audio_duration

        gt_words = extract_words_and_times_from_tg(tg_path)
        reference_text = " ".join([w["word"] for w in gt_words]).strip().lower()
        chunk_duration_sec = args.chunk_size / 1000.0

        results = transcribe(
            model=model,
            wav_file=wav_path,
            simulate_stream=True,
            language="en" if not args.multilingual else "auto",
            beam_size=args.beam_size,
            temperature=0,
            ca_kv_cache=args.ca_kv_cache,
            sa_kv_cache=args.sa_kv_cache,
            verbose=False
        )

        for step, res in enumerate(results):
            hyp_text = res.text.strip().lower()

            p_latency = getattr(res, "processing_time", 0.0)
            all_chunk_latencies.append(p_latency)
            total_processing_time_sec += p_latency

            audio_time_rho = (step + 1) * chunk_duration_sec
            gt_text_rho = get_gt_prefix_at_time(gt_words, audio_time_rho).lower()

            i, d, s, c = calculate_idsc(gt_text_rho, hyp_text)
            global_rwer_num += (i + d + s)
            global_rwer_den += (c + d + s)

            real_time_tau = audio_time_rho + p_latency
            gt_text_tau = get_gt_prefix_at_time(gt_words, real_time_tau).lower()

            i_a, d_a, s_a, c_a = calculate_idsc(gt_text_tau, hyp_text)
            global_arwer_num += (i_a + d_a + s_a)
            global_arwer_den += (c_a + d_a + s_a)

        predicted_text = results[-1].text if results else ""
        predictions.append(predicted_text.strip().lower())
        references.append(reference_text)

        if args.verbose:
            print("Pred: " + predicted_text)
            print("Label:" + reference_text)
            print("-" * 30)

    # 4. Final Aggregated Metric Calculation
    wer = jiwer.wer(references, predictions) if references else 0
    rwer = global_rwer_num / global_rwer_den if global_rwer_den > 0 else 0
    arwer = global_arwer_num / global_arwer_den if global_arwer_den > 0 else 0

    avg_latency = np.mean(all_chunk_latencies) if all_chunk_latencies else 0
    rtf = total_processing_time_sec / total_audio_duration_sec if total_audio_duration_sec > 0 else 0

    stats = {
        "model_run": args.model,
        "checkpoint": ckpt_path.name if ckpt_path != None else "N/A",
        "checkpoint_path": str(ckpt_path) if ckpt_path != None else "N/A",
        "dataset": args.dataset_name,
        "partition": args.dataset_partition,
        "fraction": args.dataset_fraction,
        "wer": float(wer),
        "rwer": float(rwer),
        "arwer": float(arwer),
        "avg_latency_ms": float(avg_latency * 1000),
        "rtf": float(rtf),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Always save next to the run directory now
    save_path = ckpt_path.parent.parent / "evaluation.json"
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Stats saved to: {save_path}")

    print("\n" + "=" * 30)
    print(f"RESULTS FOR:")
    print(f"MODEL: {args.model} on DATASET: {args.dataset_name}")
    print(f"PARTITION: {args.dataset_partition} % SAMPLES: {args.dataset_fraction}")
    print(f"CHECKPOINT:    {ckpt_path.name}")
    print(f"WER:           {wer * 100:.2f}%")
    print(f"RWER:          {rwer * 100:.2f}%")
    print(f"ARWER:         {arwer * 100:.2f}%")
    print("-" * 20)
    print(f"Avg Latency:   {avg_latency * 1000:.1f} ms")
    print(f"RTF:           {rtf:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    evaluate()
