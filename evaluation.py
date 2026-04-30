import argparse
import os
import re
import time
import pandas as pd
import torch
import jiwer
from tqdm import tqdm
from praatio import textgrid
from pathlib import Path
import librosa
import numpy as np

from careless_whisper_stream import load_streaming_model
from careless_whisper_stream.normalizers import (
    BasicTextNormalizer,
    EnglishTextNormalizer,
    GermanTextNormalizer,
)
from careless_whisper_stream.streaming_transcribe import transcribe
from training_code.ds_dict import ds_paths
from evaluation_print import print_latest_rows
from evaluation_saving import append_evaluation_row

ckpt_root = f"{os.environ.get('HOME')}/ma/data/models/ckpts"
evaluation_file = f"{os.environ.get('HOME')}/ma/data/evaluation.csv"


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


def _infer_language(
    dataset_name: str,
    explicit_lang: str | None = None,
    checkpoint_cfg: dict | None = None,
) -> str | None:
    if explicit_lang:
        return _canonicalize_language(explicit_lang)

    if checkpoint_cfg:
        for key in ("lang", "language"):
            value = checkpoint_cfg.get(key)
            if value:
                return _canonicalize_language(value)

    dataset_upper = dataset_name.upper()
    if "CV-DE" in dataset_upper or "-DE-" in dataset_upper or dataset_upper.endswith("-DE"):
        return "de"
    if "LIBRI" in dataset_upper or "-EN-" in dataset_upper or dataset_upper.endswith("-EN"):
        return "en"

    return None


def _canonicalize_language(language: str | None) -> str | None:
    if language is None:
        return None

    language = str(language).strip().lower()
    if language in {"en", "en-us", "en-gb", "english"}:
        return "en"
    if language in {"de", "de-de", "german", "deutsch"}:
        return "de"

    return language or None


def _get_normalizer(language: str | None):
    if language == "en":
        return EnglishTextNormalizer()

    if language == "de":
        try:
            return GermanTextNormalizer()
        except ImportError as exc:
            print(f"Warning: {exc} Falling back to BasicTextNormalizer for German evaluation.")
            return BasicTextNormalizer()

    return BasicTextNormalizer()


def _normalize_for_eval(text: str, normalizer) -> str:
    return normalizer(str(text or "")).strip()


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


def _build_strict_word_buffer(results, normalizer, correction_distance: int = 2):
    """
    Build a constrained word buffer from streaming hypotheses.

    After each forward pass, the model may only revise the last
    `correction_distance` words of the current buffer.
    """
    correction_distance = max(0, int(correction_distance))
    strict_words = []

    for res in results:
        candidate_words = _normalize_for_eval(getattr(res, "text", ""), normalizer).split()

        if not strict_words:
            strict_words = candidate_words
            continue

        frozen_prefix_len = max(0, len(strict_words) - correction_distance)
        strict_words = strict_words[:frozen_prefix_len] + candidate_words[frozen_prefix_len:]

    return " ".join(strict_words)


def calculate_word_instability(results, normalizer):
    """
    Count how many previously emitted words get revised in later predictions.

    For each pair of consecutive streaming hypotheses, we find their longest
    common prefix in word space. Every previously emitted word beyond that
    prefix is treated as a revision event. This lets the same final word accrue
    multiple changes over time if the hypothesis keeps getting rewritten.
    """
    normalized_hypotheses = [
        _normalize_for_eval(getattr(res, "text", ""), normalizer).split()
        for res in results
    ]

    if not normalized_hypotheses:
        return 0, 0

    changed_word_count = 0
    previous_words = normalized_hypotheses[0]

    for current_words in normalized_hypotheses[1:]:
        common_prefix_len = 0
        for prev_word, curr_word in zip(previous_words, current_words):
            if prev_word != curr_word:
                break
            common_prefix_len += 1

        changed_word_count += max(0, len(previous_words) - common_prefix_len)
        previous_words = current_words

    total_word_count = len(normalized_hypotheses[-1])
    return changed_word_count, total_word_count


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
    parser.add_argument("--lang", type=str, default=None, help="Language code for normalization/transcription, e.g. en or de. If omitted, infer from checkpoint or dataset.")
    parser.add_argument("--strict_k", type=int, default=2, help="Max word correction distance for strict WER.")
    parser.add_argument("-sa_kv_cache", action="store_true", help="Use self-attention KV cache")
    parser.add_argument("-ca_kv_cache", action="store_true", help="Use cross-attention KV cache")
    parser.add_argument("-verbose", action="store_true", help="Prints additional info while evaluating")
    parser.add_argument("-cw", action="store_true", help="Uses a CW whisper base model instead of a local model.")

    # Dataset Setup
    parser.add_argument("--dataset_name", type=str, required=True, help="Key from ds_paths in ds_dict.py")

    args = parser.parse_args()

    if not args.cw:
        ckpt_path = _resolve_checkpoint_path(args.model, args.checkpoint)
        print(f"Using checkpoint of local model: {ckpt_path}")

        # Infer actual model size from checkpoint metadata if available, otherwise fall back to run name
        checkpoint_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hparams = checkpoint_obj.get("hyper_parameters", checkpoint_obj.get("cfg", {}))
        base_model_name = hparams.get("size", args.model)
        print(f"model size: {base_model_name}")
    else:
        print(f"Using cw model. size: {args.model} chunk size: {args.chunk_size}.")
        ckpt_path = None
        hparams = {}

    default_language = _infer_language(args.dataset_name, explicit_lang=args.lang, checkpoint_cfg=hparams)
    default_normalizer = _get_normalizer(default_language)
    print(f"Evaluation language: {default_language or 'auto/basic'}")
    print(f"Default normalizer: {type(default_normalizer).__name__}")

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
    global_wer_i, global_wer_d, global_wer_s, global_wer_c = 0, 0, 0, 0
    global_strict_wer_i, global_strict_wer_d, global_strict_wer_s, global_strict_wer_c = 0, 0, 0, 0
    global_wir_changes, global_wir_total_words = 0, 0

    all_chunk_latencies = []
    total_audio_duration_sec = 0.0
    total_processing_time_sec = 0.0
    predictions, references = [], []
    strict_predictions = []

    chunk_duration_sec = model.encoder.gran * 0.02
    print(f"model chunk size (s): {chunk_duration_sec}")

    # 3. Inference Loop
    print(f"Starting evaluation on {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        wav_path = _resolve_csv_relative_path(csv_path, row["wav_path"])
        tg_path = _resolve_csv_relative_path(csv_path, row["tg_path"])
        row_language = (
            _canonicalize_language(row["lang"])
            if args.multilingual and "lang" in row and pd.notna(row["lang"])
            else default_language
        )
        normalizer = _get_normalizer(row_language)

        audio_duration = librosa.get_duration(path=wav_path)
        total_audio_duration_sec += audio_duration

        gt_words = extract_words_and_times_from_tg(tg_path)
        reference_text = _normalize_for_eval(" ".join([w["word"] for w in gt_words]), normalizer)

        results = transcribe(
            model=model,
            wav_file=wav_path,
            simulate_stream=True,
            language=row_language if row_language else ("auto" if args.multilingual else "en"),
            beam_size=args.beam_size,
            temperature=0,
            ca_kv_cache=args.ca_kv_cache,
            sa_kv_cache=args.sa_kv_cache,
            verbose=False
        )

        for step, res in enumerate(results):
            hyp_text = _normalize_for_eval(res.text, normalizer)

            p_latency = getattr(res, "processing_time", 0.0)
            all_chunk_latencies.append(p_latency)
            total_processing_time_sec += p_latency

            audio_time_rho = (step + 1) * chunk_duration_sec
            gt_text_rho = _normalize_for_eval(get_gt_prefix_at_time(gt_words, audio_time_rho), normalizer)

            i, d, s, c = calculate_idsc(gt_text_rho, hyp_text)
            global_rwer_num += (i + d + s)
            global_rwer_den += (c + d + s)

            real_time_tau = audio_time_rho + p_latency
            gt_text_tau = _normalize_for_eval(get_gt_prefix_at_time(gt_words, real_time_tau), normalizer)

            i_a, d_a, s_a, c_a = calculate_idsc(gt_text_tau, hyp_text)
            global_arwer_num += (i_a + d_a + s_a)
            global_arwer_den += (c_a + d_a + s_a)

        predicted_text = results[-1].text if results else ""
        normalized_prediction = _normalize_for_eval(predicted_text, normalizer)
        strict_prediction = _build_strict_word_buffer(results, normalizer, args.strict_k)
        wir_changes, wir_total_words = calculate_word_instability(results, normalizer)
        predictions.append(normalized_prediction)
        strict_predictions.append(strict_prediction)
        references.append(reference_text)
        global_wir_changes += wir_changes
        global_wir_total_words += wir_total_words

        # count IDS once per sample, using final hypothesis vs full reference
        i_f, d_f, s_f, c_f = calculate_idsc(reference_text, normalized_prediction)
        global_wer_i += i_f
        global_wer_d += d_f
        global_wer_s += s_f
        global_wer_c += c_f

        i_strict, d_strict, s_strict, c_strict = calculate_idsc(reference_text, strict_prediction)
        global_strict_wer_i += i_strict
        global_strict_wer_d += d_strict
        global_strict_wer_s += s_strict
        global_strict_wer_c += c_strict

        if args.verbose:
            print("Pred: " + normalized_prediction)
            print("Strict Pred: " + strict_prediction)
            print("Label:" + reference_text)
            print(f"I={i_f}, D={d_f}, S={s_f}, C={c_f}")
            print(f"WIR changes={wir_changes}, total_words={wir_total_words}")
            print("-" * 30)

    # 4. Final Aggregated Metric Calculation
    wer = jiwer.wer(references, predictions) if references else 0
    strict_wer = jiwer.wer(references, strict_predictions) if references else 0
    rwer = global_rwer_num / global_rwer_den if global_rwer_den > 0 else 0
    arwer = global_arwer_num / global_arwer_den if global_arwer_den > 0 else 0
    wir = global_wir_changes / global_wir_total_words if global_wir_total_words > 0 else 0

    avg_latency = np.mean(all_chunk_latencies) if all_chunk_latencies else 0
    rtf = total_processing_time_sec / total_audio_duration_sec if total_audio_duration_sec > 0 else 0

    stats = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_file": evaluation_file,
        "model_name": args.model,
        "model_run": args.model,
        "base_model_name": args.model if args.cw else base_model_name,
        "is_cw_model": bool(args.cw),
        "checkpoint": "" if ckpt_path is None else ckpt_path.name,
        "checkpoint_epoch": "" if ckpt_path is None else int(_extract_epoch_from_name(ckpt_path)),
        "checkpoint_path": "" if ckpt_path is None else str(ckpt_path),
        "dataset": args.dataset_name,
        "dataset_csv": str(csv_path),
        "partition": args.dataset_partition,
        "fraction": float(args.dataset_fraction),
        "sample_count": int(len(df)),
        "chunk_size": int(args.chunk_size),
        "chunk_duration_sec": float(chunk_duration_sec),
        "beam_size": int(args.beam_size),
        "strict_k": int(args.strict_k),
        "language": default_language or "",
        "multilingual": bool(args.multilingual),
        "device": args.device,
        "sa_kv_cache": bool(args.sa_kv_cache),
        "ca_kv_cache": bool(args.ca_kv_cache),
        "wer": float(wer),
        "strict_wer": float(strict_wer),
        "rwer": float(rwer),
        "arwer": float(arwer),
        "wir": float(wir),
        "wir_changed_words": int(global_wir_changes),
        "wir_total_words": int(global_wir_total_words),
        "wer_insertions": int(global_wer_i),
        "wer_deletions": int(global_wer_d),
        "wer_substitutions": int(global_wer_s),
        "wer_correct": int(global_wer_c),
        "strict_wer_insertions": int(global_strict_wer_i),
        "strict_wer_deletions": int(global_strict_wer_d),
        "strict_wer_substitutions": int(global_strict_wer_s),
        "strict_wer_correct": int(global_strict_wer_c),
        "avg_latency_ms": float(avg_latency * 1000),
        "rtf": float(rtf),
        "total_audio_duration_sec": float(total_audio_duration_sec),
        "total_processing_time_sec": float(total_processing_time_sec),
    }

    append_evaluation_row(evaluation_file, stats)
    print(f"Stats saved to: {evaluation_file}")
    print()
    print_latest_rows(evaluation_file, row_count=1)


if __name__ == "__main__":
    evaluate()
