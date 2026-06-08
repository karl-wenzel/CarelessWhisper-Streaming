import argparse
import difflib
import json
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
from evaluation_caching import (
    build_cache_identity,
    cache_records_to_results,
    dataset_selection_fingerprint,
    evaluation_cache_dir,
    file_fingerprint,
    load_cached_run,
    pre_evaluation_parameters,
    sample_cache_record,
    save_cached_run,
    validate_parameter_classification,
)
from evaluation_print import print_latest_rows
from evaluation_saving import append_evaluation_row

ckpt_root = f"{os.environ.get('HOME')}/ma/data/models/ckpts"
evaluation_file = f"{os.environ.get('HOME')}/ma/data/evaluation.csv"


def _get_hparam(hparams, key: str, default=None):
    missing = object()

    def find_value(container):
        if container is None:
            return missing

        if isinstance(container, dict):
            if key in container:
                return container[key]

            for nested_key in ("hyper_parameters", "cfg"):
                nested_cfg = container.get(nested_key)
                if nested_cfg is not None:
                    nested_value = find_value(nested_cfg)
                    if nested_value is not missing:
                        return nested_value

            return missing

        if hasattr(container, key):
            return getattr(container, key)

        nested_cfg = getattr(container, "cfg", None)
        if nested_cfg is not None:
            return find_value(nested_cfg)

        return missing

    value = find_value(hparams)
    return default if value is missing else value


def _load_run_cfg(model_run_name: str) -> dict:
    cfg_path = Path(ckpt_root) / model_run_name / "cfg.json"
    if not cfg_path.exists():
        return {}

    try:
        with open(cfg_path, "r") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Warning: could not read run cfg from {cfg_path}: {exc}")
        return {}


def _score_to_float(score) -> float | None:
    if score is None:
        return None

    if torch.is_tensor(score):
        if score.numel() != 1:
            return None
        return float(score.detach().cpu().item())

    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def _resolve_checkpoint_reference(path_value, checkpoint_dir: Path, ckpt_by_name: dict[str, Path]) -> Path | None:
    if not path_value:
        return None

    candidate = Path(str(path_value))
    if candidate.exists():
        return candidate

    # Lightning stores absolute best_model_path values. Evaluation can run in a
    # moved container path, so resolve by filename inside the requested run dir.
    return ckpt_by_name.get(candidate.name)


def _find_best_wer_checkpoint_from_callbacks(
    ckpt: dict,
    checkpoint_dir: Path,
    ckpt_by_name: dict[str, Path],
) -> tuple[float | None, Path | None, str]:
    scored_candidates: list[tuple[float, Path, str]] = []
    unscored_candidates: list[tuple[Path, str]] = []

    for callback_name, cb_state in ckpt.get("callbacks", {}).items():
        if not isinstance(cb_state, dict):
            continue

        monitor = str(cb_state.get("monitor", ""))
        if "wer" not in monitor.lower():
            continue

        best_k_models = cb_state.get("best_k_models")
        if isinstance(best_k_models, dict):
            for raw_path, raw_score in best_k_models.items():
                resolved_path = _resolve_checkpoint_reference(raw_path, checkpoint_dir, ckpt_by_name)
                score = _score_to_float(raw_score)
                if resolved_path is not None and score is not None:
                    scored_candidates.append((score, resolved_path, f"{callback_name}:best_k_models"))

        best_model_path = _resolve_checkpoint_reference(
            cb_state.get("best_model_path"),
            checkpoint_dir,
            ckpt_by_name,
        )
        if best_model_path is not None:
            score = _score_to_float(cb_state.get("best_model_score"))
            if score is None:
                unscored_candidates.append((best_model_path, f"{callback_name}:best_model_path"))
            else:
                scored_candidates.append((score, best_model_path, f"{callback_name}:best_model_path"))

    if scored_candidates:
        score, path, source = min(scored_candidates, key=lambda item: item[0])
        return score, path, source

    if unscored_candidates:
        path, source = unscored_candidates[0]
        return None, path, source

    return None, None, ""


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

    ckpt_by_name = {p.name: p for p in ckpt_files}
    best_score = None
    best_path = None
    best_source = ""

    for ckpt_path in sorted(ckpt_files):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            score, path, source = _find_best_wer_checkpoint_from_callbacks(
                ckpt,
                checkpoint_dir,
                ckpt_by_name,
            )
            if path is None:
                continue
            if score is None:
                if best_path is None:
                    best_path = path
                    best_source = source
                continue
            if best_score is None or score < best_score:
                best_score = score
                best_path = path
                best_source = source
        except Exception as exc:
            print(f"Warning: could not inspect checkpoint metadata in {ckpt_path}: {exc}")

    if best_path is not None:
        if best_score is None:
            print(f"Selected best checkpoint from Lightning metadata: {best_path} ({best_source})")
        else:
            print(f"Selected best WER checkpoint from Lightning metadata: {best_path} ({best_source}, score={best_score:.6f})")
        return best_path

    print("Warning: no usable best WER checkpoint metadata found; falling back to highest epoch checkpoint.")
    return max(ckpt_files, key=_extract_epoch_from_name)


def _resolve_csv_relative_path(csv_path: str, value: str) -> str:
    value = str(value)
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(csv_path)), value))


def _cache_dataset_sample_records(df, csv_path: str) -> list[dict]:
    sample_records = []
    for sample_index, (_, row) in enumerate(df.iterrows()):
        wav_path = _resolve_csv_relative_path(csv_path, row["wav_path"])
        tg_path = _resolve_csv_relative_path(csv_path, row["tg_path"])
        record = {
            "sample_index": sample_index,
            "wav_path": wav_path,
            "tg_path": tg_path,
            "wav_file": file_fingerprint(wav_path),
            "tg_file": file_fingerprint(tg_path),
        }
        if "lang" in row and pd.notna(row["lang"]):
            record["lang"] = _canonicalize_language(row["lang"])
        if "raw_text" in row and pd.notna(row["raw_text"]):
            record["raw_text"] = str(row["raw_text"])
        sample_records.append(record)
    return sample_records


def _infer_language(
    dataset_name: str,
    explicit_lang: str | None = None,
    checkpoint_cfg=None,
) -> str | None:
    if explicit_lang:
        return _canonicalize_language(explicit_lang)

    if checkpoint_cfg:
        for key in ("lang", "language"):
            value = _get_hparam(checkpoint_cfg, key)
            if value:
                return _canonicalize_language(value)

    dataset_upper = dataset_name.upper()
    if "CV-DE" in dataset_upper or "-DE-" in dataset_upper or dataset_upper.endswith("-DE"):
        return "de"
    if (
        "LIBRI" in dataset_upper
        or "REV" in dataset_upper
        or "TEDLIUM" in dataset_upper
        or "-EN-" in dataset_upper
        or dataset_upper.endswith("-EN")
    ):
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


def _resolve_encoder_positional_mode(
    requested_mode: str,
    model_run_name: str,
    checkpoint_hparams,
) -> str:
    if requested_mode != "auto":
        return requested_mode

    checkpoint_mode = _get_hparam(checkpoint_hparams, "encoder_positional_mode")
    if checkpoint_mode in {"sinusoidal", "alibi"}:
        return checkpoint_mode

    run_cfg = _load_run_cfg(model_run_name)
    cfg_mode = _get_hparam(run_cfg, "encoder_positional_mode")
    if cfg_mode in {"sinusoidal", "alibi"}:
        return cfg_mode

    if "alibi" in model_run_name.lower():
        print(
            "Warning: run name contains 'alibi' but checkpoint/cfg metadata does not "
            "declare encoder_positional_mode. Inferring alibi; pass "
            "--encoder_positional_mode sinusoidal to override this for non-ALiBi runs."
        )
        return "alibi"

    return "sinusoidal"


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
        candidate_words = _normalize_for_eval(_result_text_for_eval(res), normalizer).split()

        if not strict_words:
            strict_words = candidate_words
            continue

        frozen_prefix_len = max(0, len(strict_words) - correction_distance)
        strict_words = strict_words[:frozen_prefix_len] + candidate_words[frozen_prefix_len:]

    return " ".join(strict_words)


def _row_text(row, column_name: str) -> str:
    if column_name not in row or pd.isna(row[column_name]):
        return ""
    return str(row[column_name]).strip()


def _reference_text_for_sample(row, gt_words, normalizer) -> str:
    raw_text = _normalize_for_eval(_row_text(row, "raw_text"), normalizer)
    if raw_text:
        return raw_text

    return _normalize_for_eval(" ".join([w["word"] for w in gt_words]), normalizer)


def _reference_debug_lines(wav_path, tg_path, row, gt_words, normalizer, audio_duration=None):
    """
    Build compact verbose diagnostics for suspected REVLONG reference mismatches.

    Some long-form samples can have CSV text and TextGrid words that are clipped
    differently, so verbose mode prints the first word-level disagreement.
    """
    raw_text = _normalize_for_eval(_row_text(row, "raw_text"), normalizer)
    tg_text = _normalize_for_eval(" ".join([w["word"] for w in gt_words]), normalizer)
    raw_words = raw_text.split()
    tg_words = tg_text.split()

    lines = [
        f"WAV: {wav_path}",
        f"TextGrid: {tg_path}",
        f"raw_text words: {len(raw_words)}",
        f"TextGrid words: {len(tg_words)}",
    ]
    if audio_duration is not None:
        last_word_end = max((w["end"] for w in gt_words), default=0.0)
        # MFA can align only the supplied transcript. A large uncovered audio tail
        # means WER may count real spoken words as insertions.
        lines.append(f"audio duration: {audio_duration:.2f}s")
        lines.append(f"last TextGrid word end: {last_word_end:.2f}s")
        lines.append(f"uncovered audio tail: {max(0.0, audio_duration - last_word_end):.2f}s")

    if raw_words == tg_words:
        lines.append("raw_text/TextGrid: match")
        return lines

    lines.append("raw_text/TextGrid: mismatch")
    matcher = difflib.SequenceMatcher(a=raw_words, b=tg_words, autojunk=False)
    for tag, raw_start, raw_end, tg_start, tg_end in matcher.get_opcodes():
        if tag == "equal":
            continue

        raw_context_start = max(0, raw_start - 8)
        raw_context_end = min(len(raw_words), raw_end + 8)
        tg_context_start = max(0, tg_start - 8)
        tg_context_end = min(len(tg_words), tg_end + 8)
        lines.append(
            f"First ref mismatch: {tag} "
            f"raw[{raw_start}:{raw_end}] tg[{tg_start}:{tg_end}]"
        )
        lines.append("raw context: " + " ".join(raw_words[raw_context_start:raw_context_end]))
        lines.append("tg context: " + " ".join(tg_words[tg_context_start:tg_context_end]))
        break

    return lines


def _result_text_for_eval(result) -> str:
    full_text = str(getattr(result, "full_text", "") or "").strip()
    if full_text:
        return full_text
    return str(getattr(result, "text", "") or "")


def calculate_word_instability(results, normalizer):
    """
    Count how many previously emitted words get revised in later predictions.

    For each pair of consecutive streaming hypotheses, we find their longest
    common prefix in word space. Every previously emitted word beyond that
    prefix is treated as a revision event. This lets the same final word accrue
    multiple changes over time if the hypothesis keeps getting rewritten.
    """
    normalized_hypotheses = [
        _normalize_for_eval(_result_text_for_eval(res), normalizer).split()
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


def calculate_word_instability_with_suffix_tolerance(results, normalizer, suffix_tolerance: int = 0):
    """
    Count revised words while ignoring changes inside the trailing
    `suffix_tolerance` words of the previous hypothesis.
    """
    normalized_hypotheses = [
        _normalize_for_eval(_result_text_for_eval(res), normalizer).split()
        for res in results
    ]

    if not normalized_hypotheses:
        return 0, 0

    suffix_tolerance = max(0, int(suffix_tolerance))
    changed_word_count = 0
    previous_words = normalized_hypotheses[0]

    for current_words in normalized_hypotheses[1:]:
        common_prefix_len = 0
        for prev_word, curr_word in zip(previous_words, current_words):
            if prev_word != curr_word:
                break
            common_prefix_len += 1

        countable_previous_len = max(0, len(previous_words) - suffix_tolerance)
        changed_word_count += max(0, countable_previous_len - common_prefix_len)
        previous_words = current_words

    total_word_count = len(normalized_hypotheses[-1])
    return changed_word_count, total_word_count


def _format_wir_summary(wir_stats_by_n: dict[int, dict[str, int | float]]) -> str:
    summary_parts = []
    for suffix_tolerance in sorted(wir_stats_by_n):
        stats = wir_stats_by_n[suffix_tolerance]
        summary_parts.append(
            f"n={suffix_tolerance}: {stats['wir'] * 100:.2f}% "
            f"({stats['changed_words']}/{stats['total_words']})"
        )
    return " | ".join(summary_parts)


def _format_strict_summary(
    strict_wer_by_k: dict[int, float],
    strict_counts_by_k: dict[int, dict[str, int]],
) -> str:
    summary_parts = []
    for strict_k in sorted(strict_wer_by_k):
        counts = strict_counts_by_k[strict_k]
        summary_parts.append(
            f"k={strict_k}: {strict_wer_by_k[strict_k] * 100:.2f}% "
            f"(I/D/S/C={counts['i']}/{counts['d']}/{counts['s']}/{counts['c']})"
        )
    return " | ".join(summary_parts)


def _format_time_bin_wer_summary(time_bin_counts: dict[int, dict[str, int]], bin_seconds: float) -> str:
    summary_parts = []
    for bin_index in sorted(time_bin_counts):
        counts = time_bin_counts[bin_index]
        denominator = counts["c"] + counts["d"] + counts["s"]
        errors = counts["i"] + counts["d"] + counts["s"]
        wer_text = f"{(errors / denominator) * 100:.2f}%" if denominator > 0 else "N/A"
        start = bin_index * bin_seconds
        end = start + bin_seconds
        summary_parts.append(
            f"{start:g}-{end:g}s: {wer_text} "
            f"(samples={counts['samples']}, I/D/S/C={counts['i']}/{counts['d']}/{counts['s']}/{counts['c']})"
        )
    return " | ".join(summary_parts)


def _hypothesis_at_or_before(results, time_sec: float, chunk_duration_sec: float, normalizer) -> str:
    if not results or time_sec <= 0:
        return ""

    result_index = int((time_sec / chunk_duration_sec) + 1e-9) - 1
    if result_index < 0:
        return ""

    result_index = min(result_index, len(results) - 1)
    return _normalize_for_eval(_result_text_for_eval(results[result_index]), normalizer)


def _new_hypothesis_text_for_interval(
    results,
    start_sec: float,
    end_sec: float,
    chunk_duration_sec: float,
    normalizer,
) -> str:
    start_words = _hypothesis_at_or_before(results, start_sec, chunk_duration_sec, normalizer).split()
    end_words = _hypothesis_at_or_before(results, end_sec, chunk_duration_sec, normalizer).split()

    # Time-bin WER is meant to isolate intervals. Since streaming hypotheses are
    # cumulative, remove the words that were already emitted at bin start so
    # 0-5s content is not scored again in the 5-10s bin.
    return " ".join(end_words[len(start_words):])


def _reference_text_for_interval(gt_words, start_sec: float, end_sec: float, normalizer) -> str:
    interval_words = [
        w["word"]
        for w in gt_words
        if start_sec <= w["start"] < end_sec
    ]
    return _normalize_for_eval(" ".join(interval_words), normalizer)


def _accumulate_time_bin_wer(
    time_bin_counts: dict[int, dict[str, int]],
    results,
    gt_words,
    audio_duration: float,
    chunk_duration_sec: float,
    bin_seconds: float,
    normalizer,
) -> None:
    if not results:
        return

    bin_count = int(np.ceil(audio_duration / bin_seconds))
    for bin_index in range(bin_count):
        start_sec = bin_index * bin_seconds
        end_sec = min(audio_duration, start_sec + bin_seconds)
        hypothesis_end_sec = len(results) * chunk_duration_sec if bin_index == bin_count - 1 else end_sec
        reference_text = _reference_text_for_interval(gt_words, start_sec, end_sec, normalizer)
        hypothesis_text = _new_hypothesis_text_for_interval(
            results,
            start_sec,
            hypothesis_end_sec,
            chunk_duration_sec,
            normalizer,
        )

        i, d, s, c = calculate_idsc(reference_text, hypothesis_text)
        counts = time_bin_counts.setdefault(
            bin_index,
            {"i": 0, "d": 0, "s": 0, "c": 0, "samples": 0},
        )
        counts["i"] += i
        counts["d"] += d
        counts["s"] += s
        counts["c"] += c
        counts["samples"] += 1


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate CarelessWhisper WER on a dataset")

    # When adding evaluation parameters, classify them in evaluation_caching.py
    # and run: python -m unittest tests.test_evaluation_caching_contract
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
    parser.add_argument("--dataset_sample_count", type=int, default=None, help="Evaluate on exactly this many randomly sampled dataset rows. Mutually exclusive with --dataset_fraction below 1.0.")
    parser.add_argument("--dataset_partition", type=str, default="test", help="The partition of the dataset that will be used for evaluation. 'test' by default.")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size during inference.")
    parser.add_argument("--max_sec_context", type=int, default=30, help="Max audio context window in seconds before legacy streaming reset.")
    parser.add_argument("--lang", type=str, default=None, help="Language code for normalization/transcription, e.g. en or de. If omitted, infer from checkpoint or dataset.")
    parser.add_argument(
        "--encoder_positional_mode",
        choices=["auto", "sinusoidal", "alibi"],
        default="auto",
        help="Encoder positional mode. auto reads checkpoint/cfg metadata, then infers ALiBi from run names containing 'alibi'.",
    )
    parser.add_argument(
        "--strict_k",
        type=int,
        nargs="*",
        default=[2],
        help="Word correction distances to evaluate for strict WER, e.g. --strict_k 0 1 2.",
    )
    parser.add_argument("--wir_n", type=int, nargs="*", default=[], help="Additional WIR suffix tolerances to evaluate, e.g. --wir_n 0 1 2. n means the last n words are ignored when counting WIR changes.")
    parser.add_argument("-sa_kv_cache", action="store_true", help="Use self-attention KV cache")
    parser.add_argument("-ca_kv_cache", action="store_true", help="Use cross-attention KV cache")
    parser.add_argument("--use_sliding_encoder_cache", action="store_true", help="Slide encoder KV cache instead of resetting at max context")
    parser.add_argument("--disable_encoder_kv_cache", action="store_true", help="Recompute the full encoder prefix at every streaming step for cache diagnostics.")
    parser.add_argument("--reset_decoder_on_encoder_slide", action="store_true", help="Roll decoder prefix tokens into prompt as sliding encoder cache prunes old audio.")
    parser.add_argument("--decoder_roll_overlap_seconds", type=float, default=5.0, help="Seconds of retained encoder audio kept as overlap before the active decoder prefix during rolling decoder reset.")
    parser.add_argument("--time_bin_wer", action="store_true", help="Print and save interval WER grouped by elapsed-audio time bins.")
    parser.add_argument("--time_bin_seconds", type=float, default=5.0, help="Bin size in seconds for --time_bin_wer.")
    parser.add_argument("-verbose", action="store_true", help="Prints additional info while evaluating")
    parser.add_argument("-cw", action="store_true", help="Uses a CW whisper base model instead of a local model.")
    parser.add_argument("--no_evaluation_cache", action="store_true", help="Always recalculate transcribe outputs instead of reading the evaluation cache.")

    # Dataset Setup
    parser.add_argument("--dataset_name", type=str, required=True, help="Key from ds_paths in ds_dict.py")

    args = parser.parse_args()
    validate_parameter_classification(list(vars(args).keys()))

    if args.dataset_sample_count is not None and args.dataset_fraction != 1.0:
        raise ValueError("--dataset_sample_count cannot be used together with --dataset_fraction.")
    if args.dataset_sample_count is not None and args.dataset_sample_count <= 0:
        raise ValueError("--dataset_sample_count must be a positive integer.")
    if args.time_bin_seconds <= 0:
        raise ValueError("--time_bin_seconds must be positive.")
    if args.reset_decoder_on_encoder_slide and not args.use_sliding_encoder_cache:
        raise ValueError("--reset_decoder_on_encoder_slide requires --use_sliding_encoder_cache.")
    if args.decoder_roll_overlap_seconds < 0:
        raise ValueError("--decoder_roll_overlap_seconds must be non-negative.")
    if args.decoder_roll_overlap_seconds >= args.max_sec_context:
        raise ValueError("--decoder_roll_overlap_seconds must be smaller than --max_sec_context.")

    if not args.cw:
        ckpt_path = _resolve_checkpoint_path(args.model, args.checkpoint)
        print(f"Using checkpoint of local model: {ckpt_path}")

        # Infer actual model size from checkpoint metadata if available, otherwise fall back to run name
        checkpoint_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hparams = checkpoint_obj
        base_model_name = _get_hparam(hparams, "size", args.model)
        print(f"model size: {base_model_name}")
    else:
        print(f"Using cw model. size: {args.model} chunk size: {args.chunk_size}.")
        ckpt_path = None
        hparams = {}

    encoder_positional_mode = _resolve_encoder_positional_mode(
        args.encoder_positional_mode,
        args.model,
        hparams,
    )
    default_language = _infer_language(args.dataset_name, explicit_lang=args.lang, checkpoint_cfg=hparams)
    default_normalizer = _get_normalizer(default_language)
    if len(args.strict_k) == 0:
        raise ValueError(
            "--strict_k was provided without values. Provide one or more non-negative integers, e.g. --strict_k 0 1 2."
        )
    # Normalize to stable unique ordering so CSV/report comparisons stay deterministic across repeated runs.
    strict_k_values = sorted({int(k) for k in args.strict_k})
    if any(k < 0 for k in strict_k_values):
        raise ValueError("--strict_k values must be non-negative integers.")
    wir_suffix_tolerances = sorted({0, *args.wir_n})
    if any(n < 0 for n in wir_suffix_tolerances):
        raise ValueError("--wir_n values must be non-negative integers.")
    print(f"Evaluation language: {default_language or 'auto/basic'}")
    print(f"Default normalizer: {type(default_normalizer).__name__}")
    print(f"Encoder positional mode: {encoder_positional_mode}")
    print(f"Strict correction distances: {strict_k_values}")
    print(f"WIR suffix tolerances: {wir_suffix_tolerances}")

    # 1. Load Dataset CSV
    if args.dataset_name not in ds_paths:
        raise ValueError(f"Dataset {args.dataset_name} not found in ds_dict.py")

    csv_path = ds_paths[args.dataset_name][str(args.dataset_partition)]
    print(f"Loading {args.dataset_partition} split from: {csv_path}")
    df = pd.read_csv(csv_path)

    if args.dataset_sample_count is not None:
        if args.dataset_sample_count < len(df):
            df = df.sample(n=args.dataset_sample_count, random_state=42).reset_index(drop=True)
            print(f"Subsetting dataset to {args.dataset_sample_count} samples.")
        else:
            print(f"Warning: dataset_sample_count {args.dataset_sample_count} is >= dataset size {len(df)}. Using full dataset.")
    elif 0.0 < args.dataset_fraction < 1.0:
        df = df.sample(frac=args.dataset_fraction, random_state=42).reset_index(drop=True)
        print(f"Subsetting dataset to {args.dataset_fraction * 100:.1f}%. New size: {len(df)} samples.")
    elif args.dataset_fraction <= 0 or args.dataset_fraction > 1.0:
        print(f"Warning: dataset_fraction {args.dataset_fraction} is out of bounds. Using full dataset.")

    cache_dir = evaluation_cache_dir(evaluation_file)
    sample_identity_records = _cache_dataset_sample_records(df, csv_path)
    resolved_cache_context = {
        "base_model_name": args.model if args.cw else base_model_name,
        "is_cw_model": bool(args.cw),
        "checkpoint": "" if ckpt_path is None else ckpt_path.name,
        "checkpoint_epoch": "" if ckpt_path is None else int(_extract_epoch_from_name(ckpt_path)),
        "checkpoint_file": file_fingerprint(ckpt_path),
        "dataset_csv": file_fingerprint(csv_path),
        "dataset_selection_fingerprint": dataset_selection_fingerprint(sample_identity_records),
        "default_language": default_language or "",
        "encoder_positional_mode": encoder_positional_mode,
        "transcribe_temperature": 0,
        "transcribe_simulate_stream": True,
        "transcribe_verbose": False,
        "chunk_duration_sec": float(args.chunk_size * 0.02),
    }
    evaluation_cache_key, evaluation_cache_identity = build_cache_identity(
        pre_evaluation_parameters(args),
        resolved_cache_context,
    )
    cached_run = None
    if args.no_evaluation_cache:
        print("Evaluation cache bypassed by --no_evaluation_cache; transcribe outputs will be recalculated.")
    else:
        cached_run = load_cached_run(
            cache_dir,
            evaluation_cache_key,
            expected_sample_count=len(df),
            expected_identity=evaluation_cache_identity,
        )
        if cached_run is not None:
            print(
                "EVALUATION CACHE HIT: using cached transcribe outputs "
                f"from {cache_dir / (evaluation_cache_key + '.json')}"
            )
        else:
            print(f"Evaluation cache miss for key {evaluation_cache_key[:16]}; transcribe outputs will be calculated.")

    model = None
    if cached_run is None:
        # 2. Load Model only when cached transcribe outputs are unavailable.
        model = load_streaming_model(
            name=args.model if args.cw else base_model_name,
            gran=args.chunk_size,
            multilingual=args.multilingual,
            device=args.device,
            local_ckpt_path=None if args.cw else str(ckpt_path),
            encoder_positional_mode=encoder_positional_mode,
        )
        model.eval()

    global_rwer_num, global_rwer_den = 0, 0
    global_arwer_num, global_arwer_den = 0, 0
    global_wer_i, global_wer_d, global_wer_s, global_wer_c = 0, 0, 0, 0
    global_strict_counts = {
        strict_k: {"i": 0, "d": 0, "s": 0, "c": 0}
        for strict_k in strict_k_values
    }
    global_wir_counts = {
        suffix_tolerance: {"changed_words": 0, "total_words": 0}
        for suffix_tolerance in wir_suffix_tolerances
    }
    time_bin_counts = {}

    all_chunk_latencies = []
    total_audio_duration_sec = 0.0
    total_processing_time_sec = 0.0
    predictions, references = [], []
    strict_predictions_by_k = {strict_k: [] for strict_k in strict_k_values}

    cached_samples = cached_run.get("samples", []) if cached_run is not None else []
    cache_samples_to_save = []
    chunk_duration_sec = (
        float(resolved_cache_context["chunk_duration_sec"])
        if cached_run is not None
        else model.encoder.gran * 0.02
    )
    print(f"model chunk size (s): {chunk_duration_sec}")

    # 3. Inference Loop
    print(f"Starting evaluation on {len(df)} samples...")
    for sample_index, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
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
        reference_text = _reference_text_for_sample(row, gt_words, normalizer)

        if cached_run is not None:
            cached_sample = cached_samples[sample_index]
            results = cache_records_to_results(cached_sample.get("results", []))
        else:
            results = transcribe(
                model=model,
                wav_file=wav_path,
                simulate_stream=True,
                language=row_language if row_language else ("auto" if args.multilingual else "en"),
                beam_size=args.beam_size,
                temperature=0,
                ca_kv_cache=args.ca_kv_cache,
                sa_kv_cache=args.sa_kv_cache,
                use_sliding_encoder_cache=args.use_sliding_encoder_cache,
                disable_encoder_kv_cache=args.disable_encoder_kv_cache,
                reset_decoder_on_encoder_slide=args.reset_decoder_on_encoder_slide,
                decoder_roll_overlap_seconds=args.decoder_roll_overlap_seconds,
                max_sec_context=args.max_sec_context,
                verbose=False
            )
            cache_samples_to_save.append(
                sample_cache_record(
                    sample_index=sample_index,
                    wav_path=wav_path,
                    tg_path=tg_path,
                    language=row_language,
                    audio_duration_sec=audio_duration,
                    results=results,
                )
            )
        if args.time_bin_wer:
            _accumulate_time_bin_wer(
                time_bin_counts,
                results,
                gt_words,
                audio_duration,
                chunk_duration_sec,
                args.time_bin_seconds,
                normalizer,
            )

        for step, res in enumerate(results):
            hyp_text = _normalize_for_eval(_result_text_for_eval(res), normalizer)

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

        predicted_text = _result_text_for_eval(results[-1]) if results else ""
        normalized_prediction = _normalize_for_eval(predicted_text, normalizer)
        strict_predictions_for_sample = {
            strict_k: _build_strict_word_buffer(results, normalizer, strict_k)
            for strict_k in strict_k_values
        }
        sample_wir_counts = {
            suffix_tolerance: calculate_word_instability_with_suffix_tolerance(
                results, normalizer, suffix_tolerance=suffix_tolerance
            )
            for suffix_tolerance in wir_suffix_tolerances
        }
        predictions.append(normalized_prediction)
        for strict_k, strict_prediction in strict_predictions_for_sample.items():
            strict_predictions_by_k[strict_k].append(strict_prediction)
        references.append(reference_text)
        for suffix_tolerance, (wir_changes, wir_total_words) in sample_wir_counts.items():
            global_wir_counts[suffix_tolerance]["changed_words"] += wir_changes
            global_wir_counts[suffix_tolerance]["total_words"] += wir_total_words

        # count IDS once per sample, using final hypothesis vs full reference
        i_f, d_f, s_f, c_f = calculate_idsc(reference_text, normalized_prediction)
        global_wer_i += i_f
        global_wer_d += d_f
        global_wer_s += s_f
        global_wer_c += c_f

        for strict_k, strict_prediction in strict_predictions_for_sample.items():
            i_strict, d_strict, s_strict, c_strict = calculate_idsc(reference_text, strict_prediction)
            global_strict_counts[strict_k]["i"] += i_strict
            global_strict_counts[strict_k]["d"] += d_strict
            global_strict_counts[strict_k]["s"] += s_strict
            global_strict_counts[strict_k]["c"] += c_strict

        if args.verbose:
            print("\n".join(_reference_debug_lines(wav_path, tg_path, row, gt_words, normalizer, audio_duration)))
            print("Pred: " + normalized_prediction)
            print(
                "Strict Preds: "
                + " | ".join(
                    f"k={strict_k}: {strict_predictions_for_sample[strict_k]}"
                    for strict_k in strict_k_values
                )
            )
            print("Label:" + reference_text)
            print(f"I={i_f}, D={d_f}, S={s_f}, C={c_f}")
            print(
                "WIR: "
                + ", ".join(
                    f"n={suffix_tolerance} changes={sample_wir_counts[suffix_tolerance][0]} "
                    f"total_words={sample_wir_counts[suffix_tolerance][1]}"
                    for suffix_tolerance in wir_suffix_tolerances
                )
            )
            print("-" * 30)

    evaluation_cache_used = cached_run is not None
    evaluation_cache_path = cache_dir / f"{evaluation_cache_key}.json"
    if not evaluation_cache_used:
        saved_cache_path = save_cached_run(
            cache_dir,
            evaluation_cache_key,
            evaluation_cache_identity,
            cache_samples_to_save,
        )
        evaluation_cache_path = saved_cache_path
        print(f"Evaluation cache saved: {saved_cache_path}")

    # 4. Final Aggregated Metric Calculation
    wer = jiwer.wer(references, predictions) if references else 0
    strict_wer_by_k = {
        strict_k: jiwer.wer(references, strict_predictions_by_k[strict_k]) if references else 0
        for strict_k in strict_k_values
    }
    # Keep legacy strict_* columns tied to one primary k for backward-compatible CSV consumers.
    primary_strict_k = strict_k_values[0]
    primary_strict_counts = global_strict_counts[primary_strict_k]
    strict_wer = strict_wer_by_k[primary_strict_k]
    strict_summary = _format_strict_summary(strict_wer_by_k, global_strict_counts)
    rwer = global_rwer_num / global_rwer_den if global_rwer_den > 0 else 0
    arwer = global_arwer_num / global_arwer_den if global_arwer_den > 0 else 0
    wir_stats_by_n = {}
    for suffix_tolerance, counts in global_wir_counts.items():
        total_words = counts["total_words"]
        changed_words = counts["changed_words"]
        wir_stats_by_n[suffix_tolerance] = {
            "changed_words": int(changed_words),
            "total_words": int(total_words),
            "wir": float(changed_words / total_words if total_words > 0 else 0),
        }
    wir = wir_stats_by_n[0]["wir"]
    time_bin_wer_summary = (
        _format_time_bin_wer_summary(time_bin_counts, args.time_bin_seconds)
        if args.time_bin_wer
        else ""
    )

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
        "requested_sample_count": "" if args.dataset_sample_count is None else int(args.dataset_sample_count),
        "sample_count": int(len(df)),
        "chunk_size": int(args.chunk_size),
        "chunk_duration_sec": float(chunk_duration_sec),
        "max_sec_context": int(args.max_sec_context),
        "beam_size": int(args.beam_size),
        "strict_k": int(primary_strict_k),
        "strict_k_values": " ".join(str(k) for k in strict_k_values),
        "strict_summary": strict_summary,
        "language": default_language or "",
        "multilingual": bool(args.multilingual),
        "encoder_positional_mode": encoder_positional_mode,
        "device": args.device,
        "sa_kv_cache": bool(args.sa_kv_cache),
        "ca_kv_cache": bool(args.ca_kv_cache),
        "use_sliding_encoder_cache": bool(args.use_sliding_encoder_cache),
        "disable_encoder_kv_cache": bool(args.disable_encoder_kv_cache),
        "reset_decoder_on_encoder_slide": bool(args.reset_decoder_on_encoder_slide),
        "decoder_roll_overlap_seconds": float(args.decoder_roll_overlap_seconds),
        "wer": float(wer),
        "strict_wer": float(strict_wer),
        "rwer": float(rwer),
        "arwer": float(arwer),
        "wir": float(wir),
        "wir_changed_words": int(wir_stats_by_n[0]["changed_words"]),
        "wir_total_words": int(wir_stats_by_n[0]["total_words"]),
        "wir_n_values": " ".join(str(n) for n in wir_suffix_tolerances),
        "wir_summary": _format_wir_summary(wir_stats_by_n),
        "time_bin_wer_enabled": bool(args.time_bin_wer),
        "time_bin_seconds": float(args.time_bin_seconds),
        "time_bin_wer_summary": time_bin_wer_summary,
        "wer_insertions": int(global_wer_i),
        "wer_deletions": int(global_wer_d),
        "wer_substitutions": int(global_wer_s),
        "wer_correct": int(global_wer_c),
        "strict_wer_insertions": int(primary_strict_counts["i"]),
        "strict_wer_deletions": int(primary_strict_counts["d"]),
        "strict_wer_substitutions": int(primary_strict_counts["s"]),
        "strict_wer_correct": int(primary_strict_counts["c"]),
        "avg_latency_ms": float(avg_latency * 1000),
        "rtf": float(rtf),
        "total_audio_duration_sec": float(total_audio_duration_sec),
        "total_processing_time_sec": float(total_processing_time_sec),
        "evaluation_cache_used": bool(evaluation_cache_used),
        "evaluation_cache_key": evaluation_cache_key,
        "evaluation_cache_path": str(evaluation_cache_path),
    }

    append_evaluation_row(evaluation_file, stats)
    print(f"Stats saved to: {evaluation_file}")
    if args.time_bin_wer:
        print()
        print("=== Time-Binned Interval WER ===")
        print(time_bin_wer_summary.replace(" | ", "\n") if time_bin_wer_summary else "No time-bin WER entries collected.")
    print()
    print_latest_rows(evaluation_file, row_count=1)


if __name__ == "__main__":
    evaluate()
