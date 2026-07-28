import argparse
import difflib
import json
import os
import re
import time
from types import SimpleNamespace
import pandas as pd
import torch
import jiwer
from tqdm import tqdm
from praatio import textgrid
from pathlib import Path
import librosa
import numpy as np

from careless_whisper_stream import load_model, load_streaming_model
from careless_whisper_stream.normalizers import (
    BasicTextNormalizer,
    EnglishTextNormalizer,
    GermanTextNormalizer,
)
from careless_whisper_stream.streaming_decoding import encoder_cache_diagnostics_summary_lines
from careless_whisper_stream.streaming_transcribe import transcribe as streaming_transcribe
from careless_whisper_stream.transcribe import transcribe as offline_transcribe
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


def _offline_model_fingerprint(model_name: str) -> dict:
    model_path = Path(str(model_name)).expanduser()
    if model_path.is_file():
        return file_fingerprint(model_path)
    return {}


def _offline_model_name_or_path(model_name: str) -> str:
    model_path = Path(str(model_name)).expanduser()
    if model_path.is_file():
        return str(model_path)
    return str(model_name)


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
    Count word-level edits needed to revise previously emitted hypotheses.
    """
    return calculate_word_instability_with_suffix_tolerance(
        results, normalizer, suffix_tolerance=0
    )


def _count_alignment_word_instability(previous_words, current_words, countable_previous_len=None):
    """
    Count revisions between two hypotheses using word alignment opcodes.

    This intentionally treats pure append-at-end insertions as stable streaming
    growth, while counting insertions inside an already emitted transcript as
    revisions. That matches the WIR requirement that "Deck is blue" -> "the car
    is blue" counts both "Deck" -> "the" and the inserted "car", but "the car
    is" -> "the car is blue" does not count the newly appended "blue".
    """
    previous_len = len(previous_words)
    if countable_previous_len is None:
        countable_previous_len = previous_len
    countable_previous_len = max(0, min(int(countable_previous_len), previous_len))

    matcher = difflib.SequenceMatcher(
        a=previous_words,
        b=current_words,
        autojunk=False,
    )

    changed_word_count = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        previous_span_len = i2 - i1
        current_span_len = j2 - j1
        countable_previous_span_len = max(
            0,
            min(i2, countable_previous_len) - min(i1, countable_previous_len),
        )

        if tag == "equal":
            continue
        if tag == "insert":
            if i1 < countable_previous_len:
                changed_word_count += current_span_len
            continue
        if tag == "delete":
            changed_word_count += countable_previous_span_len
            continue
        if tag == "replace":
            inserted_word_count = max(0, current_span_len - previous_span_len)
            if i1 >= countable_previous_len:
                inserted_word_count = 0
            changed_word_count += countable_previous_span_len + inserted_word_count
            continue

        raise ValueError(f"Unexpected alignment opcode: {tag}")

    return changed_word_count


def calculate_word_instability_with_suffix_tolerance(results, normalizer, suffix_tolerance: int = 0):
    """
    Count revised words with alignment-aware insertion handling.

    Changes inside the trailing `suffix_tolerance` words of the previous
    hypothesis are ignored by aligning only the countable previous prefix.
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
        countable_previous_len = max(0, len(previous_words) - suffix_tolerance)
        changed_word_count += _count_alignment_word_instability(
            previous_words,
            current_words,
            countable_previous_len=countable_previous_len,
        )
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


PREFIX_WER_SECONDS = tuple(range(10, 61, 10))
DELAY_N_MAX_WAIT_SECONDS = 1.0


def _format_prefix_wer_summary(prefix_wer_counts: dict[int, dict[str, int]]) -> str:
    summary_parts = []
    for prefix_seconds in sorted(prefix_wer_counts):
        counts = prefix_wer_counts[prefix_seconds]
        denominator = counts["c"] + counts["d"] + counts["s"]
        errors = counts["i"] + counts["d"] + counts["s"]
        wer_text = f"{(errors / denominator) * 100:.2f}%" if denominator > 0 else "N/A"
        summary_parts.append(
            f"{prefix_seconds}s prefix: {wer_text} "
            f"(samples={counts['samples']}, I/D/S/C={counts['i']}/{counts['d']}/{counts['s']}/{counts['c']})"
        )
    return " | ".join(summary_parts)


def _offline_whisper_transcribe_result(model, wav_path: str, row_language: str | None, beam_size: int):
    # Offline Whisper produces one final transcript per sample; wrap it like a
    # one-item result list so evaluation caching can stay shared with streaming.
    start_time = time.perf_counter()
    result = offline_transcribe(
        model,
        wav_path,
        language=row_language,
        beam_size=beam_size,
        temperature=0,
        # Offline Whisper shows an internal progress bar when verbose=False.
        # Use None so evaluation keeps only the outer per-dataset tqdm.
        verbose=None,
    )
    processing_time = time.perf_counter() - start_time
    text = str(result.get("text", "") or "").strip()
    return SimpleNamespace(
        text=text,
        full_text=text,
        processing_time=processing_time,
        language=result.get("language", row_language or ""),
    )


def _hypothesis_at_or_before(results, time_sec: float, chunk_duration_sec: float, normalizer) -> str:
    if not results or time_sec <= 0:
        return ""

    result_index = int((time_sec / chunk_duration_sec) + 1e-9) - 1
    if result_index < 0:
        return ""

    result_index = min(result_index, len(results) - 1)
    return _normalize_for_eval(_result_text_for_eval(results[result_index]), normalizer)


def _accumulate_prefix_wer(
    prefix_wer_counts: dict[int, dict[str, int]],
    results,
    gt_words,
    audio_duration: float,
    chunk_duration_sec: float,
    normalizer,
) -> None:
    if not results:
        return

    for prefix_seconds in PREFIX_WER_SECONDS:
        # Requirement: prefix WER should compare cumulative hypotheses at fixed
        # 10s checkpoints. Shorter samples do not contribute to later prefixes.
        if audio_duration < prefix_seconds:
            continue

        reference_text = _normalize_for_eval(
            get_gt_prefix_at_time(gt_words, prefix_seconds),
            normalizer,
        )
        hypothesis_text = _hypothesis_at_or_before(
            results,
            prefix_seconds,
            chunk_duration_sec,
            normalizer,
        )

        i, d, s, c = calculate_idsc(reference_text, hypothesis_text)
        counts = prefix_wer_counts.setdefault(
            prefix_seconds,
            {"i": 0, "d": 0, "s": 0, "c": 0, "samples": 0},
        )
        counts["i"] += i
        counts["d"] += d
        counts["s"] += s
        counts["c"] += c
        counts["samples"] += 1


def _delay_n_result_words(result, normalizer) -> list[str]:
    return _normalize_for_eval(_result_text_for_eval(result), normalizer).split()


def calculate_delay_n_display_stats(
    results,
    audio_duration: float,
    chunk_duration_sec: float,
    normalizer,
    max_wait_seconds: float = DELAY_N_MAX_WAIT_SECONDS,
) -> dict[str, float | int]:
    """
    Simulate a display policy that withholds the newest trailing word.

    Requirement: the visual layer delays the last word until the next word is
    appended to the cumulative result, but emits it after at most one second so
    silence does not leave the UI looking stuck.
    """
    if not results or audio_duration <= 0 or chunk_duration_sec <= 0:
        return {
            "perceived_processing_time_sec": 0.0,
            "rtf": 0.0,
            "latency_sum_sec": 0.0,
            "emitted_words": 0,
        }

    max_wait_seconds = max(0.0, float(max_wait_seconds))
    emitted_count = 0
    emitted_words = 0
    latency_sum_sec = 0.0
    total_processing_time_sec = 0.0
    last_available_time_sec = 0.0
    pending = None

    def emit_word(display_time_sec: float, first_seen_audio_time_sec: float) -> None:
        nonlocal emitted_words, latency_sum_sec
        emitted_words += 1
        latency_sum_sec += max(0.0, display_time_sec - first_seen_audio_time_sec)

    for step, result in enumerate(results):
        processing_time_sec = float(getattr(result, "processing_time", 0.0) or 0.0)
        total_processing_time_sec += processing_time_sec
        audio_time_sec = min(audio_duration, (step + 1) * chunk_duration_sec)
        available_time_sec = audio_time_sec + processing_time_sec
        last_available_time_sec = available_time_sec
        current_words = _delay_n_result_words(result, normalizer)

        if pending is not None and pending["deadline_sec"] <= available_time_sec:
            emit_word(pending["deadline_sec"], pending["first_seen_audio_time_sec"])
            emitted_count = max(emitted_count, pending["word_index"] + 1)
            pending = None

        if pending is not None and len(current_words) > pending["word_index"] + 1:
            emit_word(available_time_sec, pending["first_seen_audio_time_sec"])
            emitted_count = max(emitted_count, pending["word_index"] + 1)
            pending = None

        if len(current_words) <= emitted_count:
            continue

        appended_count = len(current_words) - emitted_count
        immediate_count = max(0, appended_count - 1)
        for _ in range(immediate_count):
            emit_word(available_time_sec, audio_time_sec)
            emitted_count += 1

        pending = {
            "word_index": len(current_words) - 1,
            "first_seen_audio_time_sec": audio_time_sec,
            "deadline_sec": available_time_sec + max_wait_seconds,
        }

    final_display_time_sec = last_available_time_sec
    if pending is not None:
        final_display_time_sec = max(final_display_time_sec, pending["deadline_sec"])
        emit_word(pending["deadline_sec"], pending["first_seen_audio_time_sec"])

    final_visual_hold_sec = max(0.0, final_display_time_sec - last_available_time_sec)
    perceived_processing_time_sec = total_processing_time_sec + final_visual_hold_sec
    return {
        "perceived_processing_time_sec": float(perceived_processing_time_sec),
        "rtf": float(perceived_processing_time_sec / audio_duration),
        "latency_sum_sec": float(latency_sum_sec),
        "emitted_words": int(emitted_words),
    }


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate CarelessWhisper WER on a dataset")

    # When adding evaluation parameters, classify them in evaluation_caching.py
    # and run: python -m unittest tests.test_evaluation_caching_contract
    # Model Setup
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model run name under ckpt_root, or an offline Whisper model name/path when --offline_whisper is used.",
    )
    parser.add_argument("--offline_whisper", action="store_true", help="Evaluate a non-streaming Whisper model; streaming-only metrics are not measured.")
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
    parser.add_argument("--encoder_cache_diagnostics", action="store_true", help="Print encoder cache vs recomputed-reference diff stats during transcription.")
    parser.add_argument("--encoder_cache_diagnostic_interval", type=int, default=1, help="Print encoder cache diagnostics every N decode chunks.")
    parser.add_argument("--reset_decoder_on_encoder_slide", action="store_true", help="Roll decoder prefix tokens into prompt as sliding encoder cache prunes old audio.")
    parser.add_argument("--decoder_roll_overlap_seconds", type=float, default=5.0, help="Seconds of retained encoder audio kept as overlap before the active decoder prefix during rolling decoder reset.")
    parser.add_argument("--decoder_roll_min_interval_seconds", type=float, default=2.0, help="Minimum seconds between decoder prefix rolls.")
    parser.add_argument("--decoder_roll_max_prefix_tokens", type=int, default=48, help="Maximum BPE tokens kept as active decoder prefix after a roll.")
    parser.add_argument("--decoder_token_time_lag_seconds", type=float, default=2.0, help="Seconds subtracted from first-seen token time estimates for decoder rolling.")
    parser.add_argument("--decoder_roll_diagnostics", action="store_true", help="Print decoder roll event and prefix/generated overlap diagnostics during transcription.")
    parser.add_argument("--prefix_wer", action="store_true", help="Print and save cumulative WER at 10s, 20s, 30s, 40s, 50s, and 60s audio prefixes.")
    parser.add_argument("--delay_n_rtf", action="store_true", help="Report perceived RTF and latency when visually delaying the newest trailing word until another word is appended, with a 1s timeout.")
    parser.add_argument("-verbose", action="store_true", help="Prints additional info while evaluating")
    parser.add_argument("-cw", action="store_true", help="Uses a CW whisper base model instead of a local model.")
    parser.add_argument("--force_hf_download", action="store_true", help="When used with -cw, force Hugging Face to download the CW model instead of reusing the local HF cache.")
    parser.add_argument("--no_evaluation_cache", action="store_true", help="Always recalculate transcribe outputs instead of reading the evaluation cache.")

    # Dataset Setup
    parser.add_argument("--dataset_name", type=str, required=True, help="Key from ds_paths in ds_dict.py")

    args = parser.parse_args()
    validate_parameter_classification(list(vars(args).keys()))

    if args.dataset_sample_count is not None and args.dataset_fraction != 1.0:
        raise ValueError("--dataset_sample_count cannot be used together with --dataset_fraction.")
    if args.dataset_sample_count is not None and args.dataset_sample_count <= 0:
        raise ValueError("--dataset_sample_count must be a positive integer.")
    if args.reset_decoder_on_encoder_slide and not args.use_sliding_encoder_cache:
        raise ValueError("--reset_decoder_on_encoder_slide requires --use_sliding_encoder_cache.")
    if args.encoder_cache_diagnostic_interval <= 0:
        raise ValueError("--encoder_cache_diagnostic_interval must be positive.")
    if args.decoder_roll_overlap_seconds < 0:
        raise ValueError("--decoder_roll_overlap_seconds must be non-negative.")
    if args.decoder_roll_overlap_seconds >= args.max_sec_context:
        raise ValueError("--decoder_roll_overlap_seconds must be smaller than --max_sec_context.")
    if args.decoder_roll_min_interval_seconds < 0:
        raise ValueError("--decoder_roll_min_interval_seconds must be non-negative.")
    if args.decoder_roll_max_prefix_tokens <= 0:
        raise ValueError("--decoder_roll_max_prefix_tokens must be positive.")
    if args.decoder_token_time_lag_seconds < 0:
        raise ValueError("--decoder_token_time_lag_seconds must be non-negative.")
    if args.force_hf_download and not args.cw:
        raise ValueError("--force_hf_download can only be used with -cw.")
    if args.offline_whisper and args.cw:
        raise ValueError("--offline_whisper cannot be combined with -cw; pass a Whisper model name/path via --model.")
    if args.offline_whisper and args.force_hf_download:
        raise ValueError("--force_hf_download is only supported for CarelessWhisper streaming HF checkpoints.")
    if args.offline_whisper and args.checkpoint is not None:
        raise ValueError("--checkpoint selects CarelessWhisper training checkpoints and cannot be used with --offline_whisper.")
    if args.offline_whisper and args.prefix_wer:
        raise ValueError("--prefix_wer is streaming-only and cannot be used with --offline_whisper.")
    if args.offline_whisper and args.delay_n_rtf:
        raise ValueError("--delay_n_rtf is streaming-only and cannot be used with --offline_whisper.")
    if args.offline_whisper and args.wir_n:
        raise ValueError("--wir_n is streaming-only and cannot be used with --offline_whisper.")
    offline_incompatible_flags = [
        ("-sa_kv_cache", args.sa_kv_cache),
        ("-ca_kv_cache", args.ca_kv_cache),
        ("--use_sliding_encoder_cache", args.use_sliding_encoder_cache),
        ("--disable_encoder_kv_cache", args.disable_encoder_kv_cache),
        ("--encoder_cache_diagnostics", args.encoder_cache_diagnostics),
        ("--reset_decoder_on_encoder_slide", args.reset_decoder_on_encoder_slide),
        ("--decoder_roll_diagnostics", args.decoder_roll_diagnostics),
    ]
    for flag_name, enabled in offline_incompatible_flags:
        if enabled:
            raise ValueError(f"{flag_name} is streaming-only and cannot be used with --offline_whisper.")

    evaluation_mode = "offline_whisper" if args.offline_whisper else "streaming"

    if args.offline_whisper:
        print(f"Using offline Whisper model: {args.model}")
        ckpt_path = None
        hparams = {}
        base_model_name = args.model
    elif not args.cw:
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

    encoder_positional_mode = (
        ""
        if args.offline_whisper
        else _resolve_encoder_positional_mode(
            args.encoder_positional_mode,
            args.model,
            hparams,
        )
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
    print(f"Evaluation mode: {evaluation_mode}")
    if not args.offline_whisper:
        print(f"Encoder positional mode: {encoder_positional_mode}")
        print(f"Strict correction distances: {strict_k_values}")
        print(f"WIR suffix tolerances: {wir_suffix_tolerances}")
    else:
        print("Streaming-only metrics disabled: strict/SWER, RWER, ARWER, WIR, prefix WER, and delay-n RTF.")

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
        "evaluation_mode": evaluation_mode,
        "base_model_name": args.model if args.cw else base_model_name,
        "is_cw_model": bool(args.cw),
        "checkpoint": "" if ckpt_path is None else ckpt_path.name,
        "checkpoint_epoch": "" if ckpt_path is None else int(_extract_epoch_from_name(ckpt_path)),
        "checkpoint_file": _offline_model_fingerprint(args.model) if args.offline_whisper else file_fingerprint(ckpt_path),
        "dataset_csv": file_fingerprint(csv_path),
        "dataset_selection_fingerprint": dataset_selection_fingerprint(sample_identity_records),
        "default_language": default_language or "",
        "encoder_positional_mode": encoder_positional_mode,
        "transcribe_temperature": 0,
        "transcribe_simulate_stream": not args.offline_whisper,
        "transcribe_verbose": False,
        "chunk_duration_sec": 0.0 if args.offline_whisper else float(args.chunk_size * 0.02),
    }
    evaluation_cache_key, evaluation_cache_identity = build_cache_identity(
        pre_evaluation_parameters(args),
        resolved_cache_context,
    )
    cached_run = None
    if args.no_evaluation_cache:
        print("Evaluation cache bypassed by --no_evaluation_cache; transcribe outputs will be recalculated.")
    elif args.force_hf_download:
        # Requirement: --force_hf_download must actually reload the CW model from
        # Hugging Face. A cached evaluation run would skip model loading entirely.
        print("Evaluation cache bypassed by --force_hf_download; CW model will be freshly downloaded from Hugging Face.")
    elif args.encoder_cache_diagnostics:
        print("Evaluation cache bypassed by --encoder_cache_diagnostics; transcribe outputs will be recalculated for diagnostic logging.")
    elif args.decoder_roll_diagnostics:
        print("Evaluation cache bypassed by --decoder_roll_diagnostics; transcribe outputs will be recalculated for diagnostic logging.")
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
        if args.offline_whisper:
            model = load_model(_offline_model_name_or_path(args.model), device=args.device)
        else:
            model = load_streaming_model(
                name=args.model if args.cw else base_model_name,
                gran=args.chunk_size,
                multilingual=args.multilingual,
                device=args.device,
                local_ckpt_path=None if args.cw else str(ckpt_path),
                encoder_positional_mode=encoder_positional_mode,
                force_hf_download=args.force_hf_download,
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
    prefix_wer_counts = {}
    delay_n_sample_rtfs = []
    delay_n_latency_sum_sec = 0.0
    delay_n_emitted_words = 0
    delay_n_total_perceived_processing_time_sec = 0.0

    all_chunk_latencies = []
    total_audio_duration_sec = 0.0
    total_processing_time_sec = 0.0
    predictions, references = [], []
    strict_predictions_by_k = {strict_k: [] for strict_k in strict_k_values}
    encoder_cache_diagnostic_samples = []

    cached_samples = cached_run.get("samples", []) if cached_run is not None else []
    cache_samples_to_save = []
    chunk_duration_sec = (
        0.0
        if args.offline_whisper
        else (
            float(resolved_cache_context["chunk_duration_sec"])
            if cached_run is not None
            else model.encoder.gran * 0.02
        )
    )
    if args.offline_whisper:
        print("model chunk size (s): N/A (offline Whisper)")
    else:
        print(f"model chunk size (s): {chunk_duration_sec}")

    # 3. Inference Loop
    print(f"Starting evaluation on {len(df)} samples...")
    for sample_index, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        wav_path = _resolve_csv_relative_path(csv_path, row["wav_path"])
        tg_path = _resolve_csv_relative_path(csv_path, row["tg_path"])
        row_language = (
            _canonicalize_language(row["lang"])
            if (args.multilingual or args.offline_whisper) and "lang" in row and pd.notna(row["lang"])
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
            if args.offline_whisper and results:
                p_latency = getattr(results[-1], "processing_time", 0.0)
                all_chunk_latencies.append(p_latency)
                total_processing_time_sec += p_latency
        elif args.offline_whisper:
            result = _offline_whisper_transcribe_result(
                model=model,
                wav_path=wav_path,
                row_language=row_language,
                beam_size=args.beam_size,
            )
            results = [result]
            all_chunk_latencies.append(result.processing_time)
            total_processing_time_sec += result.processing_time
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
        else:
            results = streaming_transcribe(
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
                encoder_cache_diagnostics=args.encoder_cache_diagnostics,
                encoder_cache_diagnostic_interval=args.encoder_cache_diagnostic_interval,
                print_encoder_cache_diagnostics_summary=False,
                reset_decoder_on_encoder_slide=args.reset_decoder_on_encoder_slide,
                decoder_roll_overlap_seconds=args.decoder_roll_overlap_seconds,
                decoder_roll_min_interval_seconds=args.decoder_roll_min_interval_seconds,
                decoder_roll_max_prefix_tokens=args.decoder_roll_max_prefix_tokens,
                decoder_token_time_lag_seconds=args.decoder_token_time_lag_seconds,
                decoder_roll_diagnostics=args.decoder_roll_diagnostics,
                max_sec_context=args.max_sec_context,
                verbose=False
            )
            if args.encoder_cache_diagnostics:
                encoder_cache_diagnostic_samples.extend(
                    getattr(model, "last_encoder_cache_diagnostic_samples", [])
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
        if args.prefix_wer:
            _accumulate_prefix_wer(
                prefix_wer_counts,
                results,
                gt_words,
                audio_duration,
                chunk_duration_sec,
                normalizer,
            )
        if args.delay_n_rtf:
            delay_n_stats = calculate_delay_n_display_stats(
                results,
                audio_duration,
                chunk_duration_sec,
                normalizer,
            )
            delay_n_sample_rtfs.append(float(delay_n_stats["rtf"]))
            delay_n_latency_sum_sec += float(delay_n_stats["latency_sum_sec"])
            delay_n_emitted_words += int(delay_n_stats["emitted_words"])
            delay_n_total_perceived_processing_time_sec += float(
                delay_n_stats["perceived_processing_time_sec"]
            )

        if not args.offline_whisper:
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
        strict_predictions_for_sample = {}
        sample_wir_counts = {}
        if not args.offline_whisper:
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

        if not args.offline_whisper:
            for strict_k, strict_prediction in strict_predictions_for_sample.items():
                i_strict, d_strict, s_strict, c_strict = calculate_idsc(reference_text, strict_prediction)
                global_strict_counts[strict_k]["i"] += i_strict
                global_strict_counts[strict_k]["d"] += d_strict
                global_strict_counts[strict_k]["s"] += s_strict
                global_strict_counts[strict_k]["c"] += c_strict

        if args.verbose:
            print("\n".join(_reference_debug_lines(wav_path, tg_path, row, gt_words, normalizer, audio_duration)))
            print("Pred: " + normalized_prediction)
            if not args.offline_whisper:
                print(
                    "Strict Preds: "
                    + " | ".join(
                        f"k={strict_k}: {strict_predictions_for_sample[strict_k]}"
                        for strict_k in strict_k_values
                    )
                )
            print("Label:" + reference_text)
            print(f"I={i_f}, D={d_f}, S={s_f}, C={c_f}")
            if not args.offline_whisper:
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
    if args.offline_whisper:
        # Offline baselines have no incremental hypotheses, so SWER/strict WER,
        # RWER, ARWER, and WIR are intentionally recorded as blank CSV fields.
        strict_wer_by_k = {}
        primary_strict_k = None
        primary_strict_counts = {"i": None, "d": None, "s": None, "c": None}
        strict_wer = None
        strict_summary = ""
        rwer = None
        arwer = None
        wir_stats_by_n = {}
        wir = None
    else:
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
    prefix_wer_summary = (
        _format_prefix_wer_summary(prefix_wer_counts)
        if args.prefix_wer
        else ""
    )
    delay_n_rtf = (
        float(np.mean(delay_n_sample_rtfs))
        if args.delay_n_rtf and delay_n_sample_rtfs
        else None
    )
    delay_n_weighted_rtf = (
        float(delay_n_total_perceived_processing_time_sec / total_audio_duration_sec)
        if args.delay_n_rtf and total_audio_duration_sec > 0
        else None
    )
    delay_n_avg_latency_ms = (
        float((delay_n_latency_sum_sec / delay_n_emitted_words) * 1000)
        if args.delay_n_rtf and delay_n_emitted_words > 0
        else None
    )

    avg_latency = np.mean(all_chunk_latencies) if all_chunk_latencies else 0
    rtf = total_processing_time_sec / total_audio_duration_sec if total_audio_duration_sec > 0 else 0

    stats = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_file": evaluation_file,
        "evaluation_mode": evaluation_mode,
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
        "chunk_size": "" if args.offline_whisper else int(args.chunk_size),
        "chunk_duration_sec": "" if args.offline_whisper else float(chunk_duration_sec),
        "max_sec_context": "" if args.offline_whisper else int(args.max_sec_context),
        "beam_size": int(args.beam_size),
        "strict_k": primary_strict_k,
        "strict_k_values": "" if args.offline_whisper else " ".join(str(k) for k in strict_k_values),
        "strict_summary": strict_summary,
        "language": default_language or "",
        "multilingual": bool(args.multilingual),
        "encoder_positional_mode": encoder_positional_mode,
        "device": args.device,
        "sa_kv_cache": "" if args.offline_whisper else bool(args.sa_kv_cache),
        "ca_kv_cache": "" if args.offline_whisper else bool(args.ca_kv_cache),
        "use_sliding_encoder_cache": "" if args.offline_whisper else bool(args.use_sliding_encoder_cache),
        "disable_encoder_kv_cache": "" if args.offline_whisper else bool(args.disable_encoder_kv_cache),
        "encoder_cache_diagnostics": "" if args.offline_whisper else bool(args.encoder_cache_diagnostics),
        "encoder_cache_diagnostic_interval": "" if args.offline_whisper else int(args.encoder_cache_diagnostic_interval),
        "reset_decoder_on_encoder_slide": "" if args.offline_whisper else bool(args.reset_decoder_on_encoder_slide),
        "decoder_roll_overlap_seconds": "" if args.offline_whisper else float(args.decoder_roll_overlap_seconds),
        "decoder_roll_min_interval_seconds": "" if args.offline_whisper else float(args.decoder_roll_min_interval_seconds),
        "decoder_roll_max_prefix_tokens": "" if args.offline_whisper else int(args.decoder_roll_max_prefix_tokens),
        "decoder_token_time_lag_seconds": "" if args.offline_whisper else float(args.decoder_token_time_lag_seconds),
        "wer": float(wer),
        "strict_wer": strict_wer,
        "rwer": rwer,
        "arwer": arwer,
        "wir": wir,
        "wir_changed_words": "" if args.offline_whisper else int(wir_stats_by_n[0]["changed_words"]),
        "wir_total_words": "" if args.offline_whisper else int(wir_stats_by_n[0]["total_words"]),
        "wir_n_values": "" if args.offline_whisper else " ".join(str(n) for n in wir_suffix_tolerances),
        "wir_summary": "" if args.offline_whisper else _format_wir_summary(wir_stats_by_n),
        "prefix_wer_enabled": bool(args.prefix_wer),
        "prefix_wer_seconds": " ".join(str(seconds) for seconds in PREFIX_WER_SECONDS),
        "prefix_wer_summary": prefix_wer_summary,
        "delay_n_rtf_enabled": bool(args.delay_n_rtf),
        "delay_n_rtf": delay_n_rtf,
        "delay_n_weighted_rtf": delay_n_weighted_rtf,
        "delay_n_avg_latency_ms": delay_n_avg_latency_ms,
        "delay_n_emitted_words": "" if delay_n_emitted_words == 0 else int(delay_n_emitted_words),
        "delay_n_max_wait_seconds": DELAY_N_MAX_WAIT_SECONDS,
        "wer_insertions": int(global_wer_i),
        "wer_deletions": int(global_wer_d),
        "wer_substitutions": int(global_wer_s),
        "wer_correct": int(global_wer_c),
        "strict_wer_insertions": primary_strict_counts["i"],
        "strict_wer_deletions": primary_strict_counts["d"],
        "strict_wer_substitutions": primary_strict_counts["s"],
        "strict_wer_correct": primary_strict_counts["c"],
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
    if args.prefix_wer:
        print()
        print("=== Prefix WER ===")
        print(prefix_wer_summary.replace(" | ", "\n") if prefix_wer_summary else "No prefix WER entries collected.")
    if args.delay_n_rtf:
        print()
        print("=== Delay-N Visual Emission ===")
        print(f"Avg sample RTF: {delay_n_rtf:.4f}" if delay_n_rtf is not None else "Avg sample RTF: N/A")
        print(f"Weighted RTF:   {delay_n_weighted_rtf:.4f}" if delay_n_weighted_rtf is not None else "Weighted RTF:   N/A")
        print(f"Avg latency:    {delay_n_avg_latency_ms:.1f} ms" if delay_n_avg_latency_ms is not None else "Avg latency:    N/A")
    if args.encoder_cache_diagnostics:
        print()
        print("=== Encoder Cache Diagnostics Summary ===")
        for line in encoder_cache_diagnostics_summary_lines(encoder_cache_diagnostic_samples):
            print(line)
    print()
    print_latest_rows(evaluation_file, row_count=1)


if __name__ == "__main__":
    evaluate()
