import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


EVALUATION_CACHE_VERSION = 5
MAX_EVALUATION_CACHE_RUNS = 5
DEFAULT_EVALUATION_FILE = Path(os.environ.get("HOME", str(Path.home()))) / "ma" / "data" / "evaluation.csv"

# Parameters are grouped by whether they can affect model loading, dataset
# selection, or transcribe(...). Keeping this list explicit makes future eval
# flags safer: every argparse option must be classified before caching runs.
PRE_EVALUATION_PARAMETER_NAMES = [
    "model",
    "checkpoint",
    "chunk_size",
    "multilingual",
    "device",
    "dataset_fraction",
    "dataset_sample_count",
    "dataset_partition",
    "beam_size",
    "max_sec_context",
    "lang",
    "encoder_positional_mode",
    "sa_kv_cache",
    "ca_kv_cache",
    "use_sliding_encoder_cache",
    "disable_encoder_kv_cache",
    "reset_decoder_on_encoder_slide",
    "decoder_roll_overlap_seconds",
    "cw",
    "dataset_name",
]
EVALUATION_ONLY_PARAMETER_NAMES = [
    "strict_k",
    "wir_n",
    "time_bin_wer",
    "time_bin_seconds",
    "verbose",
]
CACHE_CONTROL_PARAMETER_NAMES = ["no_evaluation_cache"]

TRANSCRIBE_RESULT_CACHE_FIELDS = [
    "text",
    "full_text",
    "processing_time",
    "language",
    "language_probs",
    "tokens",
    "retired_tokens",
    "avg_logprob",
    "no_speech_prob",
    "temperature",
    "compression_ratio",
    "timed_tokens",
    "timed_text",
]


def validate_parameter_classification(arg_names: list[str]) -> None:
    classified = set(PRE_EVALUATION_PARAMETER_NAMES)
    classified.update(EVALUATION_ONLY_PARAMETER_NAMES)
    classified.update(CACHE_CONTROL_PARAMETER_NAMES)

    unclassified = sorted(set(arg_names) - classified)
    if unclassified:
        raise ValueError(
            "Evaluation cache parameter classification is incomplete. "
            f"Classify these argparse options in evaluation_caching.py: {unclassified}"
        )


def parameter_classification_summary() -> str:
    return (
        "pre-evaluation/cache key: "
        + ", ".join(PRE_EVALUATION_PARAMETER_NAMES)
        + "\n"
        + "evaluation-only/rescore: "
        + ", ".join(EVALUATION_ONLY_PARAMETER_NAMES)
        + "\n"
        + "cache-control: "
        + ", ".join(CACHE_CONTROL_PARAMETER_NAMES)
    )


def evaluation_cache_dir(evaluation_file: str | Path) -> Path:
    return Path(evaluation_file).expanduser().parent / "evaluation_cache"


def clear_evaluation_cache(evaluation_file: str | Path = DEFAULT_EVALUATION_FILE) -> Path:
    cache_dir = evaluation_cache_dir(evaluation_file)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    return cache_dir


def pre_evaluation_parameters(args) -> dict[str, Any]:
    args_dict = vars(args)
    return {
        name: _json_safe(args_dict.get(name))
        for name in PRE_EVALUATION_PARAMETER_NAMES
    }


def file_fingerprint(path_value: str | Path | None) -> dict[str, Any]:
    if path_value is None:
        return {}

    path = Path(path_value)
    if not path.exists():
        return {"path": str(path)}

    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def stable_hash(value: Any) -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_cache_identity(pre_params: dict[str, Any], resolved_context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    identity = {
        "cache_version": EVALUATION_CACHE_VERSION,
        "pre_evaluation_parameters": _json_safe(pre_params),
        "resolved_context": _json_safe(resolved_context),
    }
    return stable_hash(identity), identity


def cache_path_for_key(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"{cache_key}.json"


def load_cached_run(
    cache_dir: Path,
    cache_key: str,
    expected_sample_count: int,
    expected_identity: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    cache_path = cache_path_for_key(cache_dir, cache_key)
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            cache_data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: evaluation cache file could not be read and will be ignored: {cache_path} ({exc})")
        return None

    if cache_data.get("cache_version") != EVALUATION_CACHE_VERSION:
        return None
    if cache_data.get("cache_key") != cache_key:
        return None
    if expected_identity is not None and cache_data.get("identity") != _json_safe(expected_identity):
        return None
    if len(cache_data.get("samples", [])) != expected_sample_count:
        return None

    cache_data["last_used_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _atomic_write_json(cache_path, cache_data)
    return cache_data


def save_cached_run(
    cache_dir: Path,
    cache_key: str,
    cache_identity: dict[str, Any],
    samples: list[dict[str, Any]],
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "cache_version": EVALUATION_CACHE_VERSION,
        "cache_key": cache_key,
        "created_at": now,
        "last_used_at": now,
        "identity": _json_safe(cache_identity),
        "sample_count": len(samples),
        "samples": _json_safe(samples),
    }

    cache_path = cache_path_for_key(cache_dir, cache_key)
    _atomic_write_json(cache_path, payload)
    prune_old_caches(cache_dir)
    return cache_path


def prune_old_caches(cache_dir: Path, keep: int = MAX_EVALUATION_CACHE_RUNS) -> None:
    if keep <= 0 or not cache_dir.exists():
        return

    cache_files = sorted(
        [path for path in cache_dir.glob("*.json") if path.is_file()],
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for stale_path in cache_files[keep:]:
        try:
            stale_path.unlink()
        except OSError as exc:
            print(f"Warning: could not prune old evaluation cache {stale_path}: {exc}")


def result_to_cache_record(result) -> dict[str, Any]:
    record = {}
    for field_name in TRANSCRIBE_RESULT_CACHE_FIELDS:
        if hasattr(result, field_name):
            record[field_name] = _json_safe(getattr(result, field_name))
    return record


def cache_records_to_results(records: list[dict[str, Any]]) -> list[SimpleNamespace]:
    return [SimpleNamespace(**record) for record in records]


def sample_cache_record(
    sample_index: int,
    wav_path: str,
    tg_path: str,
    language: str | None,
    audio_duration_sec: float,
    results,
) -> dict[str, Any]:
    return {
        "sample_index": int(sample_index),
        "wav_path": str(wav_path),
        "tg_path": str(tg_path),
        "language": language or "",
        "audio_duration_sec": float(audio_duration_sec),
        "results": [result_to_cache_record(result) for result in results],
    }


def dataset_selection_fingerprint(sample_records: list[dict[str, Any]]) -> str:
    return stable_hash(sample_records)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Manage cached evaluation transcribe outputs.")
    parser.add_argument("--evaluation_file", type=str, default=str(DEFAULT_EVALUATION_FILE), help="Evaluation CSV whose sibling cache folder should be managed.")
    parser.add_argument("--clear", action="store_true", help="Delete the evaluation cache folder.")
    args = parser.parse_args()

    if args.clear:
        # The CLI is intentionally limited to the cache folder next to the
        # selected evaluation CSV, so --clear cannot target arbitrary paths.
        removed_cache_dir = clear_evaluation_cache(args.evaluation_file)
        print(f"Evaluation cache cleared: {removed_cache_dir}")
        return

    parser.print_help()


if __name__ == "__main__":
    cli()
