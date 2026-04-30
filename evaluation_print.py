import argparse
import csv
from pathlib import Path


DEFAULT_EVALUATION_FILE = Path.home() / "ma" / "data" / "evaluation.csv"


def _read_rows(evaluation_file: str | Path) -> list[dict]:
    csv_path = Path(evaluation_file).expanduser()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _fmt_percent(value) -> str:
    return f"{_to_float(value) * 100:.2f}%"


def _fmt_latency_ms(value) -> str:
    return f"{_to_float(value):.1f} ms"


def _format_row(row: dict) -> str:
    lines = []
    lines.append("=" * 30)
    lines.append("RESULTS FOR:")
    lines.append(f"MODEL: {row.get('model_name', '')} on DATASET: {row.get('dataset', '')}")
    lines.append(
        f"PARTITION: {row.get('partition', '')} % SAMPLES: {row.get('fraction', '')}"
    )

    checkpoint = row.get("checkpoint", "")
    if checkpoint:
        lines.append(f"CHECKPOINT:    {checkpoint}")

    lines.append(f"BASE MODEL:    {row.get('base_model_name', '')}")
    lines.append(f"CHUNK SIZE:    {row.get('chunk_size', '')}")
    lines.append(f"MODE:          {'cw' if str(row.get('is_cw_model', '')).lower() == 'true' else 'local'}")
    lines.append(f"WER:           {_fmt_percent(row.get('wer'))}")
    lines.append(
        f"STRICT WER:    {_fmt_percent(row.get('strict_wer'))} (k={_to_int(row.get('strict_k'))})"
    )

    lines.append("")
    lines.append("=== Final WER IDS Breakdown ===")
    lines.append(f"Insertions:    {_to_int(row.get('wer_insertions'))}")
    lines.append(f"Deletions:     {_to_int(row.get('wer_deletions'))}")
    lines.append(f"Substitutions: {_to_int(row.get('wer_substitutions'))}")
    lines.append(f"Correct:       {_to_int(row.get('wer_correct'))}")
    lines.append(
        f"Total errors:  {_to_int(row.get('wer_insertions')) + _to_int(row.get('wer_deletions')) + _to_int(row.get('wer_substitutions'))}"
    )

    lines.append("")
    lines.append("=== Final Strict WER IDS Breakdown ===")
    lines.append(f"Insertions:    {_to_int(row.get('strict_wer_insertions'))}")
    lines.append(f"Deletions:     {_to_int(row.get('strict_wer_deletions'))}")
    lines.append(f"Substitutions: {_to_int(row.get('strict_wer_substitutions'))}")
    lines.append(f"Correct:       {_to_int(row.get('strict_wer_correct'))}")
    lines.append(
        f"Total errors:  {_to_int(row.get('strict_wer_insertions')) + _to_int(row.get('strict_wer_deletions')) + _to_int(row.get('strict_wer_substitutions'))}"
    )

    lines.append(f"RWER:          {_fmt_percent(row.get('rwer'))}")
    lines.append(f"ARWER:         {_fmt_percent(row.get('arwer'))}")
    lines.append(f"WIR:           {_fmt_percent(row.get('wir'))}")
    lines.append(f"WIR Changes:   {_to_int(row.get('wir_changed_words'))}")
    lines.append(f"WIR Words:     {_to_int(row.get('wir_total_words'))}")
    lines.append("-" * 20)
    lines.append(f"Avg Latency:   {_fmt_latency_ms(row.get('avg_latency_ms'))}")
    lines.append(f"RTF:           {_to_float(row.get('rtf')):.4f}")
    timestamp = row.get("timestamp", "")
    if timestamp:
        lines.append(f"TIMESTAMP:     {timestamp}")
    lines.append("=" * 30)
    return "\n".join(lines)


def print_rows(rows: list[dict]) -> None:
    if not rows:
        print("No evaluation rows found.")
        return

    for index, row in enumerate(rows):
        if index > 0:
            print()
        print(_format_row(row))


def print_latest_rows(evaluation_file: str | Path = DEFAULT_EVALUATION_FILE, row_count: int = 3) -> None:
    rows = _read_rows(evaluation_file)
    if row_count <= 0:
        print("row_count must be positive.")
        return

    latest_rows = list(reversed(rows[-row_count:]))
    print_rows(latest_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print recent evaluation runs from the evaluation CSV.")
    parser.add_argument("--evaluation_file", type=str, default=str(DEFAULT_EVALUATION_FILE))
    parser.add_argument("--row_count", type=int, default=3, help="How many latest rows to print.")
    args = parser.parse_args()
    print_latest_rows(args.evaluation_file, args.row_count)


if __name__ == "__main__":
    main()
