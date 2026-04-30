import csv
from pathlib import Path


def _normalize_row(row: dict) -> dict:
    normalized = {}
    for key, value in row.items():
        if value is None:
            normalized[key] = ""
        else:
            normalized[key] = value
    return normalized


def _read_existing_rows(csv_path: Path) -> tuple[list[str], list[dict]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return [], []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _merge_fieldnames(existing_fieldnames: list[str], row: dict) -> list[str]:
    fieldnames = list(existing_fieldnames)
    for key in row.keys():
        if key not in fieldnames:
            fieldnames.append(key)
    return fieldnames


def append_evaluation_row(evaluation_file: str | Path, row: dict) -> dict:
    csv_path = Path(evaluation_file).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_row = _normalize_row(row)
    existing_fieldnames, existing_rows = _read_existing_rows(csv_path)
    fieldnames = _merge_fieldnames(existing_fieldnames, normalized_row)

    all_rows = existing_rows + [normalized_row]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for existing_row in all_rows:
            writer.writerow({field: existing_row.get(field, "") for field in fieldnames})

    return normalized_row

