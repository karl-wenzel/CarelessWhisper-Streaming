# prepare_cv_for_mfa.py
import argparse
from pathlib import Path
import csv
import shutil
import re

def normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("’", "'").replace("`", "'")
    s = re.sub(r"[“”„]", '"', s)
    s = re.sub(r"[.,!?;:()\[\]{}\"…]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def main():
    parser = argparse.ArgumentParser(description="Convert Mozilla Common Voice to MFA format")
    parser.add_argument("--cv_root", type=Path, required=True,
                        help="Path to Common Voice language folder (contains clips/, *.tsv)")
    parser.add_argument("--tsv", type=str, required=True,
                        help="Which TSV file to use (e.g. validated.tsv, train.tsv)")
    parser.add_argument("--out_root", type=Path, required=True,
                        help="Output directory for MFA-formatted corpus")
    args = parser.parse_args()

    tsv_path = args.cv_root / args.tsv
    clips_dir = args.cv_root / "clips"
    out_root = args.out_root

    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")
    if not clips_dir.exists():
        raise FileNotFoundError(f"Clips dir not found: {clips_dir}")

    out_root.mkdir(parents=True, exist_ok=True)

    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rel_audio = row["path"]
            sentence = normalize_text(row["sentence"])

            if not sentence:
                continue

            src_audio = clips_dir / rel_audio
            if not src_audio.exists():
                continue

            speaker = row.get("client_id", "unknown")[:16]
            out_dir = out_root / speaker
            out_dir.mkdir(parents=True, exist_ok=True)

            stem = Path(rel_audio).stem
            dst_audio = out_dir / src_audio.name
            dst_lab = out_dir / f"{stem}.lab"

            if not dst_audio.exists():
                shutil.copy2(src_audio, dst_audio)

            dst_lab.write_text(sentence + "\n", encoding="utf-8")

    print(f"Done. Output written to: {out_root}")

if __name__ == "__main__":
    main()