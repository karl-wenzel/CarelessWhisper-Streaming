# prepare_cv_for_mfa.py
from pathlib import Path
import csv
import shutil
import re

cv_root = Path("/mtec/local/MozillaCVFull/DE/cv-corpus-25.0-2026-03-09/de")
tsv_path = cv_root / "validated.tsv"   # or train.tsv/dev.tsv/test.tsv
out_root = Path("/mtec/local/MozillaCVFull/DE/mfa_validated")

def normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("’", "'").replace("`", "'")
    s = re.sub(r"[“”„]", '"', s)
    # Conservative cleanup: remove punctuation MFA/German dictionary often won't want
    s = re.sub(r"[.,!?;:()\[\]{}\"…]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

with tsv_path.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rel_mp3 = row["path"]                 # e.g. common_voice_de_123.mp3
        sentence = normalize_text(row["sentence"])
        if not sentence:
            continue

        src_audio = cv_root / "clips" / rel_mp3
        if not src_audio.exists():
            continue

        # Optional speaker folders: MFA treats subfolders as speakers.
        speaker = row.get("client_id", "unknown")[:16]
        out_dir = out_root / speaker
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(rel_mp3).stem
        dst_audio = out_dir / f"{stem}.mp3"
        dst_lab = out_dir / f"{stem}.lab"

        if not dst_audio.exists():
            shutil.copy2(src_audio, dst_audio)

        dst_lab.write_text(sentence + "\n", encoding="utf-8")

print("Done:", out_root)