import argparse
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

SPEAKER_TO_HEADSET = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s'äöüß]", "", text)
    return text.strip()

def parse_words_xml(path):
    root = ET.parse(path).getroot()

    words = []

    for elem in root.iter():
        if elem.tag.endswith("w"):
            word = (elem.text or "").strip()

            if not word:
                continue

            start = elem.attrib.get("starttime")
            end = elem.attrib.get("endtime")

            if start is None or end is None:
                continue

            words.append({
                "word": word,
                "start": float(start),
                "end": float(end),
            })

    return words

def group_words(words, max_gap=0.7, max_duration=15.0):
    segments = []

    cur = []

    for w in words:
        if not cur:
            cur.append(w)
            continue

        gap = w["start"] - cur[-1]["end"]
        dur = w["end"] - cur[0]["start"]

        if gap > max_gap or dur > max_duration:
            segments.append(cur)
            cur = [w]
        else:
            cur.append(w)

    if cur:
        segments.append(cur)

    return segments

def cut_audio(src, dst, start, end):
    dst.parent.mkdir(parents=True, exist_ok=True)

    duration = end - start

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(src),
        "-ac", "1",
        "-ar", "16000",
        str(dst)
    ]

    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ami_root", required=True)
    ap.add_argument("--out_root", required=True)

    args = ap.parse_args()

    ami_root = Path(args.ami_root)
    out_root = Path(args.out_root)

    words_files = sorted(ami_root.rglob("*.words.xml"))

    total = 0

    for wf in words_files:
        parts = wf.name.split(".")

        if len(parts) < 3:
            continue

        meeting = parts[0]
        speaker = parts[1]

        if speaker not in SPEAKER_TO_HEADSET:
            continue

        headset_id = SPEAKER_TO_HEADSET[speaker]

        wavs = list(ami_root.rglob(f"{meeting}.Headset-{headset_id}.wav"))

        if not wavs:
            continue

        wav_path = wavs[0]

        words = parse_words_xml(wf)

        if not words:
            continue

        segments = group_words(words)

        for idx, seg in enumerate(segments):
            start = min(w["start"] for w in seg)
            end = max(w["end"] for w in seg)

            if end <= start:
                print(f"Skipping invalid segment: {meeting} {speaker} {idx} start={start} end={end}")
                continue

            if end - start < 0.3:
                continue

            text = clean_text(
                " ".join(w["word"] for w in seg)
            )

            if len(text) < 2:
                continue

            utt_id = f"{meeting}_{speaker}_{idx:05d}"

            wav_out = out_root / f"{utt_id}.wav"
            lab_out = out_root / f"{utt_id}.lab"

            cut_audio(wav_path, wav_out, start, end)

            with open(lab_out, "w", encoding="utf-8") as f:
                f.write(text)

            total += 1

            if total % 100 == 0:
                print(total)

    print("Done:", total)

if __name__ == "__main__":
    main()