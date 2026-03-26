#!/usr/bin/env python3
import sys
from pathlib import Path

# Add repo root to PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import argparse
import os
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import torch
from tqdm import tqdm
import time
from praatio import textgrid

import careless_whisper_stream
import careless_whisper_stream.tokenizer
from careless_whisper_stream.audio import SpectrogramStream
from careless_whisper_stream.tokenizer import Tokenizer


@dataclass
class Interval:
    label: str
    start: float
    end: float


def get_tokenizer_cached(multilingual: bool, lang: str | None) -> Tokenizer:
    if not hasattr(get_tokenizer_cached, "_cache"):
        get_tokenizer_cached._cache = {}

    key = ("multi", lang) if multilingual else ("en", "en")
    if key not in get_tokenizer_cached._cache:
        language = lang if multilingual else "en"
        get_tokenizer_cached._cache[key] = careless_whisper_stream.tokenizer.get_tokenizer(
            True, language=language, task="transcribe"
        )
    return get_tokenizer_cached._cache[key]


def load_intervals(tg_path: str, sr: int) -> list[Interval]:
    if tg_path.endswith(".wrd"):
        intervals: list[Interval] = []
        with open(tg_path, "r", encoding="utf-8") as f:
            for line in f:
                start, end, label = line.strip().split()
                intervals.append(Interval(label=label, start=int(start) / sr, end=int(end) / sr))
        return intervals

    tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=False)
    tier = tg.getTier("words")
    return [Interval(label=x.label, start=x.start, end=x.end) for x in tier.entries]


def calc_mel(audio, get_streamed_mel: bool, n_mels: int, spec_streamer: SpectrogramStream | None):
    if get_streamed_mel:
        assert spec_streamer is not None
        mel = spec_streamer._simulate_streaming_log_spec(torch.tensor(audio))
        return mel.squeeze(0) if mel.ndim == 3 and mel.shape[0] == 1 else mel
    return careless_whisper_stream.log_mel_spectrogram(audio)


def build_tokens_and_endpoints(
    intervals: list[Interval],
    tokenizer: Tokenizer,
) -> tuple[list[int], list[float], list[int]]:
    endpoints = [0.0, 0.0, 0.0]
    tokens: list[int] = []

    for i, interval in enumerate(intervals):
        piece = interval.label if i == 0 else " " + interval.label
        curr_tokens = tokenizer.encode(piece)

        if len(curr_tokens) == 0:
            continue

        n_diff = (interval.end - interval.start) / len(curr_tokens)
        endpoints.extend([interval.start + (j + 1) * n_diff for j in range(len(curr_tokens))])
        tokens.extend(curr_tokens)

    text = [*tokenizer.sot_sequence_including_notimestamps] + tokens
    labels = text[1:] + [tokenizer.eot]

    if endpoints:
        endpoints.append(endpoints[-1] + 0.5)
    else:
        endpoints = [0.0, 0.0, 0.0, 0.5]

    if not (len(endpoints) == len(labels) == len(text)):
        raise ValueError(
            f"Length mismatch: len(endpoints)={len(endpoints)}, len(labels)={len(labels)}, len(text)={len(text)}"
        )

    return text, endpoints, labels


def process_row(
    row,
    sample_rate: int,
    get_streamed_mel: bool,
    n_mels: int,
    multilingual: bool,
    spec_streamer: SpectrogramStream | None,
):
    wav_path = row.wav_path
    tg_path = row.tg_path
    lang = row.lang if hasattr(row, "lang") and pd.notna(row.lang) else "en"

    tokenizer = get_tokenizer_cached(multilingual=multilingual, lang=lang)

    audio = careless_whisper_stream.load_audio(wav_path, sr=sample_rate)
    audio = careless_whisper_stream.pad_or_trim(audio.flatten())
    mel = calc_mel(audio, get_streamed_mel=get_streamed_mel, n_mels=n_mels, spec_streamer=spec_streamer)

    intervals = load_intervals(tg_path, sr=sample_rate)
    dec_input_ids, endpoints, labels = build_tokens_and_endpoints(intervals, tokenizer)

    item = {
        "input_ids": mel.cpu(),
        "dec_input_ids": torch.tensor(dec_input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "endpoints": torch.tensor(endpoints, dtype=torch.float32),
        "lang": lang,
        "wav_path": wav_path,
        "tg_path": tg_path,
    }
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV used by AlignedTextGridDataset")
    parser.add_argument("--out_dir", required=True, help="Directory to save .pt samples and manifest.csv")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--get_streamed_mel", action="store_true")
    parser.add_argument("--multilingual", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    spec_streamer = SpectrogramStream(n_mels=args.n_mels) if args.get_streamed_mel else None

    manifest_rows = []

    start_time = time.time()

    pbar = tqdm(
        total=len(df),
        desc="Precomputing",
        unit="samples",
        dynamic_ncols=True
    )


    for idx, row in enumerate(df.itertuples(index=False)):
        out_path = samples_dir / f"{idx:08d}.pt"

        if out_path.exists() and not args.overwrite:
            manifest_rows.append({"index": idx, "pt_path": str(out_path)})
            pbar.update(1)
            continue

        item = process_row(
            row=row,
            sample_rate=args.sample_rate,
            get_streamed_mel=args.get_streamed_mel,
            n_mels=args.n_mels,
            multilingual=args.multilingual,
            spec_streamer=spec_streamer,
        )

        torch.save(item, out_path)

        manifest_rows.append({"index": idx, "pt_path": str(out_path)})

        # update progress bar
        pbar.set_postfix({
            "last": Path(row.wav_path.name)
        })
        pbar.update(1)

    pbar.close()

    elapsed = time.time() - start_time
    print(f"\nFinished in {elapsed/60:.2f} minutes")
    print(f"Avg speed: {len(df)/elapsed:.2f} samples/sec")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(f"Done. Wrote {len(manifest)} items to {samples_dir}")
    print(f"Manifest: {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()