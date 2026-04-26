#!/usr/bin/env python3
import sys
from pathlib import Path

# Add repo root to PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import argparse
import os
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

    effective_lang = lang or "en"
    key = ("multi", effective_lang) if multilingual else ("single", effective_lang)
    if key not in get_tokenizer_cached._cache:
        get_tokenizer_cached._cache[key] = careless_whisper_stream.tokenizer.get_tokenizer(
            True, language=effective_lang, task="transcribe"
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


def resolve_csv_relative_path(csv_path: str, value: str) -> str:
    value = str(value)
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(csv_path)), value))


def maybe_make_manifest_path(path: Path, manifest_path: Path, global_paths: bool) -> str:
    path = path.resolve()
    manifest_path = manifest_path.resolve()
    if global_paths:
        return str(path)
    return os.path.relpath(path, manifest_path.parent)


def process_row(
    row,
    csv_path: str,
    sample_rate: int,
    get_streamed_mel: bool,
    n_mels: int,
    multilingual: bool,
    default_lang: str,
    spec_streamer: SpectrogramStream | None,
):
    wav_path = resolve_csv_relative_path(csv_path, row.wav_path)
    tg_path = resolve_csv_relative_path(csv_path, row.tg_path)
    lang = row.lang if hasattr(row, "lang") and pd.notna(row.lang) else default_lang

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
    parser.add_argument("--lang", default="en", help="Default Whisper language token to bake into samples when the CSV has no lang column, e.g. de")
    parser.add_argument("--get_streamed_mel", action="store_true")
    parser.add_argument("--multilingual", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--global_paths", action="store_true", help="If set, store absolute pt_path values in manifest.csv. Default: store paths relative to manifest location.",)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

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
            manifest_rows.append({
                "index": idx,
                "pt_path": maybe_make_manifest_path(out_path, manifest_path, args.global_paths),
            })
            pbar.update(1)
            continue

        item = process_row(
            row=row,
            csv_path=args.csv,
            sample_rate=args.sample_rate,
            get_streamed_mel=args.get_streamed_mel,
            n_mels=args.n_mels,
            multilingual=args.multilingual,
            default_lang=args.lang,
            spec_streamer=spec_streamer,
        )

        torch.save(item, out_path)

        manifest_rows.append({
            "index": idx,
            "pt_path": maybe_make_manifest_path(out_path, manifest_path, args.global_paths),
        })

        pbar.set_postfix({
            "last": Path(resolve_csv_relative_path(args.csv, row.wav_path)).name
        })
        pbar.update(1)

    pbar.close()

    elapsed = time.time() - start_time
    print(f"\nFinished in {elapsed/60:.2f} minutes")
    print(f"Avg speed: {len(df)/elapsed:.2f} samples/sec")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(manifest_path, index=False)
    print(f"Done. Wrote {len(manifest)} items to {samples_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
