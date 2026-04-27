#!/usr/bin/env python3
"""
Create CSV from dataset + per-utterance MFA TextGrids.
Supports .flac, .mp3, and .wav audio files.

Usage:
    ./create_csv.py \
        --dataset_root /path/to/LibriSpeech/train-clean-100 \
        --align_root /path/to/train-clean-100-aligned \
        --output_dir /path/to/output/csvs \
        [--split] [--global_paths]
"""

import os
import csv
import random
import argparse

# -----------------------------
# Argument parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Create CSV from dataset + MFA TextGrids")
parser.add_argument("--dataset_root", type=str, required=True, help="Root folder of audio files")
parser.add_argument("--align_root", type=str, required=True, help="Root folder of MFA TextGrids")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save CSV(s)")
parser.add_argument("--split", action="store_true", help="Whether to split into train/val/test")
parser.add_argument("--train_ratio", type=float, default=0.8, help="Training split ratio")
parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--global_paths", action="store_true", help="If set, store absolute paths in CSV. Default: store paths relative to CSV location.")
parser.add_argument("-v", "--verbose", action="store_true", help="Print per-file warnings while scanning the dataset.")

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# -----------------------------
# Gather all audio and TextGrid paths
# -----------------------------
entries = []
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3")
total_audio_files = 0
missing_textgrid_count = 0
missing_transcript_count = 0


def load_transcriptions(root, files):
    transcriptions = {}

    # LibriSpeech-style folder-level transcript files
    for file in files:
        if file.endswith(".trans.txt"):
            trans_path = os.path.join(root, file)
            with open(trans_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        file_id, text = parts
                        transcriptions[file_id] = text

    # MFA/Common Voice-style per-utterance .lab files
    for file in files:
        if file.endswith(".lab"):
            file_id = os.path.splitext(file)[0]
            lab_path = os.path.join(root, file)
            with open(lab_path, "r", encoding="utf-8") as f:
                transcriptions[file_id] = f.read().strip()

    return transcriptions


def print_summary():
    print("Scan summary:")
    print(f"Audio files found: {total_audio_files}")
    print(f"Valid entries written: {len(entries)}")
    print(f"Missing TextGrids: {missing_textgrid_count}")
    print(f"Missing transcripts: {missing_transcript_count}")


for root, _, files in os.walk(args.dataset_root):
    transcriptions = load_transcriptions(root, files)

    for file in files:
        if file.lower().endswith(AUDIO_EXTENSIONS):
            total_audio_files += 1
            audio_path = os.path.join(root, file)
            file_id = os.path.splitext(file)[0]

            rel_path = os.path.relpath(audio_path, args.dataset_root)
            textgrid_path = os.path.join(
                args.align_root,
                os.path.splitext(rel_path)[0] + ".TextGrid"
            )

            if os.path.exists(textgrid_path):
                raw_text = transcriptions.get(file_id, "")
                if raw_text == "":
                    missing_transcript_count += 1
                    if args.verbose:
                        print(f"Warning: No transcript found for {file_id} in {root}")

                entries.append((audio_path, textgrid_path, raw_text))
            else:
                missing_textgrid_count += 1
                if args.verbose:
                    print(f"Skipping {audio_path}: missing TextGrid")

if len(entries) == 0:
    print_summary()
    print("No valid entries found. Exiting.")
    exit(1)

# -----------------------------
# Shuffle
# -----------------------------
random.seed(args.seed)
random.shuffle(entries)

# -----------------------------
# Helper to maybe relativize paths
# -----------------------------
def maybe_make_local(path, csv_path):
    if args.global_paths:
        return path
    base_dir = os.path.dirname(csv_path)
    return os.path.relpath(path, base_dir)

# -----------------------------
# Helper to write CSV
# -----------------------------
def write_csv(file_path, data):
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wav_path", "tg_path", "raw_text"])
        for audio, tg, txt in data:
            audio_out = maybe_make_local(audio, file_path)
            tg_out = maybe_make_local(tg, file_path)
            writer.writerow([audio_out, tg_out, txt])

# -----------------------------
# Determine dataset name
# -----------------------------
dataset_name = os.path.basename(os.path.normpath(args.dataset_root))

# -----------------------------
# Write CSV(s)
# -----------------------------
if args.split:
    n_total = len(entries)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)

    train_entries = entries[:n_train]
    val_entries = entries[n_train:n_train + n_val]
    test_entries = entries[n_train + n_val:]

    write_csv(os.path.join(args.output_dir, f"{dataset_name}_train.csv"), train_entries)
    write_csv(os.path.join(args.output_dir, f"{dataset_name}_val.csv"), val_entries)
    write_csv(os.path.join(args.output_dir, f"{dataset_name}_test.csv"), test_entries)

    print(f"CSV files written with split to {args.output_dir}")
    print(f"Train: {len(train_entries)}, Val: {len(val_entries)}, Test: {len(test_entries)}")
else:
    output_file = os.path.join(args.output_dir, f"{dataset_name}.csv")
    write_csv(output_file, entries)
    print(f"Single CSV written to {output_file} ({len(entries)} entries)")

print_summary()
