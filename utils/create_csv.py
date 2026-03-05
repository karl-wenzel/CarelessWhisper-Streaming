#!/usr/bin/env python3
"""
Create CSV from dataset + per-utterance MFA TextGrids.
Supports .flac and .wav audio files.

Usage:
    ./create_csv.py \
        --dataset_root /path/to/LibriSpeech/train-clean-100 \
        --align_root /path/to/train-clean-100-aligned \
        --output_dir /path/to/output/csvs \
        [--split]
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
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# -----------------------------
# Gather all audio and TextGrid paths
# -----------------------------
entries = []
AUDIO_EXTENSIONS = (".wav", ".flac")

for root, _, files in os.walk(args.dataset_root):
    for file in files:
        if file.endswith(AUDIO_EXTENSIONS):
            audio_path = os.path.join(root, file)
            # Compute relative path to maintain structure
            rel_path = os.path.relpath(audio_path, args.dataset_root)
            # Corresponding TextGrid path
            textgrid_path = os.path.join(args.align_root, os.path.splitext(rel_path)[0] + ".TextGrid")
            if os.path.exists(textgrid_path):
                # raw_text can be empty or derived from filename
                raw_text = ""  # leave empty if you don’t want to parse transcripts
                entries.append((audio_path, textgrid_path, raw_text))
            else:
                print(f"Skipping {audio_path}: missing TextGrid")

if len(entries) == 0:
    print("No valid entries found. Exiting.")
    exit(1)

# -----------------------------
# Shuffle
# -----------------------------
random.seed(args.seed)
random.shuffle(entries)

# -----------------------------
# Helper to write CSV
# -----------------------------
def write_csv(file_path, data):
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wav_path", "tg_path", "raw_text"])
        for audio, tg, txt in data:
            writer.writerow([audio, tg, txt])

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