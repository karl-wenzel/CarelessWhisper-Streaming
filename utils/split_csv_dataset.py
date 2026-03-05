#!/usr/bin/env python3

import argparse
import csv
import os
import random
from pathlib import Path


def load_rows(input_dir):
    rows = []
    header = None

    for file in Path(input_dir).glob("*.csv"):
        print(f"Reading {file}")
        with open(file, "r", encoding="utf8") as f:
            reader = csv.reader(f)
            file_header = next(reader)

            if header is None:
                header = file_header

            for row in reader:
                rows.append(row)

    return header, rows


def split_rows(rows, train_ratio, val_ratio, test_ratio):
    random.shuffle(rows)

    total = len(rows)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train = rows[:train_end]
    val = rows[train_end:val_end]
    test = rows[val_end:]

    return train, val, test


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Combine and split dataset CSV files")

    parser.add_argument("input_dir", help="Folder containing CSV files")
    parser.add_argument("output_dir", help="Folder where split CSVs will be written")

    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise ValueError("Train/Val/Test ratios must sum to 1")

    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    header, rows = load_rows(args.input_dir)

    if not rows:
        print("No rows found in CSV files.")
        return

    print(f"Total samples: {len(rows)}")

    train, val, test = split_rows(rows, args.train, args.val, args.test)

    write_csv(Path(args.output_dir) / "train.csv", header, train)
    write_csv(Path(args.output_dir) / "val.csv", header, val)
    write_csv(Path(args.output_dir) / "test.csv", header, test)


if __name__ == "__main__":
    main()