#!/usr/bin/env python3

import os
import argparse
import subprocess
import json
import csv
from statistics import mean, median, stdev
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

AUDIO_EXTENSIONS = {".flac", ".mp3", ".wav", ".wave"}


def get_audio_duration(path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return float(result.stdout.strip())
    except:
        return None


def scan_audio_files(folder):
    audio_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS:
                audio_files.append(os.path.join(root, f))
    return audio_files


def compute_stats(durations):

    durations = sorted(durations)

    stats = {
        "num_files": len(durations),
        "total_length_sec": float(sum(durations)),
        "total_length_hours": float(sum(durations) / 3600),
        "avg_length_sec": float(mean(durations)),
        "median_length_sec": float(median(durations)),
        "shortest_length_sec": float(min(durations)),
        "longest_length_sec": float(max(durations)),
        "std_dev_sec": float(stdev(durations)) if len(durations) > 1 else 0,
        "percentile_25_sec": float(np.percentile(durations, 25)),
        "percentile_75_sec": float(np.percentile(durations, 75))
    }

    return stats


def save_histogram(durations, out_path, title):

    plt.figure()
    plt.hist(durations, bins=50)
    plt.xlabel("Sample Length (seconds)")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_csv(durations, out_path):

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "length_sec"])
        for i, d in enumerate(durations):
            writer.writerow([i, d])


def process_dataset(folder, name, stats_dir):

    print(f"\nScanning dataset: {name}")

    files = scan_audio_files(folder)

    durations = []

    for f in tqdm(files, desc="Processing audio"):
        d = get_audio_duration(f)
        if d is not None:
            durations.append(d)

    if not durations:
        return None

    stats = compute_stats(durations)

    dataset_dir = os.path.join(stats_dir, name)
    os.makedirs(dataset_dir, exist_ok=True)

    # histogram
    save_histogram(
        durations,
        os.path.join(dataset_dir, "length_histogram.png"),
        f"Length Distribution ({name})"
    )

    # csv
    save_csv(
        durations,
        os.path.join(dataset_dir, "lengths.csv")
    )

    return stats


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("-multiple", action="store_true")

    args = parser.parse_args()

    base_folder = os.path.abspath(args.folder)

    stats_dir = os.path.join(base_folder, "statistics")
    os.makedirs(stats_dir, exist_ok=True)

    output = {}

    if not args.multiple:

        name = os.path.basename(base_folder)
        stats = process_dataset(base_folder, name, stats_dir)

        output[name] = stats

    else:

        for entry in os.scandir(base_folder):

            if entry.is_dir():

                stats = process_dataset(entry.path, entry.name, stats_dir)

                if stats:
                    output[entry.name] = stats

    # imbalance detection
    if len(output) > 1:

        counts = [d["num_files"] for d in output.values()]
        output["_dataset_balance"] = {
            "largest_dataset": max(counts),
            "smallest_dataset": min(counts),
            "imbalance_ratio": max(counts) / min(counts)
        }

    json_path = os.path.join(stats_dir, "statistics.json")

    with open(json_path, "w") as f:
        json.dump(output, f, indent=4)

    print("\nDone.")
    print(f"Results saved to: {stats_dir}")


if __name__ == "__main__":
    main()