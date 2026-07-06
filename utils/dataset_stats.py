#!/usr/bin/env python3

import os
import argparse
import subprocess
import json
import csv
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, median, stdev
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    import soundfile as sf
except ImportError:
    sf = None

AUDIO_EXTENSIONS = {".flac", ".mp3", ".wav", ".wave"}
DEFAULT_WORKERS = max(1, min(32, (os.cpu_count() or 1) * 2))


def get_audio_duration(path):
    # Fast path for large dataset scans: WAV/FLAC duration can usually be read
    # from container metadata without starting one ffprobe process per file.
    if sf is not None:
        try:
            info = sf.info(path)
            if info.samplerate > 0:
                return float(info.frames / info.samplerate)
        except Exception:
            pass

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
            text=True,
            check=True
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def scan_audio_files(folder, exclude_dirs=None):
    exclude_dirs = {os.path.abspath(d) for d in (exclude_dirs or [])}
    audio_files = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [
            d for d in dirs
            if os.path.abspath(os.path.join(root, d)) not in exclude_dirs
        ]

        if os.path.abspath(root) in exclude_dirs:
            continue

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


def collect_durations(files, workers):
    if workers <= 1:
        durations = []
        for f in tqdm(files, desc="Processing audio"):
            d = get_audio_duration(f)
            if d is not None:
                durations.append(d)
        return durations

    # Duration probing is mostly file I/O plus subprocess fallback, so threads
    # improve throughput without forcing every path to be multiprocessing-safe.
    durations = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for d in tqdm(
            executor.map(get_audio_duration, files),
            total=len(files),
            desc="Processing audio",
        ):
            if d is not None:
                durations.append(d)

    return durations


def process_dataset(folder, name, stats_dir, workers):

    print(f"\nScanning dataset: {name}")

    files = scan_audio_files(folder, exclude_dirs=[stats_dir])

    durations = collect_durations(files, workers)

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
    parser.add_argument(
        "--output_folder",
        "--output_dir",
        default=None,
        help="folder where statistics outputs will be written (default: <folder>/statistics)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"parallel duration probes to run (default: {DEFAULT_WORKERS})",
    )

    args = parser.parse_args()

    base_folder = os.path.abspath(args.folder)

    stats_dir = (
        os.path.abspath(args.output_folder)
        if args.output_folder
        else os.path.join(base_folder, "statistics")
    )
    os.makedirs(stats_dir, exist_ok=True)

    output = {}

    if not args.multiple:

        name = os.path.basename(base_folder)
        stats = process_dataset(base_folder, name, stats_dir, args.workers)

        output[name] = stats

    else:

        for entry in os.scandir(base_folder):

            if entry.is_dir() and os.path.abspath(entry.path) != os.path.abspath(stats_dir):

                stats = process_dataset(entry.path, entry.name, stats_dir, args.workers)

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
