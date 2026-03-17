#!/usr/bin/env python3

import argparse
import csv
import sys
from pathlib import Path

def merge_csvs(input_files, output_path):
    combined_rows = []
    header = None

    for file_path in input_files:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File {file_path} not found. Skipping.")
            continue
        
        print(f"Reading {path}...")
        with open(path, "r", encoding="utf8", newline="") as f:
            reader = csv.reader(f)
            try:
                file_header = next(reader)
            except StopIteration:
                print(f"Warning: {file_path} is empty. Skipping.")
                continue

            # Set the master header from the first valid file
            if header is None:
                header = file_header
            
            # Add all rows from this file to our list
            for row in reader:
                combined_rows.append(row)

    if not combined_rows:
        print("No data found to merge.")
        return

    # Write the combined data to the output file
    print(f"Writing {len(combined_rows)} rows to {output_path}...")
    with open(output_path, "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(combined_rows)
    
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Merge multiple CSV files into one.")

    # nargs="+" allows you to pass one or more file paths
    parser.add_argument("input_files", nargs="+", help="List of CSV files to combine")
    parser.add_argument("-o", "--output", required=True, help="The name/path of the new combined CSV file")

    args = parser.parse_args()

    merge_csvs(args.input_files, args.output)

if __name__ == "__main__":
    main()