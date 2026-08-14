"""Command-line endpoint for printing saved evaluation results."""

import argparse

from evaluation.presentation import DEFAULT_EVALUATION_FILE, print_latest_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print recent evaluation runs from the evaluation CSV.")
    parser.add_argument("--evaluation_file", default=str(DEFAULT_EVALUATION_FILE))
    parser.add_argument("--row_count", type=int, default=3, help="Number of recent rows to print.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print_latest_rows(args.evaluation_file, args.row_count)


if __name__ == "__main__":
    main()
