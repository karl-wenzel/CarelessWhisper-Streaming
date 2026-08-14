"""Command-line endpoint for CarelessWhisper evaluation."""

import argparse

import torch

from evaluation.caching import validate_parameter_classification
from evaluation.runner import evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CarelessWhisper WER on a dataset")

    # When adding parameters, classify them in evaluation/caching.py and run
    # tests.test_evaluation_caching_contract.
    parser.add_argument("--model", required=True, help="Model run name, or an offline Whisper model name/path.")
    parser.add_argument("--offline_whisper", action="store_true", help="Evaluate a non-streaming Whisper model.")
    parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint epoch, e.g. 7 for checkpoint-0007.")
    parser.add_argument("--chunk_size", type=int, default=300, help="Streaming chunk size (granularity).")
    parser.add_argument("--multilingual", action="store_true", help="Use a multilingual model.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset_fraction", type=float, default=1.0, help="Fraction of the dataset to evaluate.")
    parser.add_argument("--dataset_sample_count", type=int, default=None, help="Evaluate exactly this many randomly sampled rows.")
    parser.add_argument("--dataset_partition", default="test", help="Dataset partition.")
    parser.add_argument("--samples_over", type=int, default=None, metavar="SECONDS", help="Keep only samples longer than this duration.")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size during inference.")
    parser.add_argument("--enable_relative_beam_stop", action="store_true", help="Wait until at least 20%% of beams emit EOS.")
    parser.add_argument("--max_sec_context", type=int, default=30, help="Maximum retained audio context in seconds.")
    parser.add_argument("--lang", default=None, help="Language code, such as en or de.")
    parser.add_argument("--encoder_positional_mode", choices=["auto", "sinusoidal", "alibi"], default="auto")
    parser.add_argument("--strict_k", type=int, nargs="*", default=[2], help="Strict-WER word correction distances.")
    parser.add_argument("--wir_n", type=int, nargs="*", default=[], help="Additional WIR suffix tolerances.")
    parser.add_argument("--sa_kv_cache", "-sa_kv_cache", dest="sa_kv_cache", action="store_true", help="Use decoder self-attention KV caching.")
    parser.add_argument("--ca_kv_cache", "-ca_kv_cache", dest="ca_kv_cache", action="store_true", help="Use decoder cross-attention KV caching.")
    parser.add_argument("--use_sliding_encoder_cache", action="store_true", help="Slide the encoder cache instead of resetting it.")
    parser.add_argument("--disable_encoder_kv_cache", action="store_true", help="Recompute the full encoder prefix.")
    parser.add_argument("--encoder_cache_diagnostics", action="store_true", help="Collect encoder-cache parity statistics.")
    parser.add_argument("--encoder_cache_diagnostic_interval", type=int, default=1, help="Sample cache parity every N chunks.")
    parser.add_argument("--reset_decoder_on_encoder_slide", action="store_true", help="Roll the decoder prefix when old encoder audio is pruned.")
    parser.add_argument("--decoder_roll_overlap_seconds", type=float, default=5.0)
    parser.add_argument("--decoder_roll_min_interval_seconds", type=float, default=2.0)
    parser.add_argument("--decoder_roll_max_prefix_tokens", type=int, default=48)
    parser.add_argument("--decoder_token_time_lag_seconds", type=float, default=2.0)
    parser.add_argument("--decoder_roll_diagnostics", action="store_true", help="Deprecated no-op.")
    parser.add_argument("--prefix_wer", action="store_true", help="Calculate WER for cumulative audio prefixes.")
    parser.add_argument("--delay_n_rtf", action="store_true", help="Calculate delayed visual-emission latency and RTF.")
    parser.add_argument("--verbose", "-verbose", dest="verbose", action="store_true")
    parser.add_argument("--cw", "-cw", dest="cw", action="store_true", help="Use a CarelessWhisper base model.")
    parser.add_argument("--force_hf_download", action="store_true", help="Force a fresh CarelessWhisper Hugging Face download.")
    parser.add_argument("--no_evaluation_cache", action="store_true", help="Do not reuse cached transcriptions.")
    parser.add_argument("--dataset_name", required=True, help="Key from training_code.ds_dict.ds_paths.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_parameter_classification(list(vars(args)))
    evaluate(args)


if __name__ == "__main__":
    main()
