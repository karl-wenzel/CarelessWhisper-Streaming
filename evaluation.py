import argparse
import os
import time
import json
import pandas as pd
import torch
import jiwer
from tqdm import tqdm
from praatio import textgrid
from pathlib import Path
import librosa
import numpy as np

from careless_whisper_stream import load_streaming_model
from careless_whisper_stream.streaming_transcribe import transcribe
from training_code.ds_dict import ds_paths

def extract_words_and_times_from_tg(tg_path):
    """Reconstructs the transcript and timestamps from a TextGrid using praatio."""
    try:
        tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=False)
        text_intervals = tg.getTier("words")
        
        # Return a list of dictionaries with word, start, and end times
        words = [
            {"word": interval.label.strip(), "start": interval.start, "end": interval.end} 
            for interval in text_intervals if interval.label.strip()
        ]
        return words
    except Exception as e:
        print(f"Error parsing TextGrid {tg_path}: {e}")
        return []

def get_gt_prefix_at_time(gt_words, current_time):
    """Returns the ground truth string spoken up to 'current_time'."""
    # We include words whose start time has passed. 
    return " ".join([w["word"] for w in gt_words if w["start"] <= current_time])

def calculate_idsc(ref, hyp):
    """Calculates Insertions, Deletions, Substitutions, and Correct hits."""
    if not ref and not hyp:
        return 0, 0, 0, 0
    if not ref:
        return len(hyp.split()), 0, 0, 0
    if not hyp:
        return 0, len(ref.split()), 0, 0
    
    out = jiwer.process_words(ref, hyp)
    return out.insertions, out.deletions, out.substitutions, out.hits

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate CarelessWhisper WER on a dataset")
    
    # Model Setup
    parser.add_argument("--model", type=str, default="small", help="Model size")
    parser.add_argument("--chunk_size", type=int, default=300, help="Chunk size (gran)")
    parser.add_argument("--multilingual", action="store_true", help="Use multilingual model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local_model_path", type=str, default=None, help="Path to local .pt file")
    parser.add_argument("--dataset_fraction", type=float, default=1.0, help="Fraction of the dataset, that will be used. 1.0 (100%) by default.")
    parser.add_argument("--dataset_partition", type=str, default="test", help="The partition of the dataset that will be used for evaluation. 'Test' by default.")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size during inference.")
    parser.add_argument("-flush_last_frame", action="store_true", help="Calculates last frame with final spectogram and streaming mode off")
    parser.add_argument("-pad_last_frame", action="store_true", help="Pads the last frame")
    parser.add_argument("-verbose", action="store_true", help="Prints additional info while evaluating")
    
    # Dataset Setup
    parser.add_argument("--dataset_name", type=str, required=True, help="Key from ds_paths in ds_dict.py")
    
    args = parser.parse_args()

    # 1. Load Model
    print(f"Loading model: {args.model}...")
    model = load_streaming_model(
        name=args.model,
        gran=args.chunk_size,
        multilingual=args.multilingual,
        device=args.device,
        local_ckpt_path=args.local_model_path
    )
    model.eval()

    # 2. Load Dataset CSV
    if args.dataset_name not in ds_paths:
        raise ValueError(f"Dataset {args.dataset_name} not found in ds_dict.py")
    
    csv_path = ds_paths[args.dataset_name][str(args.dataset_partition)]
    print(f"Loading test split from: {csv_path}")
    df = pd.read_csv(csv_path)

    if 0.0 < args.dataset_fraction < 1.0:
        df = df.sample(frac=args.dataset_fraction, random_state=42).reset_index(drop=True)
        print(f"Subsetting dataset to {args.dataset_fraction * 100:.1f}%. New size: {len(df)} samples.")
    elif args.dataset_fraction <= 0 or args.dataset_fraction > 1.0:
        print(f"Warning: dataset_fraction {args.dataset_fraction} is out of bounds. Using full dataset.")

    global_rwer_num, global_rwer_den = 0, 0
    global_arwer_num, global_arwer_den = 0, 0
    all_chunk_latencies = []
    total_audio_duration_sec = 0.0
    total_processing_time_sec = 0.0
    predictions, references = [], []

    # 3. Inference Loop
    print(f"Starting evaluation on {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        wav_path = row['wav_path']
        tg_path = row['tg_path']

        audio_duration = librosa.get_duration(path=wav_path)
        total_audio_duration_sec += audio_duration
        
        gt_words = extract_words_and_times_from_tg(tg_path)
        reference_text = " ".join([w["word"] for w in gt_words]).strip().lower()
        chunk_duration_sec = args.chunk_size / 1000.0 
        
        results = transcribe(
            model=model,
            wav_file=wav_path,
            simulate_stream=True,
            language="en" if not args.multilingual else "auto",
            beam_size=args.beam_size,
            temperature=0,
            flush_last_frame=args.flush_last_frame,
            pad_last_frame=args.pad_last_frame,
            verbose=False
        )
        
        # RWER / ARWER Calculation per sample
        for step, res in enumerate(results):
            hyp_text = res.text.strip().lower()
            
            # --- Latency Tracking ---
            p_latency = getattr(res, 'processing_time', 0.0)
            all_chunk_latencies.append(p_latency)
            total_processing_time_sec += p_latency

            # --- RWER/ARWER logic ---
            audio_time_rho = (step + 1) * chunk_duration_sec 
            gt_text_rho = get_gt_prefix_at_time(gt_words, audio_time_rho).lower()
            
            i, d, s, c = calculate_idsc(gt_text_rho, hyp_text)
            global_rwer_num += (i + d + s)
            global_rwer_den += (c + d + s)

            real_time_tau = audio_time_rho + p_latency
            gt_text_tau = get_gt_prefix_at_time(gt_words, real_time_tau).lower()
            
            i_a, d_a, s_a, c_a = calculate_idsc(gt_text_tau, hyp_text)
            global_arwer_num += (i_a + d_a + s_a)
            global_arwer_den += (c_a + d_a + s_a)

        # Final transcript for standard WER
        predicted_text = results[-1].text if results else ""
        predictions.append(predicted_text.strip().lower())
        references.append(reference_text)

        if args.verbose:
            print("Pred: " + predicted_text)
            print("Label:" + reference_text)
            print("-" * 30)

    # 4. Final Aggregated Metric Calculation
    wer = jiwer.wer(references, predictions) if references else 0
    rwer = global_rwer_num / global_rwer_den if global_rwer_den > 0 else 0
    arwer = global_arwer_num / global_arwer_den if global_arwer_den > 0 else 0
    
    # Latency & RTF
    avg_latency = np.mean(all_chunk_latencies) if all_chunk_latencies else 0
    rtf = total_processing_time_sec / total_audio_duration_sec if total_audio_duration_sec > 0 else 0
    
    # --- Prepare Stats Dictionary ---
    stats = {
        "dataset": args.dataset_name,
        "partition": args.dataset_partition,
        "fraction": args.dataset_fraction,
        "wer": float(wer),
        "rwer": float(rwer),
        "arwer": float(arwer),
        "avg_latency_ms": float(avg_latency * 1000),
        "rtf": float(rtf),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # --- Save JSON if using local model ---
    if args.local_model_path:
        # Get the directory where the .pt file is located
        model_dir = Path(args.local_model_path).parent.parent
        save_path = model_dir / "evaluation.json"
        
        with open(save_path, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"Stats saved to: {save_path}")

    print("\n" + "="*30)
    print(f"RESULTS FOR: {args.dataset_name}")
    print(f"WER:           {wer * 100:.2f}%")
    print(f"RWER:          {rwer * 100:.2f}%")
    print(f"ARWER:         {arwer * 100:.2f}%")
    print("-" * 20)
    print(f"Avg Latency:   {avg_latency * 1000:.1f} ms")
    print(f"RTF:           {rtf:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()