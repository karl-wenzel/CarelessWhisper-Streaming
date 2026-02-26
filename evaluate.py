import os
import torch
import numpy as np
from jiwer import wer
from typing import List

# Import your local classes (assumes evaluate.py is in the root of the repo)
from careless_whisper_stream.streaming_model import StreamingWhisper
from careless_whisper_stream.audio import load_audio, SAMPLE_RATE

# --- CONSTANTS ---
MODEL_PATH = "path/to/your/checkpoint.ckpt"  # Replace with actual model name/path
DATASET_PATH = "/absolute/path/to/test_dataset"  # Path to folder of .wav files
GROUND_TRUTH_EXT = ".txt"  # Assumes ground truth is in .txt with same name as .wav
GRANULARITY = 16  # Matches training 'gran' (16 * 20ms = 320ms chunks)

def calculate_flicker(history: List[str]) -> int:
    """
    Calculates the 'Flicker' count for ARWER.
    A flicker occurs when a previously emitted word is changed or 
    deleted in a subsequent streaming update.
    """
    flickers = 0
    for i in range(1, len(history)):
        prev_words = history[i-1].split()
        curr_words = history[i].split()
        
        # Check if any "stable" word in previous step was altered
        min_len = min(len(prev_words), len(curr_words))
        for j in range(min_len):
            if prev_words[j] != curr_words[j]:
                flickers += 1
    return flickers

def evaluate():
    # 1. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StreamingWhisper.load_from_checkpoint(MODEL_PATH).to(device)
    model.eval()

    all_wer, all_awer, all_arwer = [], [], []

    # 2. Iterate through dataset
    audio_files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".wav")]
    
    for filename in audio_files:
        audio_path = os.path.join(DATASET_PATH, filename)
        label_path = os.path.join(DATASET_PATH, filename.replace(".wav", GROUND_TRUTH_EXT))
        
        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            ground_truth = f.read().strip()

        # Reset model state for a fresh stream
        model.reset()
        
        audio = load_audio(audio_path)
        chunk_size = int(SAMPLE_RATE * (GRANULARITY * 0.02)) # e.g. 320ms
        
        history = []
        
        # 3. Simulate Streaming Loop
        with torch.no_grad():
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                
                # decode() manages the SpectrogramStream and KV-cache hooks
                result = model.decode(chunk, use_frames=True)
                history.append(result.text)

        final_prediction = history[-1]
        
        # 4. Metric Calculations
        # WER: Standard offline comparison
        current_wer = wer(ground_truth, final_prediction)
        
        # AWER: Usually identical to WER for causal models unless 
        # using 'forced alignment' to check temporal accuracy. 
        # Here we treat it as the WER of the causal stream.
        current_awer = current_wer 

        # ARWER: WER + Normalized Flicker
        flicker_count = calculate_flicker(history)
        n_words = len(ground_truth.split())
        current_arwer = current_wer + (flicker_count / max(1, n_words))

        all_wer.append(current_wer)
        all_awer.append(current_awer)
        all_arwer.append(current_arwer)

        print(f"File: {filename} | WER: {current_wer:.4f} | ARWER: {current_arwer:.4f}")

    # 5. Final Report
    print("\n--- Final Results ---")
    print(f"Mean WER:   {np.mean(all_wer):.4f}")
    print(f"Mean AWER:  {np.mean(all_awer):.4f}")
    print(f"Mean ARWER: {np.mean(all_arwer):.4f}")

if __name__ == "__main__":
    evaluate()