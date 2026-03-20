import argparse
import os
import pandas as pd
import torch
import jiwer
from tqdm import tqdm
from praatio import textgrid

from careless_whisper_stream import load_streaming_model
from careless_whisper_stream.streaming_transcribe import transcribe
from training_code.ds_dict import ds_paths

def extract_text_from_tg(tg_path):
    """Reconstructs the transcript from a TextGrid using praatio."""
    try:
        # Match the implementation in AlignedTextGridDataset
        tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=False)
        text_intervals = tg.getTier("words")
        
        # praatio intervals have a .label attribute
        words = [interval.label for interval in text_intervals if interval.label.strip()]
        return " ".join(words)
    except Exception as e:
        print(f"Error parsing TextGrid {tg_path}: {e}")
        return ""

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

    predictions = []
    references = []

    # 3. Inference Loop
    print(f"Starting evaluation on {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        wav_path = row['wav_path']
        tg_path = row['tg_path']
        reference_text = extract_text_from_tg(tg_path)
        
        # Use the transcribe logic from streaming_transcribe.py 
        # setting simulate_stream=True to handle the file as a stream
        results = transcribe(
            model=model,
            wav_file=wav_path,
            simulate_stream=True,
            language="en" if not args.multilingual else "auto",
            beam_size=5,
            temperature=0,
            flush_last_frame=args.flush_last_frame,
            pad_last_frame=args.pad_last_frame,
            verbose=False
        )
        
        # transcribe returns a list of result objects for each chunk
        # We take the text from the final result in the stream
        if results:
            predicted_text = results[-1].text
        else:
            predicted_text = ""

        pred = predicted_text.strip().lower()
        ref = reference_text.strip().lower()

        predictions.append(pred)
        references.append(ref)

        if (args.verbose):
            print("Pred: " + pred)
            print("Label:" + ref)
            print(f"WER: {jiwer.wer(ref, pred):.2%}")
            print("-"*30)

    # 4. Metric Calculation
    wer = jiwer.wer(references, predictions)
    
    print("\n" + "="*30)
    print(f"RESULTS FOR: {args.dataset_name}")
    print(f"WER: {wer * 100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    evaluate()