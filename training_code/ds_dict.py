import os 
"""
Taken from README.md:
### Dataset Structure

Before starting model training using the command-line interface provided below, you must first configure your dataset dictionary file located at `training_code/ds_dict.py`.

This file defines a Python dictionary named `ds_paths`, where you should specify paths to the `train`, `val`, and `test` partitions of your dataset. Each partition should be a CSV file with the following three columns:

1. `wav_path` — Path to the WAV audio file.  
2. `tg_path` — Path to the corresponding `.TextGrid` file containing forced alignment.  
3. `raw_text` — Ground truth transcription.

> **Note:** The dictionary key (i.e., the name of the dataset) will be used by the training script to identify and load the dataset correctly.
"""

ds_paths = {
    'LIBRI-960-ALIGNED': {
        'train': "/mtec/local/LibriSpeech_aligned/csvs/train-960.csv",
        'val': "/mtec/local/LibriSpeech_aligned/csvs/dev.csv",
        'test': "/mtec/local/LibriSpeech_aligned/csvs/test.csv",
        'test-clean': "/mtec/local/LibriSpeech_aligned/csvs/test-clean.csv",
        'test-other': "/mtec/local/LibriSpeech_aligned/csvs/test-other.csv",

        'precomputed': {
            'train': "/mtec/local/LibriSpeech_aligned/precomputed/train-960/manifest.csv",
            'val': "/mtec/local/LibriSpeech_aligned/precomputed/dev/manifest.csv",
            'test': "/mtec/local/LibriSpeech_aligned/precomputed/test/manifest.csv",
            'test-clean': "/mtec/local/LibriSpeech_aligned/precomputed/test-clean/manifest.csv",
            'test-other': "/mtec/local/LibriSpeech_aligned/precomputed/test-other/manifest.csv",
        },
    },

    'CV-DE-ALIGNED': {
        'train': "/mtec/local/MozillaCommonVoice/DE/csvs/train.csv",
        'val': "/mtec/local/MozillaCommonVoice/DE/csvs/dev.csv",
        'test': "/mtec/local/MozillaCommonVoice/DE/csvs/test.csv",

        'precomputed': {
            'train': "/mtec/local/MozillaCommonVoice/DE/precomputed/train/manifest.csv",
            'val': "/mtec/local/MozillaCommonVoice/DE/precomputed/dev/manifest.csv",
            'test': "/mtec/local/MozillaCommonVoice/DE/precomputed/test/manifest.csv",
        },
    },

    'LIBRI-100-CLEAN-ALIGNED': {
        'train': "/mtec/local/LibriSpeech_aligned/csvs/train-clean-100.csv",
        'val': "/mtec/local/LibriSpeech_aligned/csvs/dev-clean.csv",
        'test': "/mtec/local/LibriSpeech_aligned/csvs/test-clean.csv",

        'precomputed': {
            'train': "/mtec/local/LibriSpeech_aligned/precomputed/train-clean-100/manifest.csv",
            'val': "/mtec/local/LibriSpeech_aligned/precomputed/dev-clean/manifest.csv",
            'test': "/mtec/local/LibriSpeech_aligned/precomputed/test-clean/manifest.csv",
        },
    },
}
