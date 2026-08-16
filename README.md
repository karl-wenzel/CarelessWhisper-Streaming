# CarelessWhisper - Causal Whisper Streaming Model
Causal Whisper Streaming is a fine tuned version of OpenAI Whisper, which can handle causal data and perform real-time transcription. 

[![arXiv](https://img.shields.io/badge/arXiv-2508.12301-b31b1b.svg)](https://arxiv.org/abs/2508.12301)  [![Demo on Hugging Face](https://img.shields.io/badge/🤗%20Demo-Hugging%20Face-blueviolet?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/MLSpeech/CarelessWhisper-causal-streaming)

## 📄 Paper

For more details, see our [paper](https://arxiv.org/abs/2508.12301).

## 🔧 Setup
We used Python 3.9.16, PyTorch 2.6.0, and PyTorch-Lightning 2.5.0 to train and test our models.
Portions of this code are adapted from [OpenAI's Whisper](https://github.com/openai/whisper).

To set up the project environment using `conda`, follow these steps:

1. **Clone the repository**  
   ```bash
   git clone https://github.com/tomer9080/CarelessWhisper-streaming
   cd CarelessWhisper-streaming
   ```

> 💡 Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed before proceeding.

2. **Create the conda environment**
    ```bash
    conda env create -f environment.yml
    ```

3. **Activate The environment**
    ```bash
    conda activate careless_whisper
    ```

4. **Install the appropriate PyTorch version**  
   Depending on your hardware and CUDA version, install PyTorch by following the instructions at [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally).  
   This project was tested with CUDA 12.4, but it should also work with compatible earlier or later versions.
 
After installing all of the dependencies, you can try to run inference.

## 🤖 Available Models
We fine-tuned three different sizes of Whisper, all support english only transcription.
A `large-v2` that was fine tuned on multilingual data is available, and supports English, French, Spanish, German and Portuguese with chunk size of 300 miliseconds.

| Size | Chunk Size [msec] | Multilingual | 
|:----:|:-----------------:|:------------:|
| base | 40, 100, 200, 300 |  N/A         |
| small| 40, 100, 200, 300, 1000| N/A     |
|large-v2| 40, 100, 200, 300, 1000| 300   |


## 🎤 Running Inference
To run inference, download the repo content, and run from the repository root accroding to following sections.

> **Note:** The models are hosted on the [Hugging Face Hub](https://huggingface.co/), which requires an access token.  
> Make sure you are logged in with your token to access the models.

### How to Apply Your Hugging Face 🤗 Access Token

1. **Create a Hugging Face account** (if you don’t have one) at [https://huggingface.co/join](https://huggingface.co/join).

2. **Generate an access token:**
   - Go to your Hugging Face account settings: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Click on **"New token"**, give it a name, select the appropriate scopes (usually `read` is enough), and create it.

3. **Login using the Hugging Face CLI:**  
   Install the CLI if you don’t have it:
   ```bash
   pip install huggingface_hub
   ```
   Then login:
   ```bash
   huggingface-cli login
   ```
   Paste your token when prompted.


### 🖥️ CLI Usage
The transcription model is easily activated using the next command:
```bash
# Using a local microphone for streaming transcription, dumping the recording to out.wav
python transcribe.py \
--output_filename out.wav \
--channels 2 \
--model small \ 
--chunk_size 300 \
--device cuda \
--beam_size 5 \
--ca_kv_cache \
```

A simulation of a stream on a wav file is also available:
```bash
# Simulating a stream on a wav file
python transcribe.py \
--model small \
--chunk_size 300 \
--device cuda \
--beam_size 5 \
--ca_kv_cache \
--wav_file /path/to/audio.wav \
--simulate_stream \
--use_latency
```

### 🐍 Python Usage
If you prefer using python, a code sinppet utilizing a microphone or a wav file is provided below:

```python
import torch
import careless_whisper_stream

model_size = "small" # model size
chunk_size = 300 # chunk size in milliseconds
multilingual = False # currently on large-v2_300msec supports other languages than english.
device = "cuda" if torch.cuda.is_available() else "cpu"

model = careless_whisper_stream.load_streaming_model(name=model_size,
                                                   gran=chunk_size,
                                                   multilingual=multilingual,
                                                   device=device)

# using a local microphone recording 
texts_microphone = model.transcribe(output_filename="/path/to/dump/file.wav",
                         channels=2,
                         beam_size=5,
                         ca_kv_cache=True)

# Simulating on a wav file
texts_wav_simulation = model.transcribe(simulate_stream=True,
                                        wav_file="/path/to/file/you/want/to/transcribe.wav",
                                        beam_size=5,
                                        ca_kv_cache=True)
```

## Evaluation

`evaluation.py` loads a model, selects a dataset partition from
`training_code/ds_dict.py`, transcribes the selected samples, calculates the
requested accuracy and timing metrics, and appends one row to
`$HOME/ma/data/evaluation.csv`. Streaming transcription results are cached in
the sibling `$HOME/ma/data/evaluation_cache` directory. Compatible later runs
can reuse those transcriptions when only evaluation-only metrics change.

The implementation is in the `evaluation/` package. The root files are slim
command-line endpoints: `evaluation.py` runs and saves evaluations, while
`evaluation_print.py` reads and prints saved rows without loading a model.

Register datasets in `training_code/ds_dict.py` first. Selected CSV rows must
provide `wav_path`, `tg_path`, and `raw_text`; an optional `lang` column can
override the language per sample.

### Running an evaluation

For a local training run, `--model` is resolved below
`$HOME/ma/data/models/ckpts/<model>/checkpoint/`. Without `--checkpoint`, the
runner prefers the best WER checkpoint recorded by Lightning and otherwise
uses the highest checkpoint epoch.

```bash
python evaluation.py \
  --model example_training_base_model \
  --dataset_name LIBRI-960-ALIGNED \
  --dataset_partition test \
  --checkpoint 7 \
  --chunk_size 300 \
  --device cuda \
  --beam_size 5 \
  --ca_kv_cache
```

ALiBi sliding-cache evaluation with encoder cache-parity diagnostics:

```bash
python evaluation.py \
  --model example_alibi_model \
  --dataset_name REVLONG \
  --encoder_positional_mode alibi \
  --chunk_size 300 \
  --max_sec_context 30 \
  --use_sliding_encoder_cache \
  --encoder_cache_diagnostics \
  --encoder_cache_diagnostic_interval 5 \
  --beam_size 5
```

Do not combine parity diagnostics with `--ca_kv_cache`: that path does not
retain the complete `audio_features` tensor required for the comparison. The
diagnostic is reported at the end and saved as one concatenated CSV field.

Enable decoder rolling when old decoder text should be retired as the encoder
window slides:

```bash
python evaluation.py \
  --model example_alibi_model \
  --dataset_name REVLONG \
  --encoder_positional_mode alibi \
  --use_sliding_encoder_cache \
  --reset_decoder_on_encoder_slide \
  --decoder_roll_overlap_seconds 5 \
  --decoder_roll_min_interval_seconds 2 \
  --decoder_roll_max_prefix_tokens 48 \
  --decoder_token_time_lag_seconds 2
```

Compare against standard non-streaming Whisper:

```bash
python evaluation.py \
  --model small \
  --offline_whisper \
  --dataset_name LIBRI-960-ALIGNED \
  --beam_size 5 \
  --device cuda
```

Streaming-only cache, rolling, WIR, prefix-WER, and delay-N options cannot be
used with `--offline_whisper`.

### Evaluation parameter reference

| Parameter | Default | Meaning and example |
|---|---:|---|
| `-h`, `--help` | — | Print CLI help and exit. |
| `--model NAME` | required | Local run name, CarelessWhisper name with `--cw`, or Whisper name/path with `--offline_whisper`; e.g. `--model small`. |
| `--dataset_name NAME` | required | Dataset key from `training_code/ds_dict.py`; e.g. `--dataset_name REVLONG`. |
| `--dataset_partition NAME` | `test` | Dataset partition; e.g. `--dataset_partition val`. |
| `--dataset_fraction F` | `1.0` | Random dataset fraction; e.g. `--dataset_fraction 0.1`. Cannot be below `1.0` with `--dataset_sample_count`. |
| `--dataset_sample_count N` | all | Evaluate exactly `N` randomly selected rows; e.g. `--dataset_sample_count 100`. |
| `--samples_over SECONDS` | off | Keep samples strictly longer than the threshold after other selection; e.g. `--samples_over 30`. |
| `--checkpoint EPOCH` | automatic | Select `checkpoint-epoch=XXXX.ckpt`; e.g. `--checkpoint 7`. Local runs only. |
| `--offline_whisper` | off | Evaluate standard, non-streaming Whisper using the model name/path in `--model`. |
| `--cw` | off | Load a CarelessWhisper base model instead of a local run; e.g. `--cw --model small`. Legacy `-cw` is accepted. |
| `--force_hf_download` | off | Force a fresh Hugging Face download. Requires `--cw` and bypasses the evaluation cache. |
| `--device DEVICE` | CUDA if available, else CPU | Inference device; e.g. `--device cuda`. |
| `--multilingual` | off | Use the multilingual model variant. |
| `--lang CODE` | inferred | Transcription and normalization language; e.g. `--lang de`. |
| `--chunk_size N` | `300` | Streaming chunk granularity in milliseconds; e.g. `--chunk_size 40`. |
| `--beam_size N` | `5` | Number of decoding beams; e.g. `--beam_size 10`. |
| `--enable_relative_beam_stop` | off | Require at least 20% of beams to emit EOS rather than stopping after the first EOS beam. |
| `--max_sec_context N` | `30` | Retained audio context in seconds. Legacy mode resets here; sliding mode prunes to this window. |
| `--encoder_positional_mode MODE` | `auto` | `auto`, `sinusoidal`, or `alibi`. Auto uses checkpoint/config metadata and then the run name. |
| `--sa_kv_cache` | off | Enable decoder self-attention KV caching. Legacy `-sa_kv_cache` is accepted. |
| `--ca_kv_cache` | off | Enable decoder cross-attention KV caching. Legacy `-ca_kv_cache` is accepted. |
| `--use_sliding_encoder_cache` | off | Prune old encoder state instead of performing a full reset. Requires an ALiBi encoder and single-frame streaming mel. |
| `--disable_encoder_kv_cache` | off | Recompute the full encoder prefix at every step as an uncached baseline. Incompatible with sliding encoder caching. |
| `--encoder_cache_diagnostics` | off | Compare cached features with a recomputed reference, print a final summary, and save it in the CSV. |
| `--encoder_cache_diagnostic_interval N` | `1` | Sample parity every `N` decode chunks; e.g. `--encoder_cache_diagnostic_interval 10`. |
| `--reset_decoder_on_encoder_slide` | off | Roll the decoder prefix after encoder pruning. Requires `--use_sliding_encoder_cache`. |
| `--decoder_roll_overlap_seconds S` | `5.0` | Encoder-audio overlap retained before the active decoder prefix; e.g. `4`. Must be below `--max_sec_context`. |
| `--decoder_roll_min_interval_seconds S` | `2.0` | Minimum seconds between rolls; e.g. `3`. |
| `--decoder_roll_max_prefix_tokens N` | `48` | Maximum recent BPE tokens retained after a roll; e.g. `64`. |
| `--decoder_token_time_lag_seconds S` | `2.0` | Backdate first-seen token times when deciding which tokens to retire; e.g. `1.5`. |
| `--decoder_roll_diagnostics` | off | Deprecated compatibility option; currently a no-op. |
| `--strict_k K...` | `2` | Strict-WER correction distances; e.g. `--strict_k 0 1 2`. |
| `--wir_n N...` | none | Extra word-instability suffix tolerances; e.g. `--wir_n 0 1 2`. |
| `--prefix_wer` | off | Save cumulative WER at two-second audio-prefix intervals from 0 through 60 seconds. |
| `--delay_n_rtf` | off | Compare normal visual-emission RTF/latency with a policy that delays the newest word, with a one-second timeout. |
| `--no_evaluation_cache` | off | Recalculate transcriptions even if a compatible cache exists. |
| `--verbose` | off | Print additional per-sample information. Legacy `-verbose` is accepted. |

### Printing saved evaluations

Print the three newest rows from the default table:

```bash
python evaluation_print.py
```

Print the newest ten rows from another table:

```bash
python evaluation_print.py \
  --evaluation_file /path/to/evaluation.csv \
  --row_count 10
```

| Parameter | Default | Meaning and example |
|---|---:|---|
| `-h`, `--help` | — | Print CLI help and exit. |
| `--evaluation_file PATH` | `$HOME/ma/data/evaluation.csv` | CSV to read; e.g. `--evaluation_file results/evaluation.csv`. |
| `--row_count N` | `3` | Newest rows to print, newest first; e.g. `--row_count 10`. Must be positive. |

Cache-parity fields are printed only when `--encoder_cache_diagnostics` was
enabled for that row. Decoder-roll settings are printed only when
`--reset_decoder_on_encoder_slide` was enabled.

## 🦾 Training
In order to train using LoRA, you can use our existing code. Make sure all the requirements are installed. 

### 📂 Dataset Structure

Before starting model training using the command-line interface provided below, you must first configure your dataset dictionary file located at `training_code/ds_dict.py`.

This file defines a Python dictionary named `ds_paths`, where you should specify paths to the `train`, `val`, and `test` partitions of your dataset. Each partition should be a CSV file with the following three columns:

1. `wav_path` — Path to the WAV audio file.  
2. `tg_path` — Path to the corresponding `.TextGrid` file containing forced alignment.  
3. `raw_text` — Ground truth transcription.

> **Note:** The dictionary key (i.e., the name of the dataset) will be used by the training script to identify and load the dataset correctly.

You can find an example entry in `training_code/ds_dict.py`.

> **Note:** We used [Montreal Forced Aligner (MFA)](https://montreal-forced-aligner.readthedocs.io/en/latest/index.html) to force-align our dataset.

To run the same force-alignment process as described in the paper, use:

```bash
mfa align --clean /dataset/root/path english_us_arpa english_us_arpa /aligned_dataset/root/path
```

For more details on how to run using `mfa` command, visit [MFA site](https://montreal-forced-aligner.readthedocs.io/en/latest/index.html).

### 🖥️ CLI Interface
```bash
python training_code/train.py \
--lora \
--streaming_train \
--simulate_stream \
--dataset LIBRI-960-ALIGNED \
--name example_training_base_model \
--size base \
--batch_size 32 \
--epochs 10 \
--learning_rate 1e-5 \
--rank 32 \
--gran 15 \
--extra_gran_blocks 1 \
--streaming_fraction 0.25 \
--top_k 5 \
```

The current `train.py` implementation supports the combined LoRA and streaming
training path, so normal runs must specify both `--lora` and
`--streaming_train`. Full streaming training also requires
`--simulate_stream`.

### New training configurations

Use different LoRA capacities for the encoder and decoder while retaining
`--rank` as the fallback for either omitted value:

```bash
python training_code/train.py \
  --lora \
  --streaming_train \
  --simulate_stream \
  --dataset LIBRI-960-ALIGNED \
  --name split_rank_alibi \
  --size turbo \
  --rank 16 \
  --encoder_rank 8 \
  --decoder_rank 32 \
  --encoder_positional_mode alibi \
  --gran 15
```

Train on the stale hidden-state regime produced by sliding encoder-cache
inference:

```bash
python training_code/train.py \
  --lora \
  --streaming_train \
  --simulate_stream \
  --dataset REVLONG \
  --name stale_cache_alibi \
  --size turbo \
  --encoder_positional_mode alibi \
  --stale_encoder_cache_train \
  --stale_cache_context_seconds 25 \
  --stale_cache_max_stale_seconds 30 \
  --stale_cache_fresh_fraction 0.2 \
  --stale_cache_bucket_weights 1,2,4 \
  --streaming_fraction 0.25
```

Stale-cache training mixes fresh retained-window recomputations with points
whose encoder state is assembled by simulating chunked KV-cache inference.
It requires ALiBi and cannot be combined with `--random_masking`.

Resume a run, including Lightning optimizer and trainer state:

```bash
python training_code/train.py \
  --lora --streaming_train --simulate_stream \
  --dataset LIBRI-960-ALIGNED \
  --name resumed_run \
  --ckpt previous_run
```

`--ckpt` accepts either a checkpoint file or a run name, in which case the
latest epoch checkpoint is selected. To copy model weights but start a new
optimizer/trainer history, use `--warmstart previous_run` instead.

### Training parameter reference

| Parameter | Default | Meaning and example |
|---|---:|---|
| `-h`, `--help` | — | Print the generated training CLI help and exit. |
| `--lora` | off | Enable LoRA training. Required by the currently implemented training path. |
| `--streaming_train` | off | Train at sequential streaming sample points. Requires `--simulate_stream` and, in the current path, `--lora`. |
| `--simulate_stream` | off | Supply streamed spectrogram prefixes rather than ordinary full inputs. |
| `--name NAME` | `model` | Output run name below `$HOME/ma/data/models`; e.g. `--name turbo_alibi_r16`. |
| `--size SIZE` | `tiny` | Whisper base architecture, such as `tiny`, `base`, `small`, `medium`, `large-v2`, or `turbo`. |
| `--lang CODE` | `en` | Whisper language token and normalization language; e.g. `--lang de`. |
| `--multilingual` | off | Expect multilingual data and use each row's `lang` field when available. |
| `--dataset NAME...` | `TIMIT-WORD` | One or more keys from `training_code/ds_dict.py`; e.g. `--dataset LIBRI-960-ALIGNED REVLONG`. Train and validation paths are combined. |
| `--custom_len N` | `0` | Limit each dataset to `N` samples; `0` uses its complete length. Useful for short experiments. |
| `--lmdb` | off | Use configured `train-lmdb` dataset entries where the loader supports LMDB instead of ordinary disk reads. |
| `--precomputed_features` | off | Read manifests produced by `utils/precompute_aligned_dataset.py`; dataset entries must provide `precomputed` train/val paths. |
| `--epochs N` | `10` | Maximum training epochs; e.g. `--epochs 20`. |
| `--max_training_time S` | unlimited | Wall-clock budget in seconds, starting before baseline validation and checked between epochs; e.g. `--max_training_time 21600`. Must be positive. |
| `--batch_size N` | `16` | Per-step training and validation batch size; e.g. `--batch_size 32`. |
| `--gacc N` | `1` | Lightning gradient-accumulation steps. Effective batch size is `batch_size × gacc`. |
| `--learning_rate LR` | `0.0001` | Adam learning rate; e.g. `--learning_rate 1e-5`. |
| `--weight_decay W` | `0.01` | Adam weight-decay factor. |
| `--adam_epsilon E` | `1e-6` | Adam numerical-stability epsilon. |
| `--warmup_steps N` | `100` | Scheduler warm-up steps. |
| `--precision MODE` | `16` | Lightning precision setting; examples include `16`, `32`, or a supported mixed-precision mode. |
| `--num_worker N` | `16` | DataLoader worker processes per loader; use `0` for synchronous loading. |
| `--strategy NAME` | `ddp` | Lightning distributed strategy, such as `ddp`, `fsdp`, or `ddp_find_unused_parameters_true`. |
| `--fast_dev_run N` | off | Ask Lightning to run only `N` development batches for a pipeline sanity check. |
| `--no_logger` | off | Disable the W&B/Lightning logger. |
| `--top_k N` | `1` | Number of lowest-validation-WER epoch checkpoints to keep. `-1` keeps all. Step checkpoints are also saved every 500 training steps. |
| `--early_stop` | off | Stop after validation WER fails to improve for two validation checks. |
| `--ckpt PATH_OR_RUN` | none | Resume complete Lightning training state from a file or the latest epoch of a named local run. |
| `--warmstart RUN` | none | Copy weights from the latest checkpoint of another run, but start new optimizer and trainer state. |
| `--use_from_ft_ckpt` | off | Initialize through the packaged fine-tuned streaming-checkpoint loader instead of constructing from the base Whisper training loader. |
| `--save_untrained` | off | Save the freshly initialized or warm-started model as `checkpoint-epoch=-001.ckpt` and exit without training. |
| `--rank N` | `16` | Global LoRA rank and fallback for both model halves; e.g. `--rank 32`. |
| `--encoder_rank N` | `--rank` | Encoder LoRA rank override; e.g. `--encoder_rank 8`. |
| `--decoder_rank N` | `--rank` | Decoder self- and cross-attention LoRA rank override; e.g. `--decoder_rank 32`. Resolved ranks are stored in new checkpoints. |
| `--lora_ckpt PATH` | none | Legacy LoRA-checkpoint argument. It is normalized and stored in configuration but is not currently applied by `train.py`; use `--ckpt` or `--warmstart` instead. |
| `--gran N` | `15` | Encoder attention granularity in 20 ms encoder frames. `15` corresponds to a 300 ms chunk. |
| `--extra_gran_blocks N` | `1` | Extra causal encoder blocks initially visible as look-ahead/initialization context. |
| `--streaming_fraction F` | `1.0` | Fraction of eligible streaming positions trained per sample; e.g. `--streaming_fraction 0.25`. |
| `--streaming_random` | off | Randomize selected streaming positions instead of processing the chosen points sequentially. |
| `--random_masking` | off | Train with randomly sized causal masking blocks instead of regular fixed-granularity stream points. Incompatible with stale-cache training. |
| `--num_slices N` | `20` | Number of random-mask sample points/slices considered when `--random_masking` is enabled. |
| `--encoder_positional_mode MODE` | `sinusoidal` | Encoder positions: `sinusoidal` or `alibi`. ALiBi is required for stale sliding-cache training. |
| `--stale_encoder_cache_train` | off | Mix fresh retained-window points with encoder features created through a simulated sliding KV cache. |
| `--stale_cache_context_seconds S` | `25.0` | Retained encoder window for stale training, rounded down to `--gran`; it must not exceed Whisper's 30-second audio context. |
| `--stale_cache_max_stale_seconds S` | `30.0` | Additional audio duration beyond the retained window available for selecting increasingly stale points. Must be non-negative. |
| `--stale_cache_fresh_fraction F` | `0.2` | Requested fraction of selected points encoded as fresh retained windows; values are clamped to `[0,1]`. |
| `--stale_cache_bucket_weights CSV` | `1` | Relative sampling weights for equal-width staleness buckets; e.g. `1,2,4` favors older states. Values must be non-negative with at least one positive weight. |
| `--self_supervision` | off | Replace available teacher labels with the decoder's current predictions up to predicted EOT in the regular streaming loss path. |
| `--extra_eval` | off | Calculate RWER and ARWER during training validation in addition to WER. |

The script writes `cfg.json`, W&B logs unless disabled, WER-ranked epoch
checkpoints, and periodic step checkpoints below the run's model directories.
It performs a full baseline validation before fresh or resumed training. The
`--max_training_time` budget includes that baseline validation but excludes
model construction, warm-start loading, and trainer initialization.

For more options and training configurations, run:
```bash
python training_code/train.py --help
```

## 📜 License

This repository uses a dual license:

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)  
Portions derived from [OpenAI Whisper](https://github.com/openai/whisper) are licensed under the **MIT License**.  

[![CC BY-NC 4.0 License](https://img.shields.io/badge/License-CC--BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)  
All other original code in this repository is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.  

See the [LICENSE](./LICENSE) file for full details.
