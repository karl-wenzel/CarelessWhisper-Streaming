import sys
sys.path.append('.')

import time

import argparse
from typing import TYPE_CHECKING, List

import numpy as np
import torch
import time

from .audio import (
    SAMPLE_RATE,
    SpectrogramStream,
    MyStream
)
from .streaming_decoding import DecodingOptions

if TYPE_CHECKING:
    from .streaming_model import StreamingWhisper

class ChunkResultWrapper:
    """
    Wraps the decode result to attach processing latency 
    without breaking existing attribute access
    """
    def __init__(self, original_result, processing_time: float):
        self._original = original_result
        self.processing_time = processing_time
        
    def __getattr__(self, item):
        return getattr(self._original, item)


def _append_with_word_overlap(prefix: str, suffix: str, max_overlap_words: int = 64) -> str:
    prefix_words = str(prefix or "").strip().split()
    suffix_words = str(suffix or "").strip().split()
    if not prefix_words:
        return " ".join(suffix_words)
    if not suffix_words:
        return " ".join(prefix_words)

    max_overlap = min(max_overlap_words, len(prefix_words), len(suffix_words))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if prefix_words[-size:] == suffix_words[:size]:
            overlap = size
            break

    return " ".join(prefix_words + suffix_words[overlap:])
        
def transcribe(
    model: "StreamingWhisper" = None,
    output_filename: str = None,
    channels: int = 2,
    language: str = "en",
    simulate_stream: bool = False,
    wav_file: str = None,
    single_frame_mel: bool = True,
    temperature: float = 0,
    beam_size: int = 5,
    stream_decode: bool = True,
    ca_kv_cache: bool = False,
    sa_kv_cache: bool = False,
    use_latency: bool = False,
    get_times: bool = False,
    pad_trim: bool = False,
    max_sec_context: int = 30,
    use_sliding_encoder_cache: bool = False,
    disable_encoder_kv_cache: bool = False,
    reset_decoder_on_encoder_slide: bool = False,
    streaming_timestamps: bool = False,
    force_first_tokens_timestamps: bool = False,
    verbose: bool = True,
    ms_granularity: int = None,
    extra_initial_blocks: int = None,
    **kwargs
) -> List[str]:
    """
    Open a stream and transcribe it using streaming whisper model

    A very thin implementation of the transcribe function, compared to Whisper implementation.

    Parameters
    ----------
    model: Whisper
        The Whisper model instance

    Returns - 
    -------
    A list of ChunkResultWrapper objects containing the text, tokens, 
    and the processing_time field.
    """
    model.reset(use_stream=True) # we first reset the model before starting a stream, cleaning any cache.
    model.eval()
    
    # Instantiate streaming instance and open a stream
    ms_gran = model.encoder.gran * 20 if ms_granularity is None else ms_granularity
    assert ms_gran % 20 == 0, "ms_granularity must be a multiple of 20"
    stream_instance = MyStream(ms_gran,
                               channels=channels,
                               filename=output_filename, 
                               simulate_stream=simulate_stream, 
                               wav_file=wav_file, 
                               use_latency=use_latency, 
                               pad_trim=pad_trim,
                               verbose=verbose)
    
    stream_instance.open_stream()
    
    # frames - used only when filename is given, in order to save a long wav at the end of the conversation.
    frames = []

    extra_gran_blocks = extra_initial_blocks if extra_initial_blocks is not None else model.encoder.extra_gran_blocks

    # first we'll use
    decoding_options = DecodingOptions(
        language=language,
        gran=(ms_gran // 20),
        single_frame_mel=single_frame_mel,
        without_timestamps=True,
        beam_size=beam_size if temperature == 0 else None,
        temperature=temperature,
        length_penalty=None,
        look_ahead_blocks=extra_gran_blocks,
        patience=None,
        stream_decode=stream_decode,
        use_kv_cache=sa_kv_cache,
        use_ca_kv_cache=ca_kv_cache,
        maximal_seconds_context=max_sec_context,
        use_sliding_encoder_cache=use_sliding_encoder_cache,
        disable_encoder_kv_cache=disable_encoder_kv_cache,
        reset_decoder_on_encoder_slide=reset_decoder_on_encoder_slide,
        streaming_timestamps=streaming_timestamps,
        force_first_tokens_timestamps=force_first_tokens_timestamps,
        verbose=verbose,
        **kwargs
    )

    streamed_spectrogram = SpectrogramStream(n_mels=model.dims.n_mels) # default values are whisper default values

    texts = []
    times = []
    reset_len = (max_sec_context) * SAMPLE_RATE + 360 # 360 is for the mel padding
    chunk_samples = stream_instance.chunk_size
    full_text = ""
    try:
        stream_iter = iter(stream_instance.read())
        try:
            frame = next(stream_iter)
        except StopIteration:
            return []

        while True:
            try:
                next_frame = next(stream_iter)
                is_last = False
            except StopIteration:
                is_last = True

            # save frames for optional save
            frames.extend(frame)
            # Legacy mode resets at max context; sliding cache mode keeps this
            # stream alive and lets DecodingTask prune encoder-side state.
            if (not use_sliding_encoder_cache) and len(frames) >= reset_len:
                frame = np.concatenate((frames[-360:], frame))
                frames = []
                frames.extend(frame.tolist())
                model.reset(use_stream=True)
                streamed_spectrogram.reset()
                if len(texts) > 0:
                    full_text = _append_with_word_overlap(full_text, texts[-1].text)

            if get_times:
                torch.cuda.synchronize()
                start = time.time()

            frame_tensor = torch.from_numpy(frame).pin_memory()

            chunk_start_time = time.perf_counter()
            
            mel_frame = streamed_spectrogram.calc_mel_with_new_frame(frame_tensor.to(model.device, non_blocking=True), )

            # decode given the new mel frame and print results
            result = model.decode(mel_frame.squeeze(0), decoding_options)
            if getattr(result, "decoder_rebased", False) and len(texts) > 0:
                full_text = _append_with_word_overlap(full_text, texts[-1].text)
            # Long-form simulated streams can cross the legacy 30s reset boundary.
            # Keep the accumulated transcript on the result so evaluators can score
            # the whole sample instead of only the current post-reset window.
            result.full_text = _append_with_word_overlap(full_text, result.text)
            
            chunk_end_time = time.perf_counter()
            processing_latency = chunk_end_time - chunk_start_time
            wrapped_result = ChunkResultWrapper(result, processing_latency)

            if (verbose):
                print(f"{wrapped_result.text} (Latency: {processing_latency:.3f}s)")
            
            texts.append(wrapped_result)

            if is_last:
                break
            frame = next_frame

    except KeyboardInterrupt:
        stream_instance.close_stream(frames)
    
    if (verbose):
        print("Finished capturing audio.")
    
    return texts


def cli():
    parser = argparse.ArgumentParser(description="Transcribe streaming audio with customizable options")

    # Model choices
    parser.add_argument("--model", type=str, default="small", help="Model size to transcribe with")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run model inference on.")
    parser.add_argument("--chunk_size", type=int, default=300, help="Chunk size for streaming")
    parser.add_argument("--multilingual", action="store_true", help="Use a multilingual checkpoint if exists.", default=False)
    parser.add_argument("--local_model_path", type=str, default=None, help="Load the model ckpt from a local path. Overrides model, chunk_size and multilingual settings.")

    # Local streaming args
    parser.add_argument("--output_filename", type=str, help="Path to the output audio file when using local streaming")
    parser.add_argument("--channels", type=int, default=2, help="Number of audio channels - relevant for local streaming")

    # Streaming simulation wav file
    parser.add_argument("--wav_file", type=str, help="Optional WAV file path to stream, using a stream simulation")

    # Streaming behavior
    parser.add_argument("--simulate_stream", action="store_true", help="Simulate a stream from a file")
    parser.add_argument("--single_frame_mel", action="store_true", default=True, help="Use single frame MELs")
    parser.add_argument("--stream_decode", action="store_true", default=True, help="Use streaming decode")
    parser.add_argument("--ca_kv_cache", action="store_true", help="Use cross-attention key-value cache")
    parser.add_argument("--sa_kv_cache", action="store_true", help="Use self-attention key-value cache")
    parser.add_argument("--wait_for_all", action="store_true", help="Wait for all results before outputting")
    parser.add_argument("--use_latency", action="store_true", help="Add latency for streaming simulation")
    parser.add_argument("--pad_trim", action="store_true", default=False, help="Enable padding and trimming")
    parser.add_argument("--streaming_timestamps", action="store_true", help="Use timestamps in streaming")
    parser.add_argument("--force_first_tokens_timestamps", action="store_true", help="Force timestamps on first tokens")

    # Model behavior
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search decoding")
    parser.add_argument("--language", type=str, default="en", help="Language of transcription")
    parser.add_argument("--max_sec_context", type=int, default=30, help="Max context window size in seconds")
    parser.add_argument("--use_sliding_encoder_cache", action="store_true", help="Slide encoder KV cache instead of resetting at max context")
    parser.add_argument("--disable_encoder_kv_cache", action="store_true", help="Recompute full encoder prefix instead of using encoder KV cache")
    parser.add_argument("--reset_decoder_on_encoder_slide", action="store_true", help="Rebase decoder state when sliding encoder cache crosses a reset boundary")

    args = parser.parse_args().__dict__

    from . import load_streaming_model

    model_size: str = args.pop("model")
    chunk_size: int = args.pop("chunk_size")
    multilingual: bool = args.pop("multilingual")
    device: str = args.pop("device")
    local_ckpt_path: str = args.pop("local_model_path")

    model = load_streaming_model(model_size, chunk_size, multilingual, device, local_ckpt_path)

    texts = transcribe(model, **args)
    return texts

if __name__ == "__main__":
    cli()

