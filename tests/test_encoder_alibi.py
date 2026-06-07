import sys
import unittest
from unittest.mock import patch

import torch

import careless_whisper_stream
from careless_whisper_stream.model import ModelDimensions
from careless_whisper_stream.streaming_model import EncoderCacheState, StreamingWhisper
from training_code.utils import Config, parse_cmdl


def tiny_dims(n_audio_ctx: int = 8) -> ModelDimensions:
    return ModelDimensions(
        n_mels=4,
        n_audio_ctx=n_audio_ctx,
        n_audio_state=16,
        n_audio_head=4,
        n_audio_layer=2,
        n_vocab=32,
        n_text_ctx=16,
        n_text_state=16,
        n_text_head=4,
        n_text_layer=2,
    )


class EncoderAlibiTests(unittest.TestCase):
    def test_default_positional_mode_is_sinusoidal(self):
        model = StreamingWhisper(tiny_dims(), gran=2, rank=2, extra_gran_blocks=1)
        mel = torch.randn(1, model.dims.n_mels, model.dims.n_audio_ctx * 2)

        out = model.encoder(mel, index=[0, model.dims.n_audio_ctx], mask=None)

        self.assertEqual(model.encoder_positional_mode, "sinusoidal")
        self.assertEqual(model.encoder.encoder_positional_mode, "sinusoidal")
        self.assertEqual(out.shape, (1, model.dims.n_audio_ctx, model.dims.n_audio_state))

    def test_alibi_encoder_forward_shape(self):
        model = StreamingWhisper(
            tiny_dims(),
            gran=2,
            rank=2,
            extra_gran_blocks=1,
            encoder_positional_mode="alibi",
        )
        mel = torch.randn(1, model.dims.n_mels, model.dims.n_audio_ctx * 2)

        out = model.encoder(mel, index=[0, model.dims.n_audio_ctx], mask=None)

        self.assertEqual(model.encoder.encoder_positional_mode, "alibi")
        self.assertEqual(out.shape, (1, model.dims.n_audio_ctx, model.dims.n_audio_state))

    def test_alibi_streaming_encoder_cache_can_extend_past_audio_context(self):
        model = StreamingWhisper(
            tiny_dims(n_audio_ctx=4),
            gran=2,
            rank=2,
            extra_gran_blocks=0,
            encoder_positional_mode="alibi",
        )
        model.encoder._use_stream(True)
        cache, hooks = model.install_encoder_kv_cache_hooks()

        try:
            for mel_frames in (4, 8, 12):
                mel = torch.randn(1, model.dims.n_mels, mel_frames)
                out = model.encoder(mel, kv_cache=cache, mask=None)
                self.assertEqual(out.shape, (1, model.gran, model.dims.n_audio_state))
        finally:
            for hook in hooks:
                hook.remove()

        cached_lengths = {
            value.shape[1]
            for value in cache.values()
            if torch.is_tensor(value)
        }
        self.assertEqual(cached_lengths, {6})

    def test_alibi_recomputed_cache_matches_full_prefix_masked_encoder(self):
        torch.manual_seed(0)
        model = StreamingWhisper(
            tiny_dims(n_audio_ctx=16),
            gran=5,
            rank=2,
            extra_gran_blocks=1,
            encoder_positional_mode="alibi",
        )
        model.eval()
        model.encoder._use_mask(True)
        mel = torch.randn(1, model.dims.n_mels, 30)
        overlap_frames = model.gran * (1 + model.extra_gran_blocks)

        full_prefix_chunks = []
        model.encoder._use_stream(False)
        for prefix_frames, new_frames in ((10, 10), (15, model.gran + overlap_frames)):
            full_out = model.encoder(
                mel[..., : prefix_frames * 2],
                index=[0, prefix_frames],
                mask=True,
            )
            full_prefix_chunks.append(full_out[:, -new_frames:])

        model.encoder._use_stream(True)
        cache, hooks = model.install_encoder_kv_cache_hooks()
        cached_chunks = []
        try:
            cached_chunks.append(model.encoder(mel[..., :20], kv_cache=cache, mask=None))
            model.prune_encoder_kv_cache_tail(cache, overlap_frames)
            original_gran = model.encoder.gran
            model.encoder.gran = original_gran + overlap_frames
            try:
                cached_chunks.append(model.encoder(mel[..., :30], kv_cache=cache, mask=True))
            finally:
                model.encoder.gran = original_gran
        finally:
            for hook in hooks:
                hook.remove()

        for cached, expected in zip(cached_chunks, full_prefix_chunks):
            self.assertTrue(
                torch.allclose(cached, expected, atol=1e-5),
                f"max diff={torch.max(torch.abs(cached - expected)).item()}",
            )

    def test_sinusoidal_recompute_after_full_initial_rollback_does_not_reapply_start_buffer(self):
        model = StreamingWhisper(
            tiny_dims(n_audio_ctx=64),
            gran=15,
            rank=2,
            extra_gran_blocks=1,
            encoder_positional_mode="sinusoidal",
        )
        model.encoder._use_stream(True)
        state = EncoderCacheState()
        cache, hooks = model.install_encoder_kv_cache_hooks(cache_state=state)
        mel = torch.randn(1, model.dims.n_mels, 92)

        try:
            initial = model.encoder(mel[..., :62], kv_cache=cache, mask=None)
            state.commit(initial.shape[1])

            overlap_frames = model.gran * (1 + model.extra_gran_blocks)
            model.prune_encoder_kv_cache_tail(cache, overlap_frames)
            state.cached_frames -= overlap_frames

            original_gran = model.encoder.gran
            model.encoder.gran = original_gran + overlap_frames
            try:
                recomputed = model.encoder(mel, kv_cache=cache, mask=True)
            finally:
                model.encoder.gran = original_gran
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(recomputed.shape[1], model.gran * (2 + model.extra_gran_blocks))

    def test_alibi_sliding_encoder_cache_prunes_kv_and_tracks_offsets(self):
        model = StreamingWhisper(
            tiny_dims(n_audio_ctx=4),
            gran=2,
            rank=2,
            extra_gran_blocks=0,
            encoder_positional_mode="alibi",
        )
        model.encoder._use_stream(True)
        state = EncoderCacheState(use_sliding=True, max_frames=4)
        cache, hooks = model.install_encoder_kv_cache_hooks(cache_state=state)
        mel_window = None

        try:
            for _ in range(4):
                mel_frame = torch.randn(1, model.dims.n_mels, model.gran * 2)
                mel_window = mel_frame if mel_window is None else torch.cat([mel_window, mel_frame], dim=-1)

                out = model.encoder(mel_window, kv_cache=cache, mask=None)
                frames_to_prune = state.commit(out.shape[1])
                model.prune_encoder_kv_cache(cache, frames_to_prune)

                if frames_to_prune > 0:
                    mel_window = mel_window[..., frames_to_prune * 2:]

                cached_lengths = {
                    value.shape[1]
                    for value in cache.values()
                    if torch.is_tensor(value)
                }
                self.assertLessEqual(max(cached_lengths), state.max_frames)
                self.assertEqual(cached_lengths, {state.cached_frames})
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(state.cached_frames, 4)
        self.assertEqual(state.cache_start_frame, 4)
        self.assertEqual(state.total_frames, 8)

    def test_alibi_sliding_recompute_builds_mask_for_transient_overlap(self):
        model = StreamingWhisper(
            tiny_dims(n_audio_ctx=4),
            gran=2,
            rank=2,
            extra_gran_blocks=0,
            encoder_positional_mode="alibi",
        )
        model.encoder._use_stream(True)
        state = EncoderCacheState(use_sliding=True, max_frames=4)
        cache, hooks = model.install_encoder_kv_cache_hooks(cache_state=state)

        try:
            mel = torch.randn(1, model.dims.n_mels, 12)
            first = model.encoder(mel[..., :4], kv_cache=cache, mask=None)
            state.commit(first.shape[1])
            second = model.encoder(mel[..., :8], kv_cache=cache, mask=None)
            state.commit(second.shape[1])

            overlap_frames = model.gran
            model.prune_encoder_kv_cache_tail(cache, overlap_frames)
            state.cached_frames -= overlap_frames

            original_gran = model.encoder.gran
            model.encoder.gran = original_gran + overlap_frames
            try:
                recomputed = model.encoder(mel, kv_cache=cache, mask=True)
            finally:
                model.encoder.gran = original_gran
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(recomputed.shape, (1, original_gran + overlap_frames, model.dims.n_audio_state))

    def test_invalid_encoder_positional_mode_raises(self):
        with self.assertRaisesRegex(ValueError, "encoder_positional_mode"):
            StreamingWhisper(tiny_dims(), encoder_positional_mode="rotary")

    def test_config_default_and_cli_switch(self):
        self.assertEqual(Config().encoder_positional_mode, "sinusoidal")

        with patch.object(sys, "argv", ["train.py", "--encoder_positional_mode", "alibi"]):
            args = parse_cmdl()

        self.assertEqual(args.encoder_positional_mode, "alibi")

    def test_local_checkpoint_preserves_alibi_mode_and_old_default(self):
        dims = tiny_dims()
        alibi_checkpoint = {
            "state_dict": {},
            "dims": vars(dims),
            "hyper_parameters": {
                "gran": 2,
                "rank": 2,
                "extra_gran_blocks": 0,
                "encoder_positional_mode": "alibi",
            },
        }
        old_checkpoint = {
            "state_dict": {},
            "dims": vars(dims),
            "hyper_parameters": {
                "gran": 2,
                "rank": 2,
                "extra_gran_blocks": 0,
            },
        }

        with patch("os.path.exists", return_value=True), patch(
            "torch.load",
            return_value=alibi_checkpoint,
        ):
            loaded = careless_whisper_stream.load_streaming_model(
                "tiny",
                device="cpu",
                local_ckpt_path="alibi.ckpt",
            )
        self.assertEqual(loaded.encoder.encoder_positional_mode, "alibi")

        with patch("os.path.exists", return_value=True), patch(
            "torch.load",
            return_value=old_checkpoint,
        ):
            loaded_old = careless_whisper_stream.load_streaming_model(
                "tiny",
                device="cpu",
                local_ckpt_path="old.ckpt",
            )
        self.assertEqual(loaded_old.encoder.encoder_positional_mode, "sinusoidal")

    def test_local_checkpoint_reads_nested_lightning_cfg(self):
        dims = tiny_dims()
        checkpoint = {
            "state_dict": {},
            "dims": vars(dims),
            "hyper_parameters": {
                "cfg": {
                    "gran": 2,
                    "rank": 2,
                    "extra_gran_blocks": 0,
                    "encoder_positional_mode": "alibi",
                }
            },
        }

        with patch("os.path.exists", return_value=True), patch(
            "torch.load",
            return_value=checkpoint,
        ):
            loaded = careless_whisper_stream.load_streaming_model(
                "tiny",
                device="cpu",
                local_ckpt_path="nested-lightning.ckpt",
            )

        self.assertEqual(loaded.encoder.encoder_positional_mode, "alibi")
        self.assertEqual(loaded.gran, 2)


if __name__ == "__main__":
    unittest.main()
