import unittest

import torch

from careless_whisper_stream.streaming_decoding import BeamStreamingDecoder


class RelativeBeamStopTests(unittest.TestCase):
    @staticmethod
    def _decoder(n_beams: int, enabled: bool) -> BeamStreamingDecoder:
        decoder = BeamStreamingDecoder.__new__(BeamStreamingDecoder)
        decoder.n_beams = n_beams
        decoder.eot = 99
        decoder.enable_relative_beam_stop = enabled
        return decoder

    def test_default_mode_stops_after_one_eot_beam(self):
        decoder = self._decoder(n_beams=10, enabled=False)
        sequences = [torch.tensor([1, 99])] + [torch.tensor([1, 2])] * 9

        self.assertTrue(decoder._has_reached_eot_threshold(sequences))

    def test_relative_mode_requires_twenty_percent_of_beams(self):
        decoder = self._decoder(n_beams=10, enabled=True)

        self.assertFalse(
            decoder._has_reached_eot_threshold(
                [torch.tensor([1, 99])] + [torch.tensor([1, 2])] * 9
            )
        )
        self.assertTrue(
            decoder._has_reached_eot_threshold(
                [torch.tensor([1, 99])] * 2 + [torch.tensor([1, 2])] * 8
            )
        )

    def test_relative_threshold_rounds_up_for_partial_fifths(self):
        decoder = self._decoder(n_beams=6, enabled=True)

        self.assertFalse(
            decoder._has_reached_eot_threshold(
                [torch.tensor([99])] + [torch.tensor([1])] * 5
            )
        )
        self.assertTrue(
            decoder._has_reached_eot_threshold(
                [torch.tensor([99])] * 2 + [torch.tensor([1])] * 4
            )
        )


if __name__ == "__main__":
    unittest.main()
