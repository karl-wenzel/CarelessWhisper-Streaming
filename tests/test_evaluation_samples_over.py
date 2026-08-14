import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd

from evaluation import _filter_samples_over_duration


class EvaluationSamplesOverTests(unittest.TestCase):
    def test_keeps_only_samples_strictly_longer_than_threshold(self):
        selected_rows = pd.DataFrame(
            {"wav_path": ["short.wav", "boundary.wav", "long.wav"]},
            index=[4, 7, 9],
        )
        durations = {"short.wav": 4.9, "boundary.wav": 5.0, "long.wav": 5.1}

        with patch("evaluation.runner.librosa.get_duration", side_effect=lambda *, path: durations[Path(path).name]):
            filtered = _filter_samples_over_duration(selected_rows, ".", 5)

        self.assertEqual(filtered["wav_path"].tolist(), ["long.wav"])
        self.assertEqual(filtered.index.tolist(), [0])

    def test_resolves_relative_wav_paths_from_csv_directory(self):
        selected_rows = pd.DataFrame({"wav_path": ["audio/sample.wav"]})

        with patch("evaluation.runner.librosa.get_duration", return_value=11) as get_duration:
            _filter_samples_over_duration(selected_rows, "dataset/test.csv", 10)

        self.assertTrue(get_duration.call_args.kwargs["path"].endswith("dataset\\audio\\sample.wav"))


if __name__ == "__main__":
    unittest.main()
