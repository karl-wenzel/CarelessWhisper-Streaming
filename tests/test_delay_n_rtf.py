import unittest
from types import SimpleNamespace

from evaluation import calculate_delay_n_display_stats, calculate_normal_display_stats


def _identity_normalizer(text):
    return text


def _result(text, processing_time):
    return SimpleNamespace(full_text=text, processing_time=processing_time)


class DelayNDisplayStatsTests(unittest.TestCase):
    def test_normal_emission_shows_all_appended_words_at_result_availability(self):
        stats = calculate_normal_display_stats(
            [
                _result("hello", 0.1),
                _result("hello wide world", 0.3),
            ],
            audio_duration=2.0,
            chunk_duration_sec=1.0,
            normalizer=_identity_normalizer,
        )

        self.assertEqual(3, stats["emitted_words"])
        self.assertAlmostEqual((0.1 + 0.3 + 0.3) / 3.0, stats["latency_sum_sec"] / 3.0)
        self.assertAlmostEqual(0.2, stats["rtf"])

    def test_delays_last_word_until_next_append_when_append_is_within_timeout(self):
        stats = calculate_delay_n_display_stats(
            [
                _result("hello", 0.1),
                _result("hello world", 0.1),
            ],
            audio_duration=2.0,
            chunk_duration_sec=1.0,
            normalizer=_identity_normalizer,
        )

        self.assertEqual(2, stats["emitted_words"])
        self.assertAlmostEqual(1.1, stats["latency_sum_sec"] / stats["emitted_words"])
        self.assertAlmostEqual(0.6, stats["rtf"])

    def test_emits_held_word_after_timeout_during_silence(self):
        stats = calculate_delay_n_display_stats(
            [
                _result("hello", 0.2),
                _result("hello", 0.2),
                _result("hello world", 0.2),
            ],
            audio_duration=3.0,
            chunk_duration_sec=1.0,
            normalizer=_identity_normalizer,
        )

        self.assertEqual(2, stats["emitted_words"])
        self.assertAlmostEqual(1.2, stats["latency_sum_sec"] / stats["emitted_words"])
        self.assertAlmostEqual(1.6 / 3.0, stats["rtf"])

    def test_immediately_emits_all_but_current_trailing_word(self):
        stats = calculate_delay_n_display_stats(
            [_result("one two three", 0.25)],
            audio_duration=1.0,
            chunk_duration_sec=1.0,
            normalizer=_identity_normalizer,
        )

        self.assertEqual(3, stats["emitted_words"])
        self.assertAlmostEqual((0.25 + 0.25 + 1.25) / 3.0, stats["latency_sum_sec"] / 3.0)
        self.assertAlmostEqual(1.25, stats["rtf"])


if __name__ == "__main__":
    unittest.main()
