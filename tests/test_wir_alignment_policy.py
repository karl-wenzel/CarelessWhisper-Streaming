import unittest
from types import SimpleNamespace

from evaluation import calculate_word_instability_with_suffix_tolerance


def _identity_normalizer(text):
    return text


def _result(text):
    return SimpleNamespace(full_text=text)


class WirAlignmentPolicyTests(unittest.TestCase):
    def test_internal_replacement_and_insertion_counts_only_changed_region(self):
        changed_words, total_words = calculate_word_instability_with_suffix_tolerance(
            [
                _result("Deck is blue"),
                _result("the car is blue"),
            ],
            _identity_normalizer,
        )

        self.assertEqual(2, changed_words)
        self.assertEqual(4, total_words)

    def test_same_length_replacement_keeps_later_equal_words_stable(self):
        changed_words, total_words = calculate_word_instability_with_suffix_tolerance(
            [
                _result("deck A is blue"),
                _result("the car is blue"),
            ],
            _identity_normalizer,
        )

        self.assertEqual(2, changed_words)
        self.assertEqual(4, total_words)

    def test_append_only_streaming_growth_does_not_count_as_instability(self):
        changed_words, total_words = calculate_word_instability_with_suffix_tolerance(
            [
                _result("the car is"),
                _result("the car is blue"),
            ],
            _identity_normalizer,
        )

        self.assertEqual(0, changed_words)
        self.assertEqual(4, total_words)

    def test_suffix_tolerance_ignores_revisions_inside_previous_trailing_words(self):
        changed_words, total_words = calculate_word_instability_with_suffix_tolerance(
            [
                _result("the car is blue"),
                _result("the car was red"),
            ],
            _identity_normalizer,
            suffix_tolerance=2,
        )

        self.assertEqual(0, changed_words)
        self.assertEqual(4, total_words)


if __name__ == "__main__":
    unittest.main()
