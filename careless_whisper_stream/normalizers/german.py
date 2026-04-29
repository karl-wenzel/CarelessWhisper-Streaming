import re

from .basic import remove_symbols
from german_transliterate.core import GermanTransliterate

class GermanTextNormalizer:
    def __init__(self):
        self.standardize_text = GermanTransliterate(
            transliterate_ops=[
                "acronym_phoneme",
                "accent_peculiarity",
                "amount_money",
                "date",
                "timestamp",
                "time_of_day",
                "ordinal",
                "special",
            ],
            replace={"-": " "},
            sep_abbreviation=" ",
            make_lowercase=True,
        )

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
        s = re.sub(r"\(([^)]+?)\)", "", s)
        s = self.standardize_text.transliterate(s)
        s = remove_symbols(s).lower()
        s = re.sub(r"\s+", " ", s)

        return s.strip()
