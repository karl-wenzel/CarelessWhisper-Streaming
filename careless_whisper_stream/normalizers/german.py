import re

from .basic import remove_symbols
from german_transliterate.core import GermanTransliterate

class GermanTextNormalizer:
    def __init__(self):
        self._transliterate_ops = [
            "acronym_phoneme",
            "accent_peculiarity",
            "amount_money",
            "date",
            "timestamp",
            "time_of_day",
            "ordinal",
            "special",
        ]
        self.standardize_text = self._build_transliterator(self._transliterate_ops)
        self.standardize_text_no_ordinal = self._build_transliterator(
            [op for op in self._transliterate_ops if op != "ordinal"]
        )

    def _build_transliterator(self, transliterate_ops):
        return GermanTransliterate(
            transliterate_ops=transliterate_ops,
            replace={"-": " "},
            sep_abbreviation=" ",
            make_lowercase=True,
        )

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
        s = re.sub(r"\(([^)]+?)\)", "", s)
        try:
            s = self.standardize_text.transliterate(s)
        except Exception:
            # Model hypotheses can contain malformed ordinal-like fragments
            # (for example partial tokens ending in a period). Retry without
            # ordinal expansion so validation/evaluation can continue.
            try:
                s = self.standardize_text_no_ordinal.transliterate(s)
            except Exception:
                pass
        s = remove_symbols(s).lower()
        s = re.sub(r"\s+", " ", s)

        return s.strip()
