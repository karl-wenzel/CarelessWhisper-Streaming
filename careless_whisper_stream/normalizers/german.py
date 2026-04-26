import re

from .basic import remove_symbols

try:
    from german_transliterate.core import GermanTransliterate
except ImportError as exc:
    GermanTransliterate = None
    _GERMAN_TRANSLITERATE_IMPORT_ERROR = exc
else:
    _GERMAN_TRANSLITERATE_IMPORT_ERROR = None


class GermanTextNormalizer:
    def __init__(self):
        if GermanTransliterate is None:
            raise ImportError(
                "GermanTextNormalizer requires the optional dependency "
                "'german-transliterate'. Install it with `pip install german-transliterate`."
            ) from _GERMAN_TRANSLITERATE_IMPORT_ERROR

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
