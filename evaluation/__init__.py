"""Reusable evaluation functionality with lazy access to the heavy runner."""

__all__ = [
    "_filter_samples_over_duration",
    "calculate_delay_n_display_stats",
    "calculate_word_instability_with_suffix_tolerance",
    "evaluate",
]


def __getattr__(name: str):
    # Keep lightweight consumers such as evaluation_print from importing model,
    # audio, and dataset dependencies merely to format saved CSV rows.
    if name in __all__:
        from . import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
