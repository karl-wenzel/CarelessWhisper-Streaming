import ast
import unittest
from pathlib import Path

from evaluation.caching import (
    CACHE_CONTROL_PARAMETER_NAMES,
    EVALUATION_ONLY_PARAMETER_NAMES,
    PRE_EVALUATION_PARAMETER_NAMES,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_PY = REPO_ROOT / "evaluation.py"


def _literal_string(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _argument_dest(call: ast.Call) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == "dest":
            return _literal_string(keyword.value)

    option_names = [
        option
        for option in (_literal_string(arg) for arg in call.args)
        if option and option.startswith("-")
    ]
    if not option_names:
        return None

    long_options = [option for option in option_names if option.startswith("--")]
    selected = long_options[0] if long_options else option_names[0]
    return selected.lstrip("-").replace("-", "_")


def _evaluation_argparse_destinations() -> set[str]:
    tree = ast.parse(EVALUATION_PY.read_text(encoding="utf-8"))
    destinations = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue

        destination = _argument_dest(node)
        if destination:
            destinations.add(destination)

    return destinations


class EvaluationCachingContractTests(unittest.TestCase):
    def test_all_evaluation_parameters_are_explicitly_classified_for_caching(self):
        classified_parameters = set(PRE_EVALUATION_PARAMETER_NAMES)
        classified_parameters.update(EVALUATION_ONLY_PARAMETER_NAMES)
        classified_parameters.update(CACHE_CONTROL_PARAMETER_NAMES)

        missing = sorted(_evaluation_argparse_destinations() - classified_parameters)

        self.assertEqual(
            [],
            missing,
            "Every evaluation.py argparse parameter must be classified in "
            "evaluation/caching.py as pre-evaluation, evaluation-only, or cache-control.",
        )


if __name__ == "__main__":
    unittest.main()
