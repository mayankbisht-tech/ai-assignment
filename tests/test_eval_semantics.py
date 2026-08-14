"""tests/test_eval_semantics.py — Validates evaluation harness integration."""

import unittest
from pathlib import Path
from eval import validate_output_file
from app import load_data

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"


class TestEvalSemantics(unittest.TestCase):
    def setUp(self):
        self.data = load_data()

    def test_eval_on_existing_outputs(self):
        files = list(OUTPUTS_DIR.glob("*.json"))
        for f in files:
            passed, errors = validate_output_file(f, self.data)
            self.assertTrue(passed, f"Validation failed for {f.name}: {errors}")


if __name__ == "__main__":
    unittest.main()
