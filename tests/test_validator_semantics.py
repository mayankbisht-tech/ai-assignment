"""tests/test_validator_semantics.py — Validates semantic constraints on outputs."""

import json
import unittest
from pathlib import Path
from app import load_data, validate_itinerary

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"


class TestValidatorSemantics(unittest.TestCase):
    def setUp(self):
        self.data = load_data()

    def test_chat_output_semantics(self):
        path = OUTPUTS_DIR / "CHAT.json"
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        itin = payload.get("itinerary")
        if itin and itin.get("status") == "fulfilled":
            errors = validate_itinerary(itin, self.data)
            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
