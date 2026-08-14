"""tests/test_conversation.py — Unit tests for multi-turn conversational planner."""

import unittest
from pathlib import Path

from app import (
    build_graph,
    create_initial_state,
    load_data,
    process_turn,
    validate_itinerary,
)


class TestConversationalPlanner(unittest.TestCase):
    def setUp(self):
        self.app = build_graph()
        self.data = load_data()

    def test_multi_turn_state_preservation(self):
        """Verify state is preserved across turns without requiring repetition."""
        state = create_initial_state()

        # Turn 1: Destination
        state, resp1 = process_turn(self.app, state, "I want to visit Munnar")
        self.assertEqual(state["destination"], "Munnar")
        self.assertIsNone(state["days"])
        self.assertIn("days", resp1.lower())

        # Turn 2: Days
        state, resp2 = process_turn(self.app, state, "4 days")
        self.assertEqual(state["destination"], "Munnar")
        self.assertEqual(state["days"], 4)
        self.assertIsNone(state["adults"])
        self.assertIn("people", resp2.lower())

        # Turn 3: Party
        state, resp3 = process_turn(self.app, state, "2 adults")
        self.assertEqual(state["destination"], "Munnar")
        self.assertEqual(state["days"], 4)
        self.assertEqual(state["adults"], 2)
        self.assertIn("experience", resp3.lower())

        # Turn 4: Preferences
        state, resp4 = process_turn(self.app, state, "hiking and nature")
        self.assertIn("hiking", state["preferences"])
        self.assertIn("budget", resp4.lower())

        # Turn 5: Budget
        state, resp5 = process_turn(self.app, state, "budget-conscious")
        self.assertEqual(state["status"], "fulfilled")
        self.assertIn("FINAL ITINERARY", resp5)
        self.assertIsNotNone(state["itinerary"])

    def test_natural_corrections(self):
        """Verify modifying one field does not erase other fields."""
        state = create_initial_state()

        # Set initial parameters all at once
        state, _ = process_turn(self.app, state, "Munnar for 4 days with 2 adults, hiking, budget 30k")
        self.assertEqual(state["destination"], "Munnar")
        self.assertEqual(state["days"], 4)
        self.assertEqual(state["adults"], 2)

        # Correction 1: Change days
        state, _ = process_turn(self.app, state, "Actually make it 3 days")
        self.assertEqual(state["destination"], "Munnar")
        self.assertEqual(state["days"], 3)
        self.assertEqual(state["adults"], 2)
        self.assertEqual(len(state["itinerary"]["days"]), 3)

        # Correction 2: Change destination
        state, _ = process_turn(self.app, state, "Actually Kochi")
        self.assertEqual(state["destination"], "Kochi")
        self.assertEqual(state["days"], 3)
        self.assertEqual(state["adults"], 2)
        self.assertEqual(state["itinerary"]["destination"], "Kochi")

    def test_out_of_catalog_destination_recovery(self):
        """Verify invalid destination like Goa is rejected without losing collected data."""
        state = create_initial_state()

        # User provides duration and party first
        state, _ = process_turn(self.app, state, "3 days for 2 adults")
        self.assertEqual(state["days"], 3)
        self.assertEqual(state["adults"], 2)

        # User asks for Goa
        state, resp = process_turn(self.app, state, "I want to go to Goa with nightlife")
        self.assertIsNone(state["destination"])
        self.assertIn("Goa", resp)
        self.assertIn("Available destinations are", resp)

        # Days and adults should be preserved!
        self.assertEqual(state["days"], 3)
        self.assertEqual(state["adults"], 2)

        # User selects valid destination + preferences & budget
        state, resp2 = process_turn(self.app, state, "Alleppey, relaxed pace, flexible budget")
        self.assertEqual(state["destination"], "Alleppey")
        self.assertEqual(state["status"], "fulfilled")
        self.assertIn("FINAL ITINERARY", resp2)
        self.assertEqual(len(state["itinerary"]["days"]), 3)

    def test_grounding_and_math_validation(self):
        """Verify final generated itinerary passes strict catalog grounding."""
        state = create_initial_state()
        state, _ = process_turn(self.app, state, "A weekend in Munnar for a couple, budget-conscious, hiking and tea estates")
        itin = state["itinerary"]
        self.assertIsNotNone(itin)
        errors = validate_itinerary(itin, self.data)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
