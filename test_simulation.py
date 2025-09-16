import unittest
import numpy as np
from persona import Persona
from simulation import calculate_influence, build_influence_matrix
from llm_integration import (
    construct_prompt,
    construct_distillation_prompt,
    construct_axis_generation_prompt,
    construct_vector_initialization_prompt,
    construct_statement_prompt
)

class TestSimulation(unittest.TestCase):

    def setUp(self):
        """Set up a dummy profile for all tests."""
        self.dummy_profile = "A test persona."
        self.dummy_axes = {"axis_1": "Community vs. Profit", "axis_2": "Regulation vs. Freedom"}
        self.dummy_axis_names = list(self.dummy_axes.values())

    def test_persona_creation_and_update(self):
        """Tests if a Persona object is created and updated correctly."""
        p = Persona(1, self.dummy_profile, mood=0.1)
        self.assertEqual(p.id, 1)
        self.assertEqual(p.attributes.size, 0) # Initially empty

        # Test the set_attributes method
        new_state = {self.dummy_axes["axis_1"]: -0.2, self.dummy_axes["axis_2"]: 0.3, "mood": -0.4}
        p.set_attributes(new_state, self.dummy_axis_names)
        np.testing.assert_array_equal(p.attributes, np.array([-0.2, 0.3]))
        self.assertAlmostEqual(p.mood, -0.4)

    def test_calculate_influence(self):
        """Tests the influence calculation between two personas."""
        p1 = Persona(1, self.dummy_profile); p1.attributes = np.array([0.5, 0.5])
        p2 = Persona(2, self.dummy_profile); p2.attributes = np.array([0.5, 0.5])
        self.assertAlmostEqual(calculate_influence(p1, p2), 1.0)

    def test_build_influence_matrix(self):
        """Tests that the sparse matrix is built correctly."""
        p1 = Persona(1, self.dummy_profile); p1.attributes = np.array([1, 1])
        p2 = Persona(2, self.dummy_profile); p2.attributes = np.array([-1, -1])
        p3 = Persona(3, self.dummy_profile); p3.attributes = np.array([0.9, 0.9])
        population = [p1, p2, p3]
        matrix = build_influence_matrix(population, threshold=0.8)
        self.assertEqual(matrix.nnz, 2)

class TestLLMIntegration(unittest.TestCase):

    def setUp(self):
        self.dummy_profile = "A test profile."
        self.dummy_axes = {"axis_1": "Community vs. Profit", "axis_2": "Regulation vs. Freedom"}
        self.p = Persona(1, self.dummy_profile, mood=0.25)
        self.p.attributes = np.array([0.5, -0.5])

    def test_construct_axis_generation_prompt(self):
        """Tests the meta-analysis prompt for axis generation."""
        profiles = ["Profile 1", "Profile 2"]
        first_event = "A test event."
        prompt = construct_axis_generation_prompt(profiles, first_event)
        self.assertIn("Profile 1", prompt)
        self.assertIn("Profile 2", prompt)
        self.assertIn(first_event, prompt)
        self.assertIn("ideological axes", prompt)

    def test_construct_vector_initialization_prompt(self):
        """Tests the prompt for initializing a persona's vector."""
        prompt = construct_vector_initialization_prompt(self.dummy_profile, self.dummy_axes)
        self.assertIn(self.dummy_profile, prompt)
        self.assertIn(self.dummy_axes["axis_1"], prompt)
        self.assertIn(self.dummy_axes["axis_2"], prompt)

    def test_construct_thought_generation_prompt(self):
        """Tests the main thought generation prompt."""
        peer_context = [(0.9, "A great idea."), (0.4, "A bad idea.")]
        prompt = construct_prompt(self.p, "event", "model", peer_context, "memory")
        self.assertIn("influence score: 0.90", prompt)
        self.assertIn("A great idea.", prompt)
        self.assertIn("previously thought: \"memory\"", prompt)

    def test_construct_distillation_prompt(self):
        """Tests the state distillation prompt."""
        prompt = construct_distillation_prompt(self.p, "A new thought.", self.dummy_axes)
        self.assertIn("A new thought.", prompt)
        self.assertIn(self.dummy_axes["axis_1"], prompt)
        self.assertIn(self.dummy_axes["axis_2"], prompt)

    def test_construct_statement_prompt(self):
        """Tests the prompt for generating a statement from a thought."""
        prompt = construct_statement_prompt(self.p.profile, "This is an internal thought.")
        self.assertIn(self.p.profile, prompt)
        self.assertIn("This is an internal thought.", prompt)
        self.assertIn("public statement", prompt)

if __name__ == '__main__':
    unittest.main()
