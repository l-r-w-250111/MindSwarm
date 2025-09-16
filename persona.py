import numpy as np

class Persona:
    """
    Represents an individual agent in the simulation.

    Attributes:
        id (int): A unique identifier for the persona.
        profile (str): A text description of the persona's background and beliefs.
        attributes (np.ndarray): A vector representing the persona's stance on dynamic axes.
        mood (float): Represents the current emotional state or individual deviation.
    """
    def __init__(self, persona_id: int, profile: str, mood: float = 0.0):
        """
        Initializes a Persona instance. Numerical attributes are set later.

        Args:
            persona_id (int): The unique ID for the persona.
            profile (str): A text description of the persona's background and beliefs.
            mood (float): The initial mood of the persona.
        """
        self.id = persona_id
        self.profile = profile
        self.attributes = np.array([]) # Initialized as empty, set later
        self.mood = mood

    def __repr__(self) -> str:
        """
        Provides a string representation of the Persona instance for easy debugging.
        """
        return (f"Persona(id={self.id}, "
                f"profile='{self.profile[:40]}...', "
                f"attributes={np.round(self.attributes, 2)}, "
                f"mood={self.mood:.2f})")

    def set_attributes(self, new_attributes: dict, axis_names: list[str]):
        """
        Safely sets or updates the persona's numerical attributes from a dictionary,
        based on dynamically provided axis names.

        Args:
            new_attributes (dict): A dictionary with keys matching the axis_names, plus 'mood'.
            axis_names (list[str]): The names of the axes for the attribute vector.
        """
        try:
            attr_values = [float(new_attributes.get(name, 0.0)) for name in axis_names]
            self.attributes = np.array(attr_values)

            self.mood = float(new_attributes.get('mood', self.mood))

            # Clamp values to the expected range [-1.0, 1.0]
            self.attributes = np.clip(self.attributes, -1.0, 1.0)
            self.mood = np.clip(self.mood, -1.0, 1.0)

        except (ValueError, TypeError) as e:
            print(f"Warning: Could not set attributes for Persona {self.id} due to invalid data: {new_attributes}. Error: {e}")
