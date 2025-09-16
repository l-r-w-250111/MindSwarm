import numpy as np
import argparse
import sys
import os
from persona import Persona
from simulation import build_influence_matrix
from llm_integration import (
    generate_thought,
    generate_statement_from_thought,
    distill_state_from_thought,
    generate_ideological_axes,
    initialize_persona_vector
)
from visualize import plot_influence_network, plot_mood_history

class Logger:
    """A simple logger to print to console and write to a file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        try:
            self.logfile = open(filepath, 'w', encoding='utf-8')
        except IOError as e:
            self.terminal.write(f"Error: Could not open log file at {filepath}. Error: {e}\n")
            self.logfile = None

    def log(self, message):
        self.terminal.write(message + '\n')
        if self.logfile:
            self.logfile.write(message + '\n')
            self.logfile.flush()

    def close(self):
        if self.logfile:
            self.logfile.close()

def load_scenario(filepath: str) -> list[str]:
    """Loads a scenario from a Markdown file, ensuring it's within the project directory."""
    abs_path = os.path.abspath(filepath)
    if not abs_path.startswith(os.getcwd()):
        raise FileNotFoundError(f"Access denied: {filepath} is outside the project directory.")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    events = [event.strip() for event in content.split('---')]
    return [event for event in events if event]

def load_personas_from_md(filepath: str) -> list[str]:
    """Loads persona profiles from a Markdown file."""
    abs_path = os.path.abspath(filepath)
    if not abs_path.startswith(os.getcwd()):
        raise FileNotFoundError(f"Access denied: {filepath} is outside the project directory.")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    profiles = []
    for block in content.split('---'):
        if not block.strip():
            continue
        for line in block.strip().split('\n'):
            if line.lower().startswith('profile:'):
                profiles.append(line.split(':', 1)[1].strip())
                break
    return profiles

def run_llm_simulation(model_name: str, scenario_path: str, personas_path: str, log_path: str, population_size=10):
    """
    Runs a simulation with a unified logger and dynamically generated ideological axes.
    """
    logger = Logger(log_path)

    try:
        # --- DYNAMIC SETUP PHASE ---
        logger.log("--- Starting Simulation Setup ---")

        try:
            scenario = load_scenario(scenario_path)
            time_steps = len(scenario)
            persona_profiles = load_personas_from_md(personas_path)
        except FileNotFoundError as e:
            logger.log(f"FATAL ERROR: A required file was not found. Please check file paths. Details: {e}")
            return

        if not persona_profiles:
            logger.log("Error: No persona profiles found.")
            return

        axes = generate_ideological_axes(persona_profiles, scenario[0], model_name, logger)
        axis_names = list(axes.values())

        population = []
        for i in range(population_size):
            profile_text = persona_profiles[i % len(persona_profiles)]
            persona = Persona(persona_id=i, profile=profile_text)
            population.append(persona)

        logger.log(f"\n--- Initializing Persona Vectors on Axes: {axis_names} ---")
        for persona in population:
            initial_vector = initialize_persona_vector(persona.profile, axes, model_name, logger)
            persona.set_attributes(initial_vector, axis_names)
            logger.log(f"  -> P{persona.id} ({persona.profile[:20]}...): {np.round(persona.attributes, 2)}, Mood: {persona.mood:.2f}")

        # --- SIMULATION PHASE ---
        logger.log("\n--- Starting Simulation Loop ---")

        persona_self_memories = {p.id: "" for p in population}
        statements_previous_step = {}
        mood_history = []
        influence_matrix = build_influence_matrix(population, threshold=0.6)

        for step in range(time_steps):
            event = scenario[step]
            logger.log(f"\n--- Time Step {step + 1}/{time_steps}: EVENT: {event} ---\n")

            statements_this_step = {}
            current_moods = []

            for persona in population:
                memory = persona_self_memories[persona.id]
                influencer_row = influence_matrix.getrow(persona.id)
                top_influencer_indices = influencer_row.toarray().argsort()[0][-3:-1]

                peer_context = []
                for influencer_id in top_influencer_indices:
                    score = influencer_row[0, influencer_id]
                    if score > 0:
                        influencer_statement = statements_previous_step.get(influencer_id, "")
                        if influencer_statement:
                            peer_context.append((score, influencer_statement))

                thought = generate_thought(persona, event, model_name, logger, peer_context, memory)
                statement = generate_statement_from_thought(persona.profile, thought, model_name, logger)

                logger.log(f"  -> P{persona.id} ({persona.profile[:20]}...)\n"
                           f"     - Thought (Internal): {thought}\n"
                           f"     - Statement (External): \"{statement}\"")

                if not thought.startswith("("):
                    new_attributes = distill_state_from_thought(persona, thought, model_name, axes, logger)
                    persona.set_attributes(new_attributes, axis_names)
                    logger.log(f"     ... new state: {np.round(persona.attributes, 2)}, Mood: {persona.mood:.2f}")

                persona_self_memories[persona.id] = thought # Self-memory is the internal thought
                statements_this_step[persona.id] = statement # Peer context is the external statement
                current_moods.append(persona.mood)

            average_mood = np.mean(current_moods) if current_moods else 0.0
            mood_history.append(average_mood)
            statements_previous_step = statements_this_step.copy()
            influence_matrix = build_influence_matrix(population, threshold=0.6)
            logger.log(f"\n--- End of Step {step + 1}: Avg Mood: {average_mood:.3f}, Matrix Density: {influence_matrix.nnz / (population_size**2):.2%} ---")

        logger.log("\n--- Simulation Complete ---")

        logger.log("\n--- Generating Visualizations ---")
        plot_influence_network(population, influence_matrix, logger)
        plot_mood_history(mood_history, logger)
        logger.log("--- Visualizations Generated ---")

    except Exception as e:
        logger.log(f"\n--- AN UNEXPECTED ERROR OCCURRED ---")
        logger.log(str(e))
    finally:
        logger.log("\n--- Closing Log File ---")
        logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an LLM-driven persona simulation.")
    parser.add_argument("--model", type=str, default="llama3", help="The name of the Ollama model to use (e.g., 'gemma:2b').")
    parser.add_argument("--scenario", type=str, default="scenario.md", help="Path to the Markdown file containing the simulation scenario.")
    parser.add_argument("--personas", type=str, default="personas.md", help="Path to the Markdown file containing the persona definitions.")
    parser.add_argument("--output-log", type=str, default="simulation_log.md", help="Path to save the unified output log file.")
    args = parser.parse_args()

    run_llm_simulation(
        model_name=args.model,
        scenario_path=args.scenario,
        personas_path=args.personas,
        log_path=args.output_log
    )
