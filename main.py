import numpy as np
import argparse
import sys
import os
import gui
from PySide6.QtWidgets import QApplication
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
    """A simple logger to print to console and optionally write to a file."""
    def __init__(self, filepath=None, gui_callback=None):
        self.terminal = sys.stdout
        self.logfile = None
        self.gui_callback = gui_callback
        if filepath:
            try:
                self.logfile = open(filepath, 'w', encoding='utf-8')
            except IOError as e:
                self.terminal.write(f"Error: Could not open log file at {filepath}. Error: {e}\n")

    def log(self, message):
        self.terminal.write(message + '\n')
        if self.logfile:
            self.logfile.write(message + '\n')
            self.logfile.flush()
        if self.gui_callback:
            self.gui_callback(message)

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

def run_llm_simulation(model_name: str, scenario_path: str, personas_path: str, log_path: str, listen_to_all=False, gui_mode=False, logger_callback=None):
    """
    Runs a simulation. In normal mode, it logs to a file. In GUI mode, it returns structured data.
    The number of personas is determined by the number of profiles in the personas file.
    """
    # In GUI mode, don't create a log file, but use a callback for real-time updates
    log_filepath = None if gui_mode else log_path
    logger = Logger(log_filepath, gui_callback=logger_callback)

    try:
        logger.log("--- Starting Simulation Setup ---")

        try:
            scenario = load_scenario(scenario_path)
            time_steps = len(scenario)
            persona_profiles = load_personas_from_md(personas_path)
        except FileNotFoundError as e:
            logger.log(f"FATAL ERROR: A required file was not found. Please check file paths. Details: {e}")
            return None if gui_mode else 1 # Return error status

        if not persona_profiles:
            logger.log("Error: No persona profiles found.")
            return None if gui_mode else 1

        population_size = len(persona_profiles)
        logger.log(f"Found {population_size} persona profiles. Setting population size to {population_size}.")

        axes = generate_ideological_axes(persona_profiles, scenario[0], model_name, logger)
        axis_names = list(axes.values())

        population = []
        for i, profile_text in enumerate(persona_profiles):
            persona = Persona(persona_id=i, profile=profile_text)
            population.append(persona)

        logger.log(f"\n--- Initializing Persona Vectors on Axes: {axis_names} ---")
        for persona in population:
            initial_vector = initialize_persona_vector(persona.profile, axes, model_name, logger)
            persona.set_attributes(initial_vector, axis_names)
            logger.log(f"  -> P{persona.id} ({persona.profile[:20]}...): {np.round(persona.attributes, 2)}, Mood: {persona.mood:.2f}")

        logger.log("\n--- Starting Simulation Loop ---")

        persona_self_memories = {p.id: "" for p in population}
        statements_previous_step = {}
        mood_history = []
        structured_log = []
        influence_matrix = build_influence_matrix(population, threshold=0.6)

        for step in range(time_steps):
            event = scenario[step]
            logger.log(f"\n--- Time Step {step + 1}/{time_steps}: EVENT: {event} ---\n")

            step_persona_data = []
            statements_this_step = {}
            current_moods = []

            for persona in population:
                memory = persona_self_memories[persona.id]
                peer_context = []
                # ... (peer context logic remains the same)
                if listen_to_all:
                    for other_persona in population:
                        if other_persona.id == persona.id: continue
                        statement = statements_previous_step.get(other_persona.id, "")
                        if statement: peer_context.append((1.0, statement))
                else:
                    influencer_row = influence_matrix.getrow(persona.id)
                    top_influencer_indices = influencer_row.toarray().argsort()[0][-3:-1]
                    for influencer_id in top_influencer_indices:
                        score = influencer_row[0, influencer_id]
                        if score > 0:
                            influencer_statement = statements_previous_step.get(influencer_id, "")
                            if influencer_statement: peer_context.append((score, influencer_statement))

                thought = generate_thought(persona, event, model_name, logger, peer_context, memory)
                statement = generate_statement_from_thought(persona.profile, thought, model_name, logger)

                logger.log(f"  -> P{persona.id} ({persona.profile[:20]}...)\n"
                           f"     - Thought (Internal): {thought}\n"
                           f"     - Statement (External): \"{statement}\"")

                if not thought.startswith("("):
                    new_attributes = distill_state_from_thought(persona, thought, model_name, axes, logger)
                    persona.set_attributes(new_attributes, axis_names)
                    logger.log(f"     ... new state: {np.round(persona.attributes, 2)}, Mood: {persona.mood:.2f}")

                persona_self_memories[persona.id] = thought
                statements_this_step[persona.id] = statement
                current_moods.append(persona.mood)
                step_persona_data.append({
                    'id': persona.id,
                    'profile': persona.profile,
                    'thought': thought,
                    'statement': statement
                })

            structured_log.append({'step': step + 1, 'event': event, 'personas': step_persona_data})

            average_mood = np.mean(current_moods) if current_moods else 0.0
            mood_history.append(average_mood)
            statements_previous_step = statements_this_step.copy()
            influence_matrix = build_influence_matrix(population, threshold=0.6)
            logger.log(f"\n--- End of Step {step + 1}: Avg Mood: {average_mood:.3f}, Matrix Density: {influence_matrix.nnz / (population_size**2):.2%} ---")

        logger.log("\n--- Simulation Complete ---")

        if gui_mode:
            return {
                "structured_log": structured_log,
                "population": population,
                "influence_matrix": influence_matrix,
                "mood_history": mood_history
            }
        else:
            logger.log("\n--- Generating Visualizations ---")
            plot_influence_network(population, influence_matrix, logger)
            plot_mood_history(mood_history, logger)
            logger.log("--- Visualizations Generated ---")
            return 0 # Success status

    except Exception as e:
        logger.log(f"\n--- AN UNEXPECTED ERROR OCCURRED ---\n{e}")
        return None if gui_mode else 1
    finally:
        logger.log("\n--- Closing Log ---")
        logger.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Run a command-line LLM-driven persona simulation.")
        parser.add_argument("--model", type=str, default="gemma3:12b", help="The name of the Ollama model to use (e.g., 'gemma3:12b').")
        parser.add_argument("--scenario", type=str, default="scenario.md", help="Path to the Markdown file containing the simulation scenario.")
        parser.add_argument("--personas", type=str, default="personas.md", help="Path to the Markdown file containing the persona definitions.")
        parser.add_argument("--output-log", type=str, default="simulation_log.md", help="Path to save the unified output log file.")
        parser.add_argument("--listen-to-all", action="store_true", help="If set, all personas hear all other statements.")
        args = parser.parse_args()

        run_llm_simulation(
            model_name=args.model,
            scenario_path=args.scenario,
            personas_path=args.personas,
            log_path=args.output_log,
            listen_to_all=args.listen_to_all,
            gui_mode=False
        )
    else:
        app = QApplication(sys.argv)
        window = gui.App()
        window.show()
        sys.exit(app.exec())
