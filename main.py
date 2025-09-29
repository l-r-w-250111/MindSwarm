import argparse
import sys
import gui
from PySide6.QtWidgets import QApplication

from simulation_core import (
    initialize_simulation,
    run_simulation_step,
    finalize_simulation,
    Logger
)

def run_llm_simulation(model_name: str, scenario_path: str, personas_path: str, log_path: str, listen_to_all=False, gui_mode=False, logger_callback=None):
    """
    Runs a full simulation, orchestrating initialization, step-by-step execution, and finalization.
    """
    state = initialize_simulation(model_name, scenario_path, personas_path, log_path, listen_to_all, gui_mode, logger_callback)

    if state is None:
        # Initialization failed
        return None if gui_mode else 1

    try:
        for _ in range(state.time_steps):
            state = run_simulation_step(state)

        finalize_simulation(state)

        if gui_mode:
            return {
                "structured_log": state.structured_log,
                "population": state.population,
                "influence_matrix": state.influence_matrix,
                "mood_history": state.mood_history
            }
        return 0 # Success status

    except Exception as e:
        if state and state.logger:
            state.logger.log(f"\n--- AN UNEXPECTED ERROR OCCURRED ---\n{e}")
            state.logger.close()
        else:
            print(f"An unexpected error occurred: {e}")
        return None if gui_mode else 1


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