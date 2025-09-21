# AI Social Simulator

This project is a prototype for a sophisticated social simulation platform that uses a hybrid model, combining the power of Large Language Models (LLMs) for qualitative reasoning with quantitative numerical models for large-scale influence calculation.

It allows users to define a set of personas and a multi-step scenario in simple Markdown files, and then simulates how these personas would think, speak, and influence each other in response to the unfolding events.

## Core Concepts

This simulation is built on several key concepts that evolved through iterative development:

1.  **Hybrid AI Model:** Instead of relying purely on numerical calculations or purely on qualitative text generation, this model combines both. Personas generate rich, human-like "thoughts" using an LLM, and these thoughts are then "distilled" back into numerical state changes (e.g., shifts in mood or ideological stance).

2.  **Dynamic Ideological Axes:** The simulation does not use fixed ideological axes (like a static political compass). Instead, it performs a "meta-analysis" at the beginning of each run, using an LLM to read all persona profiles and the initial event to determine the two most relevant axes of debate for that specific context.

3.  **Thought vs. Statement:** The model distinguishes between a persona's internal, unfiltered "thought" and their external, public "statement". A persona's internal state is updated by what they think, but their influence on others is determined by what they actually say.

4.  **Data-Driven Configuration:** The simulation is highly flexible. Both the personas and the multi-step scenarios are loaded from external `.md` files, allowing users to easily create and run new simulations without touching the core Python code.

## Features

-   **Dynamic Simulation Setup:** Automatically determines the most relevant ideological axes for a given scenario.
-   **LLM-Powered Personas:** Generates nuanced thoughts and public statements for each agent.
-   **Hybrid State Management:** Personas have both a qualitative state (thoughts, statements) and a quantitative state (numerical vectors, mood) that evolve over time.
-   **Influence Network:** Calculates a sparse influence matrix based on the similarity of personas' numerical states.
-   **Flexible Configuration:** Define personas and scenarios in simple Markdown files.
-   **Comprehensive Logging:** Generates a detailed Markdown log of the entire simulation, mirroring the console output for easy analysis.
-   **Visualization:** Creates plots of the final influence network and the average mood over time.

## How to Use

### 1. Setup

First, ensure you have a local Ollama server running. You can find installation instructions here: [https://ollama.com/](https://ollama.com/)

Pull the model you wish to use, for example:
```bash
ollama pull llama3
```

Next, set up the Python environment:
```bash
# It is recommended to use a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running a Simulation

You can run a simulation using the `main.py` script. It accepts several command-line arguments to customize the run.

```bash
python3 persona_sim/main.py --model <model_name> --personas <path_to_personas.md> --scenario <path_to_scenario.md> --output-log <path_to_output.md>
```

**Arguments:**

-   `--model`: (Required) The name of the Ollama model to use (e.g., `llama3`, `gemma3`).
-   `--personas`: (Optional) Path to the Markdown file containing persona definitions. Defaults to `personas.md`.
-   `--scenario`: (Optional) Path to the Markdown file containing the scenario. Defaults to `scenario.md`.
-   `--output-log`: (Optional) Path to save the unified output log file. Defaults to `simulation_log.md`.
-   `--listen-to-all`: (Optional) A flag that, if set, makes all personas hear all other personas' statements. By default, they only hear from their top influencers.

**Example:**

To run the included "Product Meeting" scenario:
```bash
python3 persona_sim/main.py --model llama3 --personas personas_meeting.md --scenario scenario_meeting.md --output-log log_product_meeting.md
```

### 3. Output Files

After a successful run, the following files will be generated:

-   **`[output-log].md`:** A comprehensive log of the entire simulation, mirroring the terminal output.
-   **`influence_network.png`:** A graph visualizing the final influence network between personas.
-   **`mood_history.png`:** A line plot showing the average mood of the population over the simulation's time steps.

## Customization

You can easily create your own simulations by creating new persona and scenario files.

-   **Personas (`.md`):** Define each persona with a `profile:` key. Separate personas with `---`.
-   **Scenarios (`.md`):** Write the event description for each time step. Separate steps with `---`.

### Tip: Multi-language Simulation

You can instruct personas to think and speak in a specific language by including it in their profile. The LLM will pick up on this instruction.

For example, to create a persona that speaks Japanese:

**`personas.md`**
```markdown
profile: 私は日本の東京出身のエンジニアです。日本語で考え、日本語で話します。イノベーションと効率性を重視します。
---
profile: I am a marketing manager from California. I think and speak in English and am focused on customer engagement.
```
