import streamlit as st
import os
import json
import pandas as pd
from simulation_core import initialize_simulation, run_simulation_step, finalize_simulation, Logger
from visualize import plot_influence_network, plot_mood_history

# --- Page Config ---
st.set_page_config(
    page_title="AI Social Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 AI Social Simulator")
st.markdown("A platform for simulating social dynamics using LLM-powered agents.")

# --- Helper Functions ---
def load_config():
    """Loads default paths from config.json."""
    config_path = 'config.json'
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

# --- Session State Initialization ---
if 'sim_state' not in st.session_state:
    st.session_state.sim_state = None
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'simulation_started' not in st.session_state:
    st.session_state.simulation_started = False
if 'finalized' not in st.session_state:
    st.session_state.finalized = False
if 'thought_vis' not in st.session_state:
    st.session_state.thought_vis = {}

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Simulation Controls")
    config = load_config()
    st.subheader("LLM Configuration")
    default_model = config.get('Ollama', {}).get('default_model', 'gpt-oss:20b')
    model_name = st.text_input("Ollama Model", value=default_model, help="The name of the Ollama model to use.")
    st.subheader("Input Files")
    default_personas = config.get('Defaults', {}).get('personas', 'personas.md')
    personas_path = st.text_input("Personas File Path", value=default_personas)
    default_scenario = config.get('Defaults', {}).get('scenario', 'scenario.md')
    scenario_path = st.text_input("Scenario File Path", value=default_scenario)
    st.subheader("Options")
    listen_to_all = st.checkbox("Listen to All Personas", value=False, help="If checked, all personas hear all other statements, regardless of influence.")
    st.markdown("---")

    if st.button("🚀 Start Simulation", type="primary", use_container_width=True, disabled=st.session_state.simulation_started):
        st.session_state.sim_state = None
        st.session_state.log_messages = []
        st.session_state.simulation_started = False
        st.session_state.finalized = False
        st.session_state.thought_vis = {} # Reset thought visibility on new run
        def logger_callback(message):
            st.session_state.log_messages.append(message)
        with st.spinner("Initializing simulation..."):
            state = initialize_simulation(
                model_name=model_name, scenario_path=scenario_path, personas_path=personas_path,
                log_path="simulation_log_streamlit.md", listen_to_all=listen_to_all,
                gui_mode=True, logger_callback=logger_callback
            )
        if state is None:
            st.error("Simulation failed to initialize. Check the logs and file paths.")
        else:
            st.session_state.sim_state = state
            st.session_state.simulation_started = True
            st.rerun()

    if st.session_state.simulation_started:
        if st.button("⏹️ Reset Simulation", use_container_width=True):
            st.session_state.sim_state = None
            st.session_state.log_messages = []
            st.session_state.simulation_started = False
            st.session_state.finalized = False
            st.session_state.thought_vis = {}
            st.rerun()

    if st.session_state.finalized:
        st.markdown("---")
        st.subheader("⬇️ Download Results")
        # Convert structured_log to a JSON string for download
        # Use a numpy-aware encoder if necessary, but here we'll just use default json
        try:
            json_string = json.dumps(st.session_state.sim_state.structured_log, indent=4)
            st.download_button(
                label="Download Full Log (JSON)",
                data=json_string,
                file_name="simulation_log.json",
                mime="application/json",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Could not prepare JSON for download: {e}")


# --- Main Page Layout ---
if not st.session_state.simulation_started:
    st.info("Configure the simulation in the sidebar and click 'Start Simulation' to begin.")
else:
    state = st.session_state.sim_state
    st.header("▶️ Step-by-Step Controls")
    if state.current_step >= state.time_steps:
        if not st.session_state.finalized:
            with st.spinner("Generating final plots..."):
                finalize_simulation(state)
                st.session_state.finalized = True
                st.rerun() # Rerun to show download button and final state
        st.success("🎉 Simulation Complete!")
    else:
        with st.form(key="step_form"):
            st.markdown(f"**Running Step {state.current_step + 1} of {state.time_steps}**")
            next_event = state.scenario[state.current_step]
            edited_event = st.text_input("📝 Next Event (Editable)", value=next_event, help="This is the scripted event for the next step. You can modify it before proceeding.")
            user_utterance = st.text_area("🗣️ Your Utterance (Optional)", placeholder="Type a message to inject into the simulation as an observer.", help="Your message will be heard by all personas in the next step.")
            submit_button = st.form_submit_button(label="➡️ Run Next Step", use_container_width=True)
            if submit_button:
                with st.spinner(f"Running step {state.current_step + 1}..."):
                    state.scenario[state.current_step] = edited_event
                    updated_state = run_simulation_step(state, user_utterance if user_utterance else None)
                    st.session_state.sim_state = updated_state
                st.rerun()

# --- Results Display ---
if st.session_state.simulation_started:
    st.markdown("---")
    st.header("📊 Simulation Results")
    state = st.session_state.sim_state

    # Persona Profiles
    st.subheader("👥 Persona Profiles")
    with st.container():
        for p in state.population:
            with st.expander(f"**Persona {p.id}:** {p.profile.splitlines()[0]}"):
                st.markdown(p.profile)

    # Conversation Timeline
    if state.structured_log:
        st.subheader("🗣️ Conversation Timeline")
        for i, step_data in enumerate(state.structured_log):
            st.markdown(f"#### Step {step_data['step']}: {step_data['event']}")
            if step_data.get('user_utterance'):
                st.markdown(f"> _User says: {step_data['user_utterance']}_")

            statements = {p['id']: p['statement'] for p in step_data['personas']}
            thoughts = {p['id']: p['thought'] for p in step_data['personas']}

            for p in state.population:
                statement = statements.get(p.id, "_(No statement)_")
                thought = thoughts.get(p.id, "_(No thought recorded)_").strip()

                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**P{p.id}:** {statement}")
                with col2:
                    # Unique key for each button
                    button_key = f"thought_{i}_{p.id}"
                    if st.button("View Thought", key=button_key, use_container_width=True):
                        # Toggle visibility state
                        st.session_state.thought_vis[button_key] = not st.session_state.thought_vis.get(button_key, False)

                # Display thought if button was toggled to visible
                if st.session_state.thought_vis.get(button_key, False):
                    st.info(f"**Internal Thought (P{p.id}):**\n\n{thought}")
            st.markdown("---")
    else:
        st.info("Run the first step to see the timeline.")

    if st.session_state.finalized:
        st.subheader("📈 Visualizations")
        plot_logger = Logger()
        col1, col2 = st.columns(2)
        with col1:
            st.write("Influence Network")
            fig_net = plot_influence_network(state.population, state.influence_matrix, plot_logger, return_fig=True)
            if fig_net: st.pyplot(fig_net)
        with col2:
            st.write("Average Mood History")
            fig_mood = plot_mood_history(state.mood_history, plot_logger, return_fig=True)
            if fig_mood: st.pyplot(fig_mood)

# --- Live Log Display ---
with st.expander("📜 Live Log", expanded=False):
    log_container = st.container(height=300)
    if st.session_state.log_messages:
        log_text = "\n".join(st.session_state.log_messages)
        log_container.text(log_text)
    else:
        log_container.text("Logs will appear here...")