import sys
import json
import os
import threading
from collections import deque

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox, QTextEdit,
    QSplitter, QTreeView, QHeaderView, QFileDialog, QMessageBox, QTableWidget
)
from PySide6.QtCore import Qt, Signal, Slot, QObject, QItemSelectionModel
from PySide6.QtGui import QStandardItemModel, QStandardItem

from main import (
    initialize_simulation,
    run_simulation_step,
    finalize_simulation,
    SimulationState,
    Logger
)
# Matplotlib integration with PySide6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Thread-safe Communication ---
class Communicate(QObject):
    log_signal = Signal(str)
    initialization_done_signal = Signal(object) # Will emit SimulationState or None
    step_done_signal = Signal(object)           # Will emit updated SimulationState
    finalization_done_signal = Signal(object)   # Will emit final SimulationState

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Social Simulator")
        self.setGeometry(100, 100, 1400, 900)

        # --- Thread-safe communication ---
        self.comm = Communicate()
        self.comm.log_signal.connect(self.handle_log_update)
        self.comm.initialization_done_signal.connect(self.handle_initialization_done)
        self.comm.step_done_signal.connect(self.handle_step_done)
        self.comm.finalization_done_signal.connect(self.handle_finalization_done)

        self.simulation_data = []
        self.sim_state = None

        # --- Main Layout ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- Controls Pane (Left) ---
        controls_frame = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_frame)
        splitter.addWidget(controls_frame)

        # Control Widgets
        controls_grid = QWidget()
        controls_grid_layout = QGridLayout(controls_grid)
        controls_layout.addWidget(controls_grid)

        # Model
        controls_grid_layout.addWidget(QLabel("Ollama Model:"), 0, 0)

        # Load config
        config = {}
        config_path = 'config.json'
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass # Use defaults if file is invalid

        default_model = config.get('Ollama', {}).get('default_model', 'gemma3:12b')
        self.model_input = QLineEdit(default_model)
        controls_grid_layout.addWidget(self.model_input, 0, 1, 1, 2)

        # Personas
        controls_grid_layout.addWidget(QLabel("Personas File:"), 1, 0)
        default_personas = config.get('Defaults', {}).get('personas', 'personas.md')
        self.personas_input = QLineEdit(default_personas)
        controls_grid_layout.addWidget(self.personas_input, 1, 1)
        self.browse_personas_btn = QPushButton("...")
        self.browse_personas_btn.clicked.connect(self.browse_personas)
        controls_grid_layout.addWidget(self.browse_personas_btn, 1, 2)

        # Scenario
        controls_grid_layout.addWidget(QLabel("Scenario File:"), 2, 0)
        default_scenario = config.get('Defaults', {}).get('scenario', 'scenario.md')
        self.scenario_input = QLineEdit(default_scenario)
        controls_grid_layout.addWidget(self.scenario_input, 2, 1)
        self.browse_scenario_btn = QPushButton("...")
        self.browse_scenario_btn.clicked.connect(self.browse_scenario)
        controls_grid_layout.addWidget(self.browse_scenario_btn, 2, 2)

        # Output Log
        controls_grid_layout.addWidget(QLabel("Output Log File:"), 3, 0)
        self.log_path_input = QLineEdit("simulation_log_gui.md")
        controls_grid_layout.addWidget(self.log_path_input, 3, 1, 1, 2)

        # Options
        self.listen_all_checkbox = QCheckBox("Listen to All Personas")
        controls_grid_layout.addWidget(self.listen_all_checkbox, 4, 0, 1, 3)

        # --- Simulation Buttons ---
        sim_button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Simulation")
        self.start_button.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        self.start_button.clicked.connect(self.start_simulation_thread)
        sim_button_layout.addWidget(self.start_button)

        self.next_step_button = QPushButton("Next Step")
        self.next_step_button.setStyleSheet("font-size: 14px; padding: 5px;")
        self.next_step_button.clicked.connect(self.run_next_step_thread)
        self.next_step_button.setEnabled(False)
        sim_button_layout.addWidget(self.next_step_button)

        controls_grid_layout.addLayout(sim_button_layout, 5, 0, 1, 3)

        # Live Log
        log_view_group = QGroupBox("Live Log")
        log_view_layout = QVBoxLayout(log_view_group)
        self.log_text_widget = QTextEdit()
        self.log_text_widget.setReadOnly(True)
        log_view_layout.addWidget(self.log_text_widget)
        controls_layout.addWidget(log_view_group)

        # --- Output Pane (Right) ---
        output_frame = QSplitter(Qt.Vertical)
        splitter.addWidget(output_frame)

        # Top part of output: Timeline and Interaction
        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)
        output_frame.addWidget(results_container)

        # Main timeline view
        timeline_group = QGroupBox("Simulation Timeline")
        timeline_layout = QVBoxLayout(timeline_group)
        self.timeline_table = QTableWidget()
        self.timeline_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.timeline_table.setWordWrap(True)
        timeline_layout.addWidget(self.timeline_table)
        results_layout.addWidget(timeline_group)

        # User Interaction
        user_interaction_group = QGroupBox("Your Intervention")
        user_interaction_layout = QHBoxLayout() # Create layout without a parent
        self.user_input_text = QTextEdit()
        self.user_input_text.setPlaceholderText("Type your message here to influence the next step...")
        self.user_input_text.setFixedHeight(60) # Make it a bit shorter
        self.user_input_text.setEnabled(False)
        user_interaction_layout.addWidget(self.user_input_text, 3) # Give more stretch to text edit

        self.submit_user_input_btn = QPushButton("Submit as Next Event")
        self.submit_user_input_btn.clicked.connect(self.on_submit_user_input)
        self.submit_user_input_btn.setEnabled(False)
        user_interaction_layout.addWidget(self.submit_user_input_btn, 1)

        user_interaction_group.setLayout(user_interaction_layout) # Set layout on the group box
        results_layout.addWidget(user_interaction_group) # Add the group box to the main layout

        # Bottom part of output: Visualizations
        plots_group = QGroupBox("Visualizations")
        plots_layout = QHBoxLayout(plots_group)

        self.fig_net = Figure()
        self.canvas_net = FigureCanvas(self.fig_net)
        plots_layout.addWidget(self.canvas_net)

        self.fig_mood = Figure()
        self.canvas_mood = FigureCanvas(self.fig_mood)
        plots_layout.addWidget(self.canvas_mood)

        output_frame.addWidget(plots_group)

        # Set initial sizes for the splitters
        splitter.setSizes([400, 1000])
        output_frame.setSizes([700, 200]) # Give more space to the timeline

    def browse_personas(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Personas File", "", "Markdown files (*.md);;All files (*.*)")
        if filename:
            self.personas_input.setText(filename)

    def browse_scenario(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Scenario File", "", "Markdown files (*.md);;All files (*.*)")
        if filename:
            self.scenario_input.setText(filename)

    def on_submit_user_input(self):
        user_text = self.user_input_text.toPlainText().strip()
        if not user_text:
            QMessageBox.warning(self, "Empty Input", "Please type a message to use as the next event.")
            return

        if not self.sim_state or self.sim_state.current_step >= self.sim_state.time_steps:
            QMessageBox.warning(self, "Invalid State", "Cannot submit an intervention at this time.")
            return

        # Log the intervention
        self.logger_callback(f"\n--- USER INTERVENTION ---")
        self.logger_callback(f"Overriding next event with: \"{user_text}\"")

        # Override the next event in the simulation state
        self.sim_state.scenario[self.sim_state.current_step] = user_text

        # Clear the input box
        self.user_input_text.clear()

        # Trigger the next step with the user's event
        self.run_next_step_thread()

    @Slot(str)
    def handle_log_update(self, message):
        self.log_text_widget.append(message)

    def logger_callback(self, message):
        self.comm.log_signal.emit(message)

    # --- Simulation Step 1: Initialization ---
    def simulation_initializer_worker(self):
        try:
            state = initialize_simulation(
                model_name=self.model_input.text(),
                scenario_path=self.scenario_input.text(),
                personas_path=self.personas_input.text(),
                log_path=self.log_path_input.text(),
                listen_to_all=self.listen_all_checkbox.isChecked(),
                gui_mode=True,
                logger_callback=self.logger_callback
            )
            self.comm.initialization_done_signal.emit(state)
        except Exception as e:
            self.comm.log_signal.emit(f"\n--- UNHANDLED EXCEPTION IN INITIALIZER ---\n{e}")
            self.comm.initialization_done_signal.emit(None)

    def start_simulation_thread(self):
        self.start_button.setEnabled(False)
        self.start_button.setText("Initializing...")
        self.next_step_button.setEnabled(False)
        self.clear_results()

        init_thread = threading.Thread(target=self.simulation_initializer_worker)
        init_thread.daemon = True
        init_thread.start()

    @Slot(object)
    def handle_initialization_done(self, state):
        if state is None:
            QMessageBox.critical(self, "Initialization Error", "The simulation failed to initialize. Check the Live Log for details.")
            self.start_button.setEnabled(True)
            self.start_button.setText("Start Simulation")
            return

        self.sim_state = state

        # Setup Timeline Table
        headers = ["Step", "Event"]
        for p in self.sim_state.population:
            headers.append(f"P{p.id}: {p.profile[:25]}...")
        self.timeline_table.setColumnCount(len(headers))
        self.timeline_table.setHorizontalHeaderLabels(headers)
        self.timeline_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.timeline_table.setColumnWidth(0, 50) # Step
        self.timeline_table.setColumnWidth(1, 250) # Event
        for i in range(2, len(headers)):
            self.timeline_table.setColumnWidth(i, 250) # Persona statements

        self.start_button.setText("Running...") # Keep it disabled but show it's active
        self.next_step_button.setEnabled(True)
        self.user_input_text.setEnabled(True)
        self.submit_user_input_btn.setEnabled(True)
        QMessageBox.information(self, "Ready", "Simulation initialized. Press 'Next Step' or enter your own event to begin.")

    # --- Simulation Step 2: Step-by-Step Execution ---
    def simulation_step_worker(self):
        try:
            updated_state = run_simulation_step(self.sim_state)
            self.comm.step_done_signal.emit(updated_state)
        except Exception as e:
            self.comm.log_signal.emit(f"\n--- UNHANDLED EXCEPTION IN STEP WORKER ---\n{e}")
            self.comm.step_done_signal.emit(None)

    def run_next_step_thread(self):
        self.next_step_button.setEnabled(False)
        self.user_input_text.setEnabled(False)
        self.submit_user_input_btn.setEnabled(False)
        step_thread = threading.Thread(target=self.simulation_step_worker)
        step_thread.daemon = True
        step_thread.start()

    @Slot(object)
    def handle_step_done(self, state):
        if state is None:
            QMessageBox.critical(self, "Step Error", "A step failed to execute. Check the Live Log for details.")
            # Reset UI to a safe state
            self.start_button.setEnabled(True)
            self.start_button.setText("Start Simulation")
            self.next_step_button.setEnabled(False)
            self.user_input_text.setEnabled(False)
            self.submit_user_input_btn.setEnabled(False)
            return

        self.sim_state = state
        self.update_step_display()

        if self.sim_state.current_step >= self.sim_state.time_steps:
            self.next_step_button.setEnabled(False)
            self.next_step_button.setText("Finished")
            self.user_input_text.setEnabled(False)
            self.submit_user_input_btn.setEnabled(False)
            self.run_finalization_thread()
        else:
            self.next_step_button.setEnabled(True)
            self.user_input_text.setEnabled(True)
            self.submit_user_input_btn.setEnabled(True)

    # --- Simulation Step 3: Finalization ---
    def simulation_finalizer_worker(self):
        try:
            finalize_simulation(self.sim_state)
            self.comm.finalization_done_signal.emit(self.sim_state)
        except Exception as e:
            self.comm.log_signal.emit(f"\n--- UNHANDLED EXCEPTION IN FINALIZER ---\n{e}")
            self.comm.finalization_done_signal.emit(None)

    def run_finalization_thread(self):
        self.start_button.setText("Finalizing...")
        final_thread = threading.Thread(target=self.simulation_finalizer_worker)
        final_thread.daemon = True
        final_thread.start()

    @Slot(object)
    def handle_finalization_done(self, state):
        if state is None:
            QMessageBox.critical(self, "Finalization Error", "Failed to finalize simulation or draw plots.")
        else:
            try:
                self.draw_plots(state.population, state.influence_matrix, state.mood_history)
                QMessageBox.information(self, "Success", "Simulation completed successfully!")
            except Exception as e:
                QMessageBox.critical(self, "GUI Error", f"Failed to draw plots:\n{e}")

        # Reset for next run
        self.start_button.setEnabled(True)
        self.start_button.setText("Start Simulation")
        self.next_step_button.setText("Next Step")
        self.next_step_button.setEnabled(False)
        self.user_input_text.setEnabled(False)
        self.submit_user_input_btn.setEnabled(False)
        self.sim_state = None

    # --- GUI Update Logic ---
    def clear_results(self):
        self.log_text_widget.clear()

        # Clear timeline table
        self.timeline_table.setRowCount(0)
        self.timeline_table.setColumnCount(0)
        self.timeline_table.setHorizontalHeaderLabels([])

        # Clear plots
        self.fig_net.clear()
        self.canvas_net.draw()
        self.fig_mood.clear()
        self.canvas_mood.draw()

        # Reset data and user input
        self.simulation_data = []
        self.user_input_text.clear()
        self.user_input_text.setEnabled(False)
        self.submit_user_input_btn.setEnabled(False)

    def update_step_display(self):
        from PySide6.QtWidgets import QTableWidgetItem
        # The latest step's data is the last one in the structured_log
        step_data = self.sim_state.structured_log[-1]

        row_position = self.timeline_table.rowCount()
        self.timeline_table.insertRow(row_position)

        # Step and Event
        self.timeline_table.setItem(row_position, 0, QTableWidgetItem(str(step_data['step'])))
        self.timeline_table.setItem(row_position, 1, QTableWidgetItem(step_data['event']))

        # Persona Statements
        persona_statements = {p['id']: p['statement'] for p in step_data['personas']}
        persona_thoughts = {p['id']: p['thought'] for p in step_data['personas']}

        for col, persona in enumerate(self.sim_state.population, start=2):
            statement = persona_statements.get(persona.id, "N/A")
            item = QTableWidgetItem(statement)

            thought = persona_thoughts.get(persona.id, "")
            item.setToolTip(f"Internal Thought:\n{thought}")
            self.timeline_table.setItem(row_position, col, item)

        self.timeline_table.resizeRowsToContents()
        self.timeline_table.scrollToBottom()

    def draw_plots(self, population, influence_matrix, mood_history):
        from visualize import plot_influence_network, plot_mood_history # Local import to avoid circular dependency issues

        plot_logger = Logger(gui_callback=self.logger_callback)

        # Influence Network Plot
        self.fig_net.clear()
        temp_fig_net = plot_influence_network(population, influence_matrix, plot_logger, return_fig=True)
        if temp_fig_net:
            temp_fig_net.canvas.setParent(None)
            ax = temp_fig_net.get_axes()[0]
            ax.remove()
            ax.figure = self.fig_net
            self.fig_net.add_axes(ax)
            self.canvas_net.draw()

        # Mood History Plot
        self.fig_mood.clear()
        temp_fig_mood = plot_mood_history(mood_history, plot_logger, return_fig=True)
        if temp_fig_mood:
            temp_fig_mood.canvas.setParent(None)
            ax = temp_fig_mood.get_axes()[0]
            ax.remove()
            ax.figure = self.fig_mood
            self.fig_mood.add_axes(ax)
            self.canvas_mood.draw()
