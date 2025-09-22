import sys
import json
import os
import threading
from collections import deque

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox, QTextEdit,
    QSplitter, QTreeView, QHeaderView, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, Signal, Slot, QObject
from PySide6.QtGui import QStandardItemModel, QStandardItem

from main import run_llm_simulation, Logger
# Matplotlib integration with PySide6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Thread-safe Communication ---
class Communicate(QObject):
    log_signal = Signal(str)
    results_signal = Signal(dict)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Social Simulator")
        self.setGeometry(100, 100, 1400, 900)

        # --- Thread-safe communication ---
        self.comm = Communicate()
        self.comm.log_signal.connect(self.handle_log_update)
        self.comm.results_signal.connect(self.handle_results_update)

        self.simulation_data = None

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
        default_model = 'gemma3:12b'
        config_path = 'config.json'
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    default_model = config.get('Ollama', {}).get('default_model', default_model)
            except (json.JSONDecodeError, IOError):
                pass
        self.model_input = QLineEdit(default_model)
        controls_grid_layout.addWidget(self.model_input, 0, 1, 1, 2)

        # Personas
        controls_grid_layout.addWidget(QLabel("Personas File:"), 1, 0)
        self.personas_input = QLineEdit("experiment-1/personas.md")
        controls_grid_layout.addWidget(self.personas_input, 1, 1)
        self.browse_personas_btn = QPushButton("...")
        self.browse_personas_btn.clicked.connect(self.browse_personas)
        controls_grid_layout.addWidget(self.browse_personas_btn, 1, 2)

        # Scenario
        controls_grid_layout.addWidget(QLabel("Scenario File:"), 2, 0)
        self.scenario_input = QLineEdit("experiment-1/scenario.md")
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

        # Run Button
        self.run_button = QPushButton("Run Simulation")
        self.run_button.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        self.run_button.clicked.connect(self.run_simulation_thread)
        controls_grid_layout.addWidget(self.run_button, 5, 0, 1, 3)

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

        # Top part of output: Results
        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)
        output_frame.addWidget(results_container)

        results_splitter = QSplitter(Qt.Horizontal)
        results_layout.addWidget(results_splitter)

        # Simulation Steps List
        steps_group = QGroupBox("Simulation Steps")
        steps_layout = QVBoxLayout(steps_group)
        self.steps_tree = QTreeView()
        self.steps_model = QStandardItemModel()
        self.steps_model.setHorizontalHeaderLabels(['Step'])
        self.steps_tree.setModel(self.steps_model)
        self.steps_tree.selectionModel().selectionChanged.connect(self.on_step_select)
        steps_layout.addWidget(self.steps_tree)
        results_splitter.addWidget(steps_group)

        # Step Details
        details_container = QWidget()
        details_layout = QVBoxLayout(details_container)
        results_splitter.addWidget(details_container)

        event_group = QGroupBox("Event")
        event_layout = QVBoxLayout(event_group)
        self.event_text = QTextEdit()
        self.event_text.setReadOnly(True)
        self.event_text.setFixedHeight(80)
        event_layout.addWidget(self.event_text)
        details_layout.addWidget(event_group)

        personas_group = QGroupBox("Persona Interactions")
        personas_layout = QVBoxLayout(personas_group)
        self.details_tree = QTreeView()
        self.details_model = QStandardItemModel()
        self.details_model.setHorizontalHeaderLabels(['ID', 'Profile Summary', 'Internal Thought', 'External Statement'])
        self.details_tree.setModel(self.details_model)
        self.details_tree.header().setSectionResizeMode(QHeaderView.Interactive)
        personas_layout.addWidget(self.details_tree)
        details_layout.addWidget(personas_group)

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
        output_frame.setSizes([600, 300])

    def browse_personas(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Personas File", "", "Markdown files (*.md);;All files (*.*)")
        if filename:
            self.personas_input.setText(filename)

    def browse_scenario(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Scenario File", "", "Markdown files (*.md);;All files (*.*)")
        if filename:
            self.scenario_input.setText(filename)

    @Slot(str)
    def handle_log_update(self, message):
        self.log_text_widget.append(message)

    @Slot(dict)
    def handle_results_update(self, results):
        self.run_button.setEnabled(True)
        self.run_button.setText("Run Simulation")

        if results is None:
            QMessageBox.critical(self, "Simulation Error", "The simulation failed. Check the Live Log for details.")
            return

        try:
            self.populate_steps_list(results.get("structured_log", []))
        except Exception as e:
            QMessageBox.critical(self, "GUI Error", f"Failed to populate step list:\n{e}")
            return

        try:
            self.draw_plots(results.get("population"), results.get("influence_matrix"), results.get("mood_history"))
        except Exception as e:
            QMessageBox.critical(self, "GUI Error", f"Failed to draw plots:\n{e}")
            return

        QMessageBox.information(self, "Success", "Simulation completed successfully!")

    def logger_callback(self, message):
        self.comm.log_signal.emit(message)

    def simulation_worker(self, model, scenario, personas, log_path, listen_all):
        try:
            results = run_llm_simulation(
                model_name=model, scenario_path=scenario, personas_path=personas,
                log_path=log_path, listen_to_all=listen_all,
                gui_mode=True, logger_callback=self.logger_callback
            )
            self.comm.results_signal.emit(results)
        except Exception as e:
            self.comm.log_signal.emit(f"\n--- UNHANDLED EXCEPTION IN WORKER ---\n{e}")
            self.comm.results_signal.emit(None)

    def run_simulation_thread(self):
        self.run_button.setEnabled(False)
        self.run_button.setText("Running...")
        self.clear_results()

        sim_thread = threading.Thread(
            target=self.simulation_worker,
            args=(
                self.model_input.text(), self.scenario_input.text(),
                self.personas_input.text(), self.log_path_input.text(),
                self.listen_all_checkbox.isChecked()
            )
        )
        sim_thread.daemon = True
        sim_thread.start()

    def clear_results(self):
        self.log_text_widget.clear()
        self.steps_model.clear()
        self.steps_model.setHorizontalHeaderLabels(['Step'])
        self.details_model.clear()
        self.details_model.setHorizontalHeaderLabels(['ID', 'Profile Summary', 'Internal Thought', 'External Statement'])
        self.event_text.clear()
        self.fig_net.clear()
        self.canvas_net.draw()
        self.fig_mood.clear()
        self.canvas_mood.draw()

    def populate_steps_list(self, simulation_data):
        self.simulation_data = simulation_data
        for i, step_data in enumerate(simulation_data):
            item = QStandardItem(f"Step {step_data['step']}")
            item.setData(i, Qt.UserRole)
            self.steps_model.appendRow(item)

    def on_step_select(self, selected, deselected):
        if not selected.indexes():
            return
        index = selected.indexes()[0]
        step_index = self.steps_model.itemFromIndex(index).data(Qt.UserRole)

        if self.simulation_data and 0 <= step_index < len(self.simulation_data):
            step_data = self.simulation_data[step_index]
            self.event_text.setText(step_data.get('event', ''))

            self.details_model.clear()
            self.details_model.setHorizontalHeaderLabels(['ID', 'Profile Summary', 'Internal Thought', 'External Statement'])
            for p_data in step_data.get('personas', []):
                id_item = QStandardItem(str(p_data.get('id', '')))
                profile_summary = p_data.get('profile', '')[:50] + '...'
                summary_item = QStandardItem(profile_summary)
                thought_item = QStandardItem(p_data.get('thought', ''))
                statement_item = QStandardItem(p_data.get('statement', ''))
                self.details_model.appendRow([id_item, summary_item, thought_item, statement_item])

            for i in range(self.details_model.columnCount()):
                self.details_tree.resizeColumnToContents(i)

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
