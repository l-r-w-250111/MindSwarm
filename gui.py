import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
from collections import deque
from main import run_llm_simulation, Logger
from visualize import plot_influence_network, plot_mood_history

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Social Simulator")
        self.geometry("1400x900")

        # --- Thread-safe data stores ---
        self.data_lock = threading.Lock()
        self.log_queue = deque()
        self.final_results = None

        self.style = ttk.Style(self)
        self.style.theme_use('clam')

        # --- Bind custom events ---
        self.bind("<<LogUpdate>>", self.handle_log_update)
        self.bind("<<ResultsUpdate>>", self.handle_results_update)

        # --- Layout ---
        main_paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        controls_frame = ttk.LabelFrame(main_paned_window, text="Controls", width=400)
        main_paned_window.add(controls_frame, weight=1)
        output_frame = ttk.Frame(main_paned_window)
        main_paned_window.add(output_frame, weight=4)

        # --- Control Widgets ---
        controls_grid = ttk.Frame(controls_frame)
        controls_grid.pack(fill=tk.X, expand=True, padx=10, pady=10)
        ttk.Label(controls_grid, text="Ollama Model:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.model_var = tk.StringVar(value="gemma3:12b")
        ttk.Entry(controls_grid, textvariable=self.model_var, width=30).grid(row=0, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        ttk.Label(controls_grid, text="Personas File:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.personas_path_var = tk.StringVar(value="experiment-1/personas.md")
        self.personas_entry = ttk.Entry(controls_grid, textvariable=self.personas_path_var)
        self.personas_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(controls_grid, text="...", command=self.browse_personas, width=3).grid(row=1, column=2, padx=5, pady=5)
        ttk.Label(controls_grid, text="Scenario File:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.scenario_path_var = tk.StringVar(value="experiment-1/scenario.md")
        self.scenario_entry = ttk.Entry(controls_grid, textvariable=self.scenario_path_var)
        self.scenario_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(controls_grid, text="...", command=self.browse_scenario, width=3).grid(row=2, column=2, padx=5, pady=5)
        ttk.Label(controls_grid, text="Output Log File:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.log_path_var = tk.StringVar(value="simulation_log_gui.md")
        ttk.Entry(controls_grid, textvariable=self.log_path_var).grid(row=3, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.listen_all_var = tk.BooleanVar()
        ttk.Checkbutton(controls_grid, text="Listen to All Personas", variable=self.listen_all_var).grid(row=4, columnspan=3, sticky="w", padx=5, pady=10)
        self.run_button = ttk.Button(controls_grid, text="Run Simulation", command=self.run_simulation_thread, style='Accent.TButton')
        self.run_button.grid(row=5, columnspan=3, pady=10, ipady=5)
        self.style.configure('Accent.TButton', font=('Helvetica', 12, 'bold'))
        log_view_frame = ttk.LabelFrame(controls_grid, text="Live Log")
        log_view_frame.grid(row=6, columnspan=3, sticky="nsew", pady=10)
        controls_grid.grid_rowconfigure(6, weight=1)
        self.log_text_widget = tk.Text(log_view_frame, height=15, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Output Area ---
        output_paned_window = ttk.PanedWindow(output_frame, orient=tk.VERTICAL)
        output_paned_window.pack(fill=tk.BOTH, expand=True)
        results_frame = ttk.Frame(output_paned_window)
        output_paned_window.add(results_frame, weight=3)
        plots_frame = ttk.LabelFrame(output_paned_window, text="Visualizations")
        output_paned_window.add(plots_frame, weight=2)
        results_paned_window = ttk.PanedWindow(results_frame, orient=tk.HORIZONTAL)
        results_paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        steps_list_frame = ttk.LabelFrame(results_paned_window, text="Simulation Steps")
        results_paned_window.add(steps_list_frame, weight=1)
        self.steps_tree = ttk.Treeview(steps_list_frame, columns=('step'), show='headings')
        self.steps_tree.heading('step', text='Step')
        self.steps_tree.pack(fill=tk.BOTH, expand=True)
        self.steps_tree.bind('<<TreeviewSelect>>', self.on_step_select)
        step_details_frame = ttk.Frame(results_paned_window)
        results_paned_window.add(step_details_frame, weight=3)
        event_frame = ttk.LabelFrame(step_details_frame, text="Event")
        event_frame.pack(fill=tk.X, padx=5, pady=5)
        self.event_text = tk.Text(event_frame, height=3, wrap=tk.WORD, state=tk.DISABLED)
        self.event_text.pack(fill=tk.X, expand=True, padx=5, pady=5)
        personas_frame = ttk.LabelFrame(step_details_frame, text="Persona Interactions")
        personas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.details_tree = ttk.Treeview(personas_frame, columns=('ID', 'Thought', 'Statement'), show='headings')
        self.details_tree.heading('ID', text='ID'); self.details_tree.column('ID', width=50, anchor='center')
        self.details_tree.heading('Thought', text='Internal Thought'); self.details_tree.column('Thought', width=300)
        self.details_tree.heading('Statement', text='External Statement'); self.details_tree.column('Statement', width=300)
        self.details_tree.pack(fill=tk.BOTH, expand=True)
        plot_container = ttk.Frame(plots_frame)
        plot_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.fig_net = plt.Figure(figsize=(5, 4), dpi=100)
        self.canvas_net = FigureCanvasTkAgg(self.fig_net, master=plot_container)
        self.canvas_net.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.fig_mood = plt.Figure(figsize=(5, 4), dpi=100)
        self.canvas_mood = FigureCanvasTkAgg(self.fig_mood, master=plot_container)
        self.canvas_mood.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

    def browse_personas(self):
        filename = filedialog.askopenfilename(title="Select Personas File", filetypes=[("All files", "*.*"), ("Markdown files", "*.md")])
        if filename:
            self.personas_path_var.set(filename)
            self.personas_entry.delete(0, tk.END)
            self.personas_entry.insert(0, filename)
            self.update_idletasks()

    def browse_scenario(self):
        filename = filedialog.askopenfilename(title="Select Scenario File", filetypes=[("All files", "*.*"), ("Markdown files", "*.md")])
        if filename:
            self.scenario_path_var.set(filename)
            self.scenario_entry.delete(0, tk.END)
            self.scenario_entry.insert(0, filename)
            self.update_idletasks()

    def handle_log_update(self, event):
        with self.data_lock:
            while self.log_queue:
                msg = self.log_queue.popleft()
                self.log_text_widget.config(state=tk.NORMAL)
                self.log_text_widget.insert(tk.END, msg + '\n')
                self.log_text_widget.see(tk.END)
                self.log_text_widget.config(state=tk.DISABLED)

    def handle_results_update(self, event):
        self.run_button.config(state=tk.NORMAL, text="Run Simulation")
        with self.data_lock:
            results = self.final_results

        if results is None:
            messagebox.showerror("Simulation Error", "The simulation failed. Check the Live Log for details.")
            return

        try:
            self.populate_steps_list(results["structured_log"])
        except Exception as e:
            messagebox.showerror("GUI Error", f"Failed to populate step list:\n{e}")
            return

        try:
            self.draw_plots(results["population"], results["influence_matrix"], results["mood_history"])
        except Exception as e:
            messagebox.showerror("GUI Error", f"Failed to draw plots:\n{e}")
            return

        messagebox.showinfo("Success", "Simulation completed successfully!")

    def logger_callback(self, message):
        with self.data_lock:
            self.log_queue.append(message)
        self.event_generate("<<LogUpdate>>", when="tail")

    def simulation_worker(self, model, scenario, personas, log_path, listen_all):
        try:
            results = run_llm_simulation(
                model_name=model, scenario_path=scenario, personas_path=personas,
                log_path=log_path, listen_to_all=listen_all,
                gui_mode=True, logger_callback=self.logger_callback
            )
            with self.data_lock:
                self.final_results = results
        except Exception as e:
            with self.data_lock:
                self.final_results = None
                self.log_queue.append(f"\n--- UNHANDLED EXCEPTION ---\n{e}")
        finally:
            self.event_generate("<<ResultsUpdate>>", when="tail")

    def run_simulation_thread(self):
        self.run_button.config(state=tk.DISABLED, text="Running...")
        self.clear_results()
        sim_thread = threading.Thread(
            target=self.simulation_worker,
            args=(
                self.model_var.get(), self.scenario_path_var.get(),
                self.personas_path_var.get(), self.log_path_var.get(),
                self.listen_all_var.get()
            )
        )
        sim_thread.daemon = True
        sim_thread.start()

    def clear_results(self):
        for item in self.steps_tree.get_children(): self.steps_tree.delete(item)
        for item in self.details_tree.get_children(): self.details_tree.delete(item)
        self.event_text.config(state=tk.NORMAL); self.event_text.delete('1.0', tk.END); self.event_text.config(state=tk.DISABLED)
        self.log_text_widget.config(state=tk.NORMAL); self.log_text_widget.delete('1.0', tk.END); self.log_text_widget.config(state=tk.DISABLED)
        self.fig_net.clear(); self.canvas_net.draw()
        self.fig_mood.clear(); self.canvas_mood.draw()

    def populate_steps_list(self, simulation_data):
        self.simulation_data = simulation_data
        for i, step_data in enumerate(self.simulation_data):
            self.steps_tree.insert('', tk.END, iid=i, values=(f"Step {step_data['step']}",))

    def on_step_select(self, event):
        selected_items = self.steps_tree.selection()
        if not selected_items: return
        selected_iid = int(selected_items[0])
        if not (0 <= selected_iid < len(self.simulation_data)): return
        step_data = self.simulation_data[selected_iid]
        self.event_text.config(state=tk.NORMAL)
        self.event_text.delete('1.0', tk.END)
        self.event_text.insert('1.0', step_data['event'])
        self.event_text.config(state=tk.DISABLED)
        for item in self.details_tree.get_children(): self.details_tree.delete(item)
        for p_data in step_data['personas']:
            self.details_tree.insert('', tk.END, values=(p_data['id'], p_data['thought'], p_data['statement']))

    def draw_plots(self, population, influence_matrix, mood_history):
        plot_logger = Logger(gui_callback=self.logger_callback)
        self.fig_net.clear()
        temp_fig_net = plot_influence_network(population, influence_matrix, plot_logger, return_fig=True)
        for ax in temp_fig_net.get_axes():
            ax.remove(); ax.figure = self.fig_net; self.fig_net.add_axes(ax)
        self.canvas_net.draw()
        plt.close(temp_fig_net)
        self.fig_mood.clear()
        temp_fig_mood = plot_mood_history(mood_history, plot_logger, return_fig=True)
        for ax in temp_fig_mood.get_axes():
            ax.remove(); ax.figure = self.fig_mood; self.fig_mood.add_axes(ax)
        self.canvas_mood.draw()
        plt.close(temp_fig_mood)
