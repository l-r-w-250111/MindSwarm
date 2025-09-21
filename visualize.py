import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from persona import Persona
import scipy.sparse as sp

def plot_influence_network(population: list[Persona], influence_matrix: sp.csr_matrix, logger, filename="influence_network.png", return_fig=False):
    """
    Visualizes the influence network, saves it to a file, and optionally returns the figure.
    """
    fig = plt.figure(figsize=(12, 12))
    G = nx.from_scipy_sparse_array(influence_matrix, create_using=nx.DiGraph())
    moods = [p.mood for p in population]
    node_colors = [plt.cm.coolwarm(mood + 0.5) for mood in moods]
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=10, arrows=True)
    plt.title("Persona Influence Network")

    if return_fig:
        return fig
    else:
        plt.savefig(filename)
        plt.close(fig)
        logger.log(f"Influence network graph saved to {filename}")
        return None

def plot_mood_history(mood_history: list[float], logger, filename="mood_history.png", return_fig=False):
    """
    Plots the average mood, saves it, and optionally returns the figure.
    """
    fig = plt.figure(figsize=(10, 6))
    plt.plot(mood_history, marker='o', linestyle='-')
    plt.title("Average Population Mood Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Average Mood")
    plt.grid(True)
    plt.xticks(range(len(mood_history)))

    if return_fig:
        return fig
    else:
        plt.savefig(filename)
        plt.close(fig)
        logger.log(f"Mood history plot saved to {filename}")
        return None
