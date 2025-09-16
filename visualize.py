import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from persona import Persona
import scipy.sparse as sp

def plot_influence_network(population: list[Persona], influence_matrix: sp.csr_matrix, logger, filename="influence_network.png"):
    """
    Visualizes the influence network and saves it to a file.
    """
    plt.figure(figsize=(12, 12))
    G = nx.from_scipy_sparse_array(influence_matrix, create_using=nx.DiGraph())
    moods = [p.mood for p in population]
    node_colors = [plt.cm.coolwarm(mood + 0.5) for mood in moods]
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=10, arrows=True)
    plt.title("Persona Influence Network")
    plt.savefig(filename)
    plt.close()
    logger.log(f"Influence network graph saved to {filename}")

def plot_mood_history(mood_history: list[float], logger, filename="mood_history.png"):
    """
    Plots the average mood of the population over time and saves it to a file.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(mood_history, marker='o', linestyle='-')
    plt.title("Average Population Mood Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Average Mood")
    plt.grid(True)
    plt.xticks(range(len(mood_history)))
    plt.savefig(filename)
    plt.close()
    logger.log(f"Mood history plot saved to {filename}")
