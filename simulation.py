import numpy as np
import scipy.sparse as sp
from persona import Persona

def build_influence_matrix(population: list[Persona], threshold: float) -> sp.csr_matrix:
    """
    Builds a sparse matrix representing the influence network.

    An entry (i, j) in the matrix represents the influence of persona j on persona i.

    Args:
        population (list[Persona]): The list of personas in the simulation.
        threshold (float): The minimum influence score to be considered a connection.

    Returns:
        scipy.sparse.csr_matrix: The sparse influence matrix.
    """
    n = len(population)
    # Use LIL format for efficient incremental construction
    influence_matrix = sp.lil_matrix((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            p1 = population[i]
            p2 = population[j]
            influence = calculate_influence(p1, p2)

            if influence >= threshold:
                influence_matrix[i, j] = influence

    # Convert to CSR format for efficient matrix-vector products
    return influence_matrix.tocsr()

def calculate_influence(p1: Persona, p2: Persona) -> float:
    """
    Calculates the influence score between two personas based on attribute similarity.
    A higher score means stronger influence.

    For this prototype, we use the inverse of the Euclidean distance.
    A small epsilon is added to the denominator to avoid division by zero.
    """
    # Ensure personas have attributes before calculating
    if p1.attributes.size == 0 or p2.attributes.size == 0:
        return 0.0

    distance = np.linalg.norm(p1.attributes - p2.attributes)
    return 1 / (1 + distance)
