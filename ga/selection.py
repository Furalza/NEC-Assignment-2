from __future__ import annotations
import random
from typing import List


def tournament_selection(pop: List[List[int]], fitness: List[float], k: int, n_parents: int) -> List[List[int]]:
    """
    Tournament selection:
    - sample k individuals uniformly at random
    - pick the one with best fitness
    - repeat to obtain n_parents
    """
    if len(pop) != len(fitness):
        raise ValueError("pop and fitness must have same length")

    n = len(pop)
    parents: List[List[int]] = []

    for _ in range(n_parents):
        idxs = random.sample(range(n), k=min(k, n))
        best_idx = max(idxs, key=lambda i: fitness[i])
        parents.append(pop[best_idx][:])

    return parents


def roulette_selection(pop: List[List[int]], fitness: List[float], n_parents: int) -> List[List[int]]:
    """
    Roulette wheel selection (fitness-proportional).
    Requires non-negative weights. If fitness contains negatives, we shift.
    """
    if len(pop) != len(fitness):
        raise ValueError("pop and fitness must have same length")

    min_f = min(fitness)
    weights = [f - min_f + 1e-12 for f in fitness]  # shift to strictly positive
    total = sum(weights)
    if total <= 0:
        # fallback: uniform random selection
        return [random.choice(pop)[:] for _ in range(n_parents)]

    parents: List[List[int]] = []
    for _ in range(n_parents):
        r = random.random() * total
        acc = 0.0
        for ind, w in zip(pop, weights):
            acc += w
            if acc >= r:
                parents.append(ind[:])
                break
    return parents
