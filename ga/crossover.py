from __future__ import annotations
import random
from typing import List, Tuple


def one_point(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    """
    One-point crossover:
    - choose a cut position in [1, n-1]
    - swap tails
    """
    if len(p1) != len(p2):
        raise ValueError("Parents must have same length")

    n = len(p1)
    if n < 2:
        return p1[:], p2[:]

    cut = random.randint(1, n - 1)
    c1 = p1[:cut] + p2[cut:]
    c2 = p2[:cut] + p1[cut:]
    return c1, c2


def uniform(p1: List[int], p2: List[int], swap_prob: float = 0.5) -> Tuple[List[int], List[int]]:
    """
    Uniform crossover:
    - for each gene, swap with probability swap_prob
    """
    if len(p1) != len(p2):
        raise ValueError("Parents must have same length")

    c1, c2 = p1[:], p2[:]
    for i in range(len(c1)):
        if random.random() < swap_prob:
            c1[i], c2[i] = c2[i], c1[i]
    return c1, c2
