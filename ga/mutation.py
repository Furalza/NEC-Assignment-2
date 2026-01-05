from __future__ import annotations
import random
from typing import List


def mutate_one_gene(chrom: List[int], k_colors: int) -> List[int]:
    """
    Always mutate exactly one random gene: set it to a random color in [0, k_colors-1].
    """
    if not chrom:
        return chrom

    if k_colors <= 0:
        raise ValueError("k_colors must be > 0")

    c = chrom[:]
    i = random.randrange(len(c))
    c[i] = random.randrange(k_colors)
    return c


def mutate_per_gene(chrom: List[int], k_colors: int, p: float) -> List[int]:
    """
    Each gene mutates independently with probability p.
    """
    if k_colors <= 0:
        raise ValueError("k_colors must be > 0")

    c = chrom[:]
    for i in range(len(c)):
        if random.random() < p:
            c[i] = random.randrange(k_colors)
    return c
