from __future__ import annotations
from dataclasses import dataclass
from typing import List
from .graph_io import GraphData


@dataclass(frozen=True)
class FitnessResult:
    conflicts: int
    n_colors_used: int
    fitness: float  # GA will maximize this


def evaluate(graph: GraphData, chrom: List[int], penalty: float = 1000.0) -> FitnessResult:
    """
    chrom[i] = color assigned to vertex i (integer)

    conflicts: number of edges (u,v) where chrom[u] == chrom[v]
    n_colors_used: number of distinct colors used in the chromosome

    We want: conflicts -> 0, and then minimize colors used.
    GA usually maximizes fitness, so we use a penalized negative objective:
        fitness = - (penalty * conflicts + n_colors_used)
    """
    if len(chrom) != graph.n_vertices:
        raise ValueError(
            f"Chromosome length {len(chrom)} does not match number of vertices {graph.n_vertices}"
        )

    conflicts = 0
    for u, v in graph.edges:
        if chrom[u] == chrom[v]:
            conflicts += 1

    n_colors_used = len(set(chrom))
    fitness = - (penalty * conflicts + n_colors_used)

    return FitnessResult(conflicts=conflicts, n_colors_used=n_colors_used, fitness=fitness)
