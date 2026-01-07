from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set

from .graph_io import GraphData
from .fitness import FitnessResult, evaluate
from .selection import tournament_selection, roulette_selection
from .crossover import one_point, uniform
from .mutation import mutate_one_gene, mutate_per_gene


@dataclass
class GAConfig:
    # Problem setup
    k_colors: int = 20
    penalty: float = 1000.0

    # GA parameters
    pop_size: int = 100
    generations: int = 2000

    selection: str = "tournament"     # "tournament" or "roulette"
    tournament_k: int = 3

    crossover: str = "one_point"      # "one_point" or "uniform"
    crossover_rate: float = 0.9

    mutation: str = "per_gene"        # "one_gene" or "per_gene"
    mutation_rate: float = 0.02       # per_gene: probability per gene; one_gene: prob to apply one mutation

    elitism: int = 1
    patience: int = 200

    # --- Hybrid (repair / local search) ---
    use_repair: bool = True
    repair_steps: int = 80  # how many conflict-fix moves to try per child


@dataclass
class GARunResult:
    best_chrom: List[int]
    best_eval: FitnessResult
    best_fitness_history: List[float]
    best_conflicts_history: List[int]
    best_colors_used_history: List[int]
    stopped_early: bool
    generations_run: int


def _init_population(n_vertices: int, pop_size: int, k_colors: int) -> List[List[int]]:
    return [[random.randrange(k_colors) for _ in range(n_vertices)] for _ in range(pop_size)]


def _select_parents(pop: List[List[int]], fits: List[float], cfg: GAConfig, n_parents: int) -> List[List[int]]:
    if cfg.selection == "tournament":
        return tournament_selection(pop, fits, k=cfg.tournament_k, n_parents=n_parents)
    if cfg.selection == "roulette":
        return roulette_selection(pop, fits, n_parents=n_parents)
    raise ValueError("cfg.selection must be 'tournament' or 'roulette'")


def _crossover(p1: List[int], p2: List[int], cfg: GAConfig) -> Tuple[List[int], List[int]]:
    if random.random() >= cfg.crossover_rate:
        return p1[:], p2[:]

    if cfg.crossover == "one_point":
        return one_point(p1, p2)
    if cfg.crossover == "uniform":
        return uniform(p1, p2, swap_prob=0.5)

    raise ValueError("cfg.crossover must be 'one_point' or 'uniform'")


def _mutate(child: List[int], cfg: GAConfig) -> List[int]:
    if cfg.mutation == "one_gene":
        return mutate_one_gene(child, cfg.k_colors) if random.random() < cfg.mutation_rate else child
    if cfg.mutation == "per_gene":
        return mutate_per_gene(child, cfg.k_colors, p=cfg.mutation_rate)
    raise ValueError("cfg.mutation must be 'one_gene' or 'per_gene'")


# ---------------------------
# FAST REPAIR (incremental)
# ---------------------------

def _count_conflicts_for_vertex(graph: GraphData, chrom: List[int], v: int) -> int:
    """How many neighbors of v share the same color as v."""
    cv = chrom[v]
    c = 0
    for nb in graph.adjacency[v]:
        if chrom[nb] == cv:
            c += 1
    return c


def _neighbor_color_counts(graph: GraphData, chrom: List[int], v: int, k_colors: int) -> List[int]:
    """counts[color] = number of neighbors of v having that color."""
    counts = [0] * k_colors
    for nb in graph.adjacency[v]:
        col = chrom[nb]
        if 0 <= col < k_colors:
            counts[col] += 1
    return counts


def _best_color_for_vertex(graph: GraphData, chrom: List[int], v: int, k_colors: int) -> Tuple[int, int]:
    """
    Pick the color that minimizes local conflicts at v (no temporary recolor loop).
    Returns (best_color, best_local_conflicts).
    """
    counts = _neighbor_color_counts(graph, chrom, v, k_colors)

    current = chrom[v]
    best_color = current
    best_conf = counts[current] if 0 <= current < k_colors else 10**9

    for color in range(k_colors):
        conf = counts[color]
        if conf < best_conf:
            best_conf = conf
            best_color = color
            if best_conf == 0:
                break

    return best_color, best_conf


def repair_solution(graph: GraphData, chrom: List[int], k_colors: int, steps: int) -> List[int]:
    """
    Fast greedy repair/local search:
    - Maintain a set of conflicted vertices.
    - Each step recolor one conflicted vertex to reduce conflicts.
    - Only update conflict-status locally (vertex + neighbors).
    """
    c = chrom[:]
    n = graph.n_vertices

    # Initial conflicted set: scan edges once using adjacency
    conflicted: Set[int] = set()
    for v in range(n):
        cv = c[v]
        for nb in graph.adjacency[v]:
            if nb > v and c[nb] == cv:
                conflicted.add(v)
                conflicted.add(nb)

    if not conflicted:
        return c

    def refresh_vertex(vx: int) -> None:
        if _count_conflicts_for_vertex(graph, c, vx) > 0:
            conflicted.add(vx)
        else:
            conflicted.discard(vx)

    for _ in range(steps):
        if not conflicted:
            break

        v = random.choice(tuple(conflicted))
        best_color, _ = _best_color_for_vertex(graph, c, v, k_colors)
        if best_color != c[v]:
            c[v] = best_color

        # local updates only
        refresh_vertex(v)
        for nb in graph.adjacency[v]:
            refresh_vertex(nb)

    return c


def run_ga(graph: GraphData, cfg: GAConfig, seed: Optional[int] = None) -> GARunResult:
    if seed is not None:
        random.seed(seed)

    if cfg.k_colors <= 0:
        raise ValueError("k_colors must be > 0")
    if cfg.pop_size < 2:
        raise ValueError("pop_size must be >= 2")

    pop = _init_population(graph.n_vertices, cfg.pop_size, cfg.k_colors)

    best_so_far_chrom: Optional[List[int]] = None
    best_so_far_eval: Optional[FitnessResult] = None
    best_so_far_fit: Optional[float] = None

    best_fit_hist: List[float] = []
    best_conf_hist: List[int] = []
    best_cols_hist: List[int] = []

    no_improve = 0
    stopped_early = False
    generations_run = 0

    for gen in range(cfg.generations):
        generations_run = gen + 1

        evals = [evaluate(graph, ind, penalty=cfg.penalty) for ind in pop]
        fits = [e.fitness for e in evals]

        best_idx = max(range(len(pop)), key=lambda i: fits[i])
        cur_best_fit = fits[best_idx]
        cur_best_eval = evals[best_idx]
        cur_best_chrom = pop[best_idx][:]

        if best_so_far_fit is None or cur_best_fit > best_so_far_fit:
            best_so_far_fit = cur_best_fit
            best_so_far_eval = cur_best_eval
            best_so_far_chrom = cur_best_chrom
            no_improve = 0
        else:
            no_improve += 1

        assert best_so_far_fit is not None and best_so_far_eval is not None and best_so_far_chrom is not None
        best_fit_hist.append(best_so_far_fit)
        best_conf_hist.append(best_so_far_eval.conflicts)
        best_cols_hist.append(best_so_far_eval.n_colors_used)

        if no_improve >= cfg.patience:
            stopped_early = True
            break

        elite_n = max(0, min(cfg.elitism, cfg.pop_size))
        elite_idxs = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)[:elite_n]
        elites = [pop[i][:] for i in elite_idxs]

        n_needed = cfg.pop_size - elite_n
        parents = _select_parents(pop, fits, cfg, n_parents=n_needed)

        next_pop: List[List[int]] = elites[:]

        i = 0
        while len(next_pop) < cfg.pop_size:
            p1 = parents[i % len(parents)]
            p2 = parents[(i + 1) % len(parents)]
            i += 2

            c1, c2 = _crossover(p1, p2, cfg)
            c1 = _mutate(c1, cfg)
            c2 = _mutate(c2, cfg)

            # Repair/local search (hybrid GA)
            if cfg.use_repair:
                c1 = repair_solution(graph, c1, cfg.k_colors, cfg.repair_steps)
                c2 = repair_solution(graph, c2, cfg.k_colors, cfg.repair_steps)

            if len(next_pop) < cfg.pop_size:
                next_pop.append(c1)
            if len(next_pop) < cfg.pop_size:
                next_pop.append(c2)

        pop = next_pop

    assert best_so_far_chrom is not None and best_so_far_eval is not None
    return GARunResult(
        best_chrom=best_so_far_chrom,
        best_eval=best_so_far_eval,
        best_fitness_history=best_fit_hist,
        best_conflicts_history=best_conf_hist,
        best_colors_used_history=best_cols_hist,
        stopped_early=stopped_early,
        generations_run=generations_run,
    )
