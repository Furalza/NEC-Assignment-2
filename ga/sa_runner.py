from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .graph_io import GraphData


def count_conflicts(graph: GraphData, chrom: List[int]) -> int:
    c = 0
    for u, v in graph.edges:
        if chrom[u] == chrom[v]:
            c += 1
    return c


def used_colors(chrom: List[int]) -> int:
    return len(set(chrom))


def conflicted_vertices(graph: GraphData, chrom: List[int]) -> List[int]:
    bad = set()
    for u, v in graph.edges:
        if chrom[u] == chrom[v]:
            bad.add(u)
            bad.add(v)
    return list(bad)


def best_color_for_vertex(graph: GraphData, chrom: List[int], v: int, k: int) -> Tuple[int, int]:
    """
    Returns (best_color, resulting_conflicts_on_edges_touching_v).
    """
    current = chrom[v]
    best_col = current
    best_local = 10**9

    for col in range(k):
        if col == current:
            continue
        local_conf = 0
        for nb in graph.adjacency[v]:
            if chrom[nb] == col:
                local_conf += 1
        if local_conf < best_local:
            best_local = local_conf
            best_col = col

    # also compute local conflicts for current
    if best_col == current:
        best_local = 0
        for nb in graph.adjacency[v]:
            if chrom[nb] == current:
                best_local += 1

    return best_col, best_local


@dataclass
class SAConfig:
    k_colors: int
    max_iters: int = 200_000
    start_temp: float = 2.0
    end_temp: float = 1e-4
    alpha: float = 0.9995  # cooling
    seed: int = 1
    target_conflicts: int = 0


@dataclass
class SAResult:
    best_chrom: List[int]
    best_conflicts: int
    best_colors_used: int
    iters_run: int
    stopped_early: bool
    best_conflicts_history: List[int]


def run_sa(graph: GraphData, cfg: SAConfig) -> SAResult:
    rnd = random.Random(cfg.seed)
    n = graph.n_vertices
    k = cfg.k_colors

    # initial random solution
    chrom = [rnd.randrange(k) for _ in range(n)]
    cur_conf = count_conflicts(graph, chrom)

    best = chrom[:]
    best_conf = cur_conf

    T = cfg.start_temp
    hist: List[int] = [best_conf]

    for it in range(1, cfg.max_iters + 1):
        if best_conf <= cfg.target_conflicts:
            return SAResult(
                best_chrom=best,
                best_conflicts=best_conf,
                best_colors_used=used_colors(best),
                iters_run=it,
                stopped_early=True,
                best_conflicts_history=hist,
            )

        # pick a conflicted vertex if possible, else random
        bads = conflicted_vertices(graph, chrom)
        v = rnd.choice(bads) if bads else rnd.randrange(n)

        old_color = chrom[v]
        new_color = rnd.randrange(k - 1)
        if new_color >= old_color:
            new_color += 1

        # compute delta conflicts efficiently (only edges touching v)
        old_local = 0
        new_local = 0
        for nb in graph.adjacency[v]:
            if chrom[nb] == old_color:
                old_local += 1
            if chrom[nb] == new_color:
                new_local += 1

        delta = new_local - old_local  # change in conflicts count
        accept = False
        if delta <= 0:
            accept = True
        else:
            # SA probability
            p = math.exp(-delta / max(T, 1e-12))
            if rnd.random() < p:
                accept = True

        if accept:
            chrom[v] = new_color
            cur_conf += delta

            if cur_conf < best_conf:
                best_conf = cur_conf
                best = chrom[:]

        # cooling
        T = max(cfg.end_temp, T * cfg.alpha)
        hist.append(best_conf)

    return SAResult(
        best_chrom=best,
        best_conflicts=best_conf,
        best_colors_used=used_colors(best),
        iters_run=cfg.max_iters,
        stopped_early=False,
        best_conflicts_history=hist,
    )
