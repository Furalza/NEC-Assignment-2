from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

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


@dataclass
class TabuConfig:
    k_colors: int
    max_iters: int = 200_000
    tabu_tenure: int = 20
    candidate_vertices: int = 60  # sample this many conflicted vertices per iteration
    seed: int = 1
    target_conflicts: int = 0


@dataclass
class TabuResult:
    best_chrom: List[int]
    best_conflicts: int
    best_colors_used: int
    iters_run: int
    stopped_early: bool
    best_conflicts_history: List[int]


def run_tabu(graph: GraphData, cfg: TabuConfig) -> TabuResult:
    rnd = random.Random(cfg.seed)
    n = graph.n_vertices
    k = cfg.k_colors

    chrom = [rnd.randrange(k) for _ in range(n)]
    cur_conf = count_conflicts(graph, chrom)

    best = chrom[:]
    best_conf = cur_conf
    hist = [best_conf]

    # tabu[v][c] = iteration until which move "set v to color c" is tabu
    tabu = [[0] * k for _ in range(n)]

    for it in range(1, cfg.max_iters + 1):
        if best_conf <= cfg.target_conflicts:
            return TabuResult(best, best_conf, used_colors(best), it, True, hist)

        bads = conflicted_vertices(graph, chrom)
        if not bads:
            # already perfect
            best = chrom[:]
            best_conf = 0
            hist.append(0)
            return TabuResult(best, best_conf, used_colors(best), it, True, hist)

        # sample candidate vertices
        rnd.shuffle(bads)
        cand_vs = bads[: min(cfg.candidate_vertices, len(bads))]

        best_move: Tuple[int, int, int] | None = None  # (delta, v, new_color)
        best_move_conf = 10**9

        for v in cand_vs:
            old_color = chrom[v]

            # local old conflicts touching v
            old_local = 0
            for nb in graph.adjacency[v]:
                if chrom[nb] == old_color:
                    old_local += 1

            for new_color in range(k):
                if new_color == old_color:
                    continue

                # local new conflicts
                new_local = 0
                for nb in graph.adjacency[v]:
                    if chrom[nb] == new_color:
                        new_local += 1

                delta = new_local - old_local
                next_conf = cur_conf + delta

                is_tabu = tabu[v][new_color] > it
                aspiration = next_conf < best_conf  # allow tabu if it improves global best

                if is_tabu and not aspiration:
                    continue

                # choose move that minimizes next_conf (best improvement)
                if next_conf < best_move_conf:
                    best_move_conf = next_conf
                    best_move = (delta, v, new_color)

        # if all moves tabu, just do a random move on a random bad vertex
        if best_move is None:
            v = rnd.choice(bads)
            old_color = chrom[v]
            new_color = rnd.randrange(k - 1)
            if new_color >= old_color:
                new_color += 1

            # compute delta
            old_local = 0
            new_local = 0
            for nb in graph.adjacency[v]:
                if chrom[nb] == old_color:
                    old_local += 1
                if chrom[nb] == new_color:
                    new_local += 1
            delta = new_local - old_local
            best_move = (delta, v, new_color)

        delta, v, new_color = best_move
        old_color = chrom[v]

        # apply move
        chrom[v] = new_color
        cur_conf += delta

        # update tabu: forbid going back to old_color for a while
        tabu[v][old_color] = it + cfg.tabu_tenure

        if cur_conf < best_conf:
            best_conf = cur_conf
            best = chrom[:]

        hist.append(best_conf)

    return TabuResult(best, best_conf, used_colors(best), cfg.max_iters, False, hist)
