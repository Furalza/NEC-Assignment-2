from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Set


@dataclass(frozen=True)
class GraphData:
    """
    Graph stored in 0-based indexing.
    edges: list of undirected edges (u, v) with u != v
    adjacency: adjacency list for fast conflict counting, etc.
    """
    n_vertices: int
    edges: List[Tuple[int, int]]
    adjacency: List[Set[int]]


def read_col(path: str) -> GraphData:
    """
    Read a DIMACS .col graph coloring instance.

    Typical format:
      c comment lines
      p edge <n_vertices> <n_edges>
      e u v     (1-based vertex ids)

    We convert vertices to 0-based indexing.
    """
    n_vertices = 0
    edges: List[Tuple[int, int]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("c"):
                continue

            parts = line.split()
            if parts[0] == "p":
                # p edge n m
                # sometimes: p col n m (still fine)
                n_vertices = int(parts[2])
            elif parts[0] == "e":
                u = int(parts[1]) - 1
                v = int(parts[2]) - 1
                if u == v:
                    continue
                # store edges normalized (small, large) to reduce duplicates
                a, b = (u, v) if u < v else (v, u)
                edges.append((a, b))

    if n_vertices <= 0:
        raise ValueError(f"Could not parse 'p edge n m' line in file: {path}")

    # remove duplicates (some files can have repeated edges)
    edges = sorted(set(edges))

    adjacency: List[Set[int]] = [set() for _ in range(n_vertices)]
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)

    return GraphData(n_vertices=n_vertices, edges=edges, adjacency=adjacency)
