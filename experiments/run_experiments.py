from __future__ import annotations

import csv
import os
from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt

from ga.graph_io import read_col
from ga.ga_runner import GAConfig, run_ga


def ensure_dirs():
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)


def make_configs(k_colors: int) -> List[GAConfig]:
    """
    At least 6 combinations (selection/crossover/mutation/pop size/mutation rate).
    Keep generations moderate so you can iterate fast; you can increase later.
    """
    base = dict(
        k_colors=k_colors,
        penalty=1000.0,
        generations=2000,
        elitism=1,
        patience=250,
        crossover_rate=0.9,
        tournament_k=3,
    )

    return [
        GAConfig(**base, pop_size=100, selection="tournament", crossover="one_point", mutation="one_gene", mutation_rate=0.25),
        GAConfig(**base, pop_size=100, selection="tournament", crossover="uniform",  mutation="one_gene", mutation_rate=0.25),
        GAConfig(**base, pop_size=150, selection="tournament", crossover="one_point", mutation="per_gene", mutation_rate=0.02),
        GAConfig(**base, pop_size=150, selection="tournament", crossover="uniform",  mutation="per_gene", mutation_rate=0.02),
        GAConfig(**base, pop_size=120, selection="roulette",   crossover="one_point", mutation="per_gene", mutation_rate=0.03),
        GAConfig(**base, pop_size=120, selection="roulette",   crossover="uniform",  mutation="per_gene", mutation_rate=0.03),
    ]


def run_dataset(dataset_path: str, tag: str, k_colors: int):
    graph = read_col(dataset_path)
    cfgs = make_configs(k_colors)

    rows: List[Dict[str, Any]] = []
    best_by_score: Tuple[int, int] | None = None
    best_run = None

    for i, cfg in enumerate(cfgs, start=1):
        # fixed seed per config for reproducibility
        result = run_ga(graph, cfg, seed=i)

        conflicts = result.best_eval.conflicts
        colors_used = result.best_eval.n_colors_used

        row = {
            "dataset": tag,
            "vertices": graph.n_vertices,
            "edges": len(graph.edges),
            "config_id": i,
            "pop_size": cfg.pop_size,
            "selection": cfg.selection,
            "crossover": cfg.crossover,
            "mutation": cfg.mutation,
            "mutation_rate": cfg.mutation_rate,
            "crossover_rate": cfg.crossover_rate,
            "elitism": cfg.elitism,
            "generations_run": result.generations_run,
            "stopped_early": result.stopped_early,
            "best_conflicts": conflicts,
            "best_colors_used": colors_used,
            "best_fitness": result.best_eval.fitness,
        }
        rows.append(row)

        # Primary: minimize conflicts, Secondary: minimize colors
        score = (conflicts, colors_used)
        if best_by_score is None or score < best_by_score:
            best_by_score = score
            best_run = (cfg, result)

        print(f"[{tag}] cfg {i}: conflicts={conflicts} colors={colors_used} gens={result.generations_run}")

    # Save results CSV
    out_csv = f"results/{tag}_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Plot best fitness evolution
    assert best_run is not None
    cfg_best, res_best = best_run

    plt.figure()
    plt.plot(res_best.best_fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness so far")
    plt.title(
        f"{tag} fitness evolution\n"
        f"best: conflicts={res_best.best_eval.conflicts}, colors={res_best.best_eval.n_colors_used} | "
        f"{cfg_best.selection}, {cfg_best.crossover}, {cfg_best.mutation}"
    )
    out_png = f"plots/{tag}_fitness_evolution.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_png}")
    print("Best config:", cfg_best)
    print("Best result:", res_best.best_eval)


def main():
    ensure_dirs()

    # Put your .col files in data/ and update names here
    datasets = [
        ("data/small.col", "small", 20),
        ("data/medium.col", "medium", 40),
        ("data/large.col", "large", 80),
    ]

    for path, tag, k_colors in datasets:
        if not os.path.exists(path):
            print(f"WARNING: file not found: {path} (skip)")
            continue
        run_dataset(path, tag, k_colors)


if __name__ == "__main__":
    main()
