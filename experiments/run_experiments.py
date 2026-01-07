from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt

from ga.graph_io import read_col
from ga.ga_runner import GAConfig, run_ga
from ga.sa_runner import SAConfig, run_sa
from ga.tabu_runner import TabuConfig, run_tabu


def ensure_dirs():
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)


def make_ga_configs(k_colors: int, generations: int, patience: int, use_repair: bool, repair_steps: int) -> List[GAConfig]:
    base = dict(
        k_colors=k_colors,
        penalty=1000.0,
        generations=generations,
        elitism=1,
        patience=patience,
        crossover_rate=0.9,
        tournament_k=3,
        use_repair=use_repair,
        repair_steps=repair_steps,
    )
    return [
        GAConfig(**base, pop_size=100, selection="tournament", crossover="one_point", mutation="one_gene", mutation_rate=0.25),
        GAConfig(**base, pop_size=100, selection="tournament", crossover="uniform",  mutation="one_gene", mutation_rate=0.25),
        GAConfig(**base, pop_size=150, selection="tournament", crossover="one_point", mutation="per_gene", mutation_rate=0.02),
        GAConfig(**base, pop_size=150, selection="tournament", crossover="uniform",  mutation="per_gene", mutation_rate=0.02),
        GAConfig(**base, pop_size=120, selection="roulette",   crossover="one_point", mutation="per_gene", mutation_rate=0.03),
        GAConfig(**base, pop_size=120, selection="roulette",   crossover="uniform",  mutation="per_gene", mutation_rate=0.03),
        # optional: add 2 extra configs if you want
        GAConfig(**base, pop_size=300, selection="tournament", crossover="uniform",  mutation="per_gene", mutation_rate=0.05),
        GAConfig(**base, pop_size=400, selection="tournament", crossover="uniform",  mutation="per_gene", mutation_rate=0.07),
    ]


def save_csv(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_history(path: str, y: List[float], xlabel: str, ylabel: str, title: str):
    plt.figure()
    plt.plot(y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(path, dpi=200)
    plt.close()


def run_ga_block(graph_path: str, tag: str, generations: int, k_colors: int, use_repair: bool, repair_steps: int):
    graph = read_col(graph_path)
    cfgs = make_ga_configs(k_colors, generations, patience=max(50, generations // 4), use_repair=use_repair, repair_steps=repair_steps)

    rows: List[Dict[str, Any]] = []
    best_score: Tuple[int, int] | None = None
    best_cfg = None
    best_res = None

    for i, cfg in enumerate(cfgs, start=1):
        res = run_ga(graph, cfg, seed=i)
        conf = res.best_eval.conflicts
        colors_used = res.best_eval.n_colors_used

        rows.append({
            "method": "GA",
            "dataset": tag,
            "vertices": graph.n_vertices,
            "edges": len(graph.edges),
            "config_id": i,
            "k_colors": cfg.k_colors,
            "pop_size": cfg.pop_size,
            "selection": cfg.selection,
            "crossover": cfg.crossover,
            "mutation": cfg.mutation,
            "mutation_rate": cfg.mutation_rate,
            "crossover_rate": cfg.crossover_rate,
            "elitism": cfg.elitism,
            "generations_target": cfg.generations,
            "generations_run": res.generations_run,
            "stopped_early": res.stopped_early,
            "use_repair": cfg.use_repair,
            "repair_steps": cfg.repair_steps,
            "best_conflicts": conf,
            "best_colors_used": colors_used,
            "best_fitness": res.best_eval.fitness,
        })

        score = (conf, colors_used)
        if best_score is None or score < best_score:
            best_score = score
            best_cfg = cfg
            best_res = res

        print(f"[{tag}] GA cfg {i}: conflicts={conf} colors={colors_used} gens={res.generations_run}")

    out_csv = f"results/{tag}_ga_k{k_colors}_results.csv"
    save_csv(out_csv, rows)

    assert best_cfg is not None and best_res is not None
    out_png = f"plots/{tag}_ga_k{k_colors}_fitness.png"
    plot_history(
        out_png,
        best_res.best_fitness_history,
        xlabel="Generation",
        ylabel="Best fitness so far",
        title=f"{tag} GA fitness (k={k_colors}) | best conflicts={best_res.best_eval.conflicts}, colors={best_res.best_eval.n_colors_used}"
    )

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")
    print("Best GA config:", best_cfg)
    print("Best GA result:", best_res.best_eval)


def run_sa_block(graph_path: str, tag: str, k_colors: int):
    graph = read_col(graph_path)

    cfg = SAConfig(
        k_colors=k_colors,
        max_iters=200_000,
        start_temp=2.0,
        end_temp=1e-4,
        alpha=0.9995,
        seed=1,
        target_conflicts=0,
    )
    res = run_sa(graph, cfg)

    rows = [{
        "method": "SA",
        "dataset": tag,
        "vertices": graph.n_vertices,
        "edges": len(graph.edges),
        "k_colors": k_colors,
        "iters_run": res.iters_run,
        "stopped_early": res.stopped_early,
        "best_conflicts": res.best_conflicts,
        "best_colors_used": res.best_colors_used,
    }]

    out_csv = f"results/{tag}_sa_k{k_colors}_results.csv"
    save_csv(out_csv, rows)

    out_png = f"plots/{tag}_sa_k{k_colors}_conflicts.png"
    plot_history(
        out_png,
        res.best_conflicts_history,
        xlabel="Iteration",
        ylabel="Best conflicts so far",
        title=f"{tag} SA conflicts (k={k_colors}) | best={res.best_conflicts}"
    )

    print(f"[{tag}] SA: conflicts={res.best_conflicts} colors={res.best_colors_used} iters={res.iters_run}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")


def run_tabu_block(graph_path: str, tag: str, k_colors: int):
    graph = read_col(graph_path)

    cfg = TabuConfig(
        k_colors=k_colors,
        max_iters=200_000,
        tabu_tenure=20,
        candidate_vertices=60,
        seed=1,
        target_conflicts=0,
    )
    res = run_tabu(graph, cfg)

    rows = [{
        "method": "TABU",
        "dataset": tag,
        "vertices": graph.n_vertices,
        "edges": len(graph.edges),
        "k_colors": k_colors,
        "iters_run": res.iters_run,
        "stopped_early": res.stopped_early,
        "best_conflicts": res.best_conflicts,
        "best_colors_used": res.best_colors_used,
    }]

    out_csv = f"results/{tag}_tabu_k{k_colors}_results.csv"
    save_csv(out_csv, rows)

    out_png = f"plots/{tag}_tabu_k{k_colors}_conflicts.png"
    plot_history(
        out_png,
        res.best_conflicts_history,
        xlabel="Iteration",
        ylabel="Best conflicts so far",
        title=f"{tag} Tabu conflicts (k={k_colors}) | best={res.best_conflicts}"
    )

    print(f"[{tag}] TABU: conflicts={res.best_conflicts} colors={res.best_colors_used} iters={res.iters_run}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, help="Path to .col file")
    parser.add_argument("--generations", type=int, default=500, help="GA generations")
    parser.add_argument("--k_colors", type=int, default=20, help="Number of colors (k)")
    parser.add_argument("--method", choices=["ga", "sa", "tabu", "all"], default="ga")
    parser.add_argument("--use_repair", action="store_true")
    parser.add_argument("--repair_steps", type=int, default=80)
    args = parser.parse_args()

    ensure_dirs()

    tag = os.path.splitext(os.path.basename(args.graph))[0]

    if args.method in ("ga", "all"):
        run_ga_block(args.graph, tag, args.generations, args.k_colors, args.use_repair, args.repair_steps)

    if args.method in ("sa", "all"):
        run_sa_block(args.graph, tag, args.k_colors)

    if args.method in ("tabu", "all"):
        run_tabu_block(args.graph, tag, args.k_colors)


if __name__ == "__main__":
    main()
