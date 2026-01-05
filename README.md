# A2 – Graph Coloring with Genetic Algorithms

Neural and Evolutionary Computation  
Universitat Rovira i Virgili (URV)

This project solves the Graph Coloring Problem using a Genetic Algorithm (GA).
The objective is to minimize the number of colors while ensuring that no two
adjacent vertices share the same color.

## Structure
- `ga/` : Genetic Algorithm implementation
- `experiments/` : Experiment scripts and parameter sweeps
- `data/` : Graph datasets (.col format)
- `results/` : CSV experiment results
- `plots/` : Fitness evolution plots

## How to run
```bash
pip install -r requirements.txt
python experiments/run_experiments.py
