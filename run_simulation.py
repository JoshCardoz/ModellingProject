"""
run_simulation.py
-----------------
Main entry point. Run this file to execute the full simulation pipeline:
  1. Load real host density from Kaggle Forest Cover Type dataset
  2. Run Monte Carlo simulation (M trials)
  3. Generate all charts and CSV output
  4. Validate results against real-world EAB benchmark

Usage:
  python run_simulation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monte_carlo import MonteCarloSimulation
from visualize import Visualizer
from validation import compare_to_benchmark, print_comparison_summary
from data_loader import load_host_density


def main():
    print("=" * 50)
    print("Invasive Species Spread Simulation")
    print("Emerald Ash Borer — Monte Carlo Model")
    print("=" * 50)

    # ── Load real host density from Kaggle dataset ────────────────────────────
    N = 40
    dataset_path = "test.csv"

    if os.path.exists(dataset_path):
        print(f"\nLoading real forest data from {dataset_path}...")
        host_density = load_host_density(dataset_path, n=N)
        print("  Real Kaggle dataset loaded successfully.")
    else:
        print("\nDataset not found — using randomly generated host density.")
        print("  To use real data, place test.csv in this folder.")
        host_density = None  # MonteCarloSimulation will generate randomly

    # ── Run simulation ────────────────────────────────────────────────────────
    print("\nRunning Monte Carlo simulation...")
    sim = MonteCarloSimulation(
        n=N,
        m=50,
        t_max=60,
        base_spread_prob=0.7,
        t_min=10.0,
        t_max_temp=30.0,
        rho_min=0.1,
        depletion_rate=0.1,
        wind_speed_range=(0.0, 5.0),
        temp_mean=20.0,
        temp_std=3.0,
        seed=42,
    )

    # Plug in real host density if loaded
    if host_density is not None:
        sim.base_host_density = host_density

    results = sim.run(verbose=True)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\nResults summary:")
    print(f"  Trials completed    : {results['params']['m']}")
    print(f"  Grid size           : {results['params']['n']}x{results['params']['n']}")
    print(f"  Mean final infested : {results['mean_infested'][-1]:.0f} cells")
    print(f"  Mean spread velocity: {results['velocities'].mean():.2f} cells/step")
    print(f"\nSensitivity ranking:")
    for var, score in sorted(results["sensitivity"].items(),
                              key=lambda x: x[1], reverse=True):
        print(f"  {var:<25} {score:.3f}")

    # ── Generate all outputs ──────────────────────────────────────────────────
    print("\nGenerating charts and CSV...")
    viz = Visualizer(results, output_dir="outputs")
    viz.run_all(show=False)

    # ── Validate against benchmark ────────────────────────────────────────────
    print("\nRunning validation against EAB benchmark...")
    comparison = compare_to_benchmark(
        results,
        cell_size_km=1.0,
        output_dir="outputs",
    )
    print_comparison_summary(comparison)

    print("\nAll done! Check the 'outputs/' folder for your charts and CSV.")


if __name__ == "__main__":
    main()
