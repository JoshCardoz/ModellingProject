"""
validation.py
-------------
Generates a synthetic benchmark dataset based on real-world EAB spread rates
from published literature, and compares it against our simulation output.

Real-world references used:
  - EAB spreads ~40 km/year under favourable conditions (Orlova-Bienkowskaja 2018)
  - All ash trees in an infested area expected to die within ~10 years (USDA)
  - EAB first detected Michigan 2002; reached 36 U.S. states by ~2023
    (~21 years, ~1,500 km range expansion = ~71 km/year at outer edge)

How we map real rates to our grid:
  - Each grid cell represents a patch of forest
  - We define 1 time step = 1 week of pest activity during the growing season
    (~26 active weeks/year)
  - We calibrate cell size so that the simulated spread velocity matches
    the real-world rate of ~40 km/year = ~1.54 km/week
  - This gives us a synthetic "expected" infestation curve to compare against
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import csv


# ── Real-world EAB parameters (from literature) ───────────────────────────────
EAB_SPREAD_KM_PER_YEAR   = 40.0    # conservative spread rate (Orlova-Bienkowskaja 2018)
EAB_ACTIVE_WEEKS_PER_YEAR = 26     # approximate active season length
EAB_SPREAD_KM_PER_STEP   = EAB_SPREAD_KM_PER_YEAR / EAB_ACTIVE_WEEKS_PER_YEAR
# ~1.54 km per week

EAB_DEPLETION_YEARS = 10.0         # years until full area depletion (USDA)
EAB_DEPLETION_STEPS = EAB_DEPLETION_YEARS * EAB_ACTIVE_WEEKS_PER_YEAR
# ~260 steps until full depletion


def generate_synthetic_benchmark(
    n: int,
    t_max: int,
    cell_size_km: float = 1.0,
) -> dict:
    """
    Generate a synthetic infestation curve based on real EAB spread rates.

    Models the pest as spreading radially outward from a central ignition point
    at the real-world rate, producing an expected number of infested cells at
    each time step.

    Parameters
    ----------
    n            : int   — grid side length (must match simulation)
    t_max        : int   — number of time steps (must match simulation)
    cell_size_km : float — how many km each grid cell represents

    Returns
    -------
    dict with keys:
        - steps          : list of time step indices
        - infested_counts: expected infested cells at each step
        - notes          : explanation string for the report
    """
    steps = list(range(t_max + 1))
    infested_counts = []

    # Spread radius in cells at each time step
    spread_rate_cells_per_step = EAB_SPREAD_KM_PER_STEP / cell_size_km

    total_cells = n * n
    center = n / 2.0

    for t in steps:
        # Radius of infestation in cells at this time step
        radius = spread_rate_cells_per_step * t

        # Count cells within this radius (circular spread from centre)
        # This is an approximation: pi * r^2, capped at total cells
        infested = min(np.pi * radius ** 2, total_cells * 0.85)

        # Apply a logistic growth dampening — spread slows as forest depletes
        # Uses a logistic curve that saturates at 85% of total cells
        carrying_capacity = total_cells * 0.85
        growth_rate = 0.15
        logistic = carrying_capacity / (
            1 + np.exp(-growth_rate * (t - t_max / 2))
        )

        # Blend linear radial spread with logistic growth
        blended = 0.6 * infested + 0.4 * logistic
        infested_counts.append(min(blended, carrying_capacity))

    return {
        "steps":           steps,
        "infested_counts": infested_counts,
        "cell_size_km":    cell_size_km,
        "spread_rate":     EAB_SPREAD_KM_PER_STEP,
        "notes": (
            f"Synthetic benchmark based on EAB spread rate of "
            f"{EAB_SPREAD_KM_PER_YEAR} km/year ({EAB_SPREAD_KM_PER_STEP:.2f} km/week). "
            f"Each grid cell = {cell_size_km} km. "
            f"1 time step = 1 active week. "
            f"Source: Orlova-Bienkowskaja (2018), USDA Forest Service."
        ),
    }


def compare_to_benchmark(
    simulation_results: dict,
    cell_size_km: float = 1.0,
    save: bool = True,
    show: bool = False,
    output_dir: str = "outputs",
) -> dict:
    """
    Compare simulation output against the synthetic EAB benchmark.

    Parameters
    ----------
    simulation_results : dict returned by MonteCarloSimulation.run()
    cell_size_km       : float — km per grid cell (tune this to match your grid)
    save               : bool — save chart to output_dir
    show               : bool — display chart interactively
    output_dir         : str  — folder to save outputs

    Returns
    -------
    comparison : dict with RMSE, correlation, and benchmark data
    """
    os.makedirs(output_dir, exist_ok=True)

    n     = simulation_results["params"]["n"]
    t_max = simulation_results["params"]["t_max"]
    mean  = simulation_results["mean_infested"]
    std   = simulation_results["std_infested"]

    # Generate benchmark
    benchmark = generate_synthetic_benchmark(n, t_max, cell_size_km)
    bench_counts = np.array(benchmark["infested_counts"])

    # ── Metrics ───────────────────────────────────────────────────────────────
    # Normalise both to [0,1] before comparing (different absolute scales)
    sim_norm   = mean / (n * n)
    bench_norm = bench_counts / (n * n)

    rmse = float(np.sqrt(np.mean((sim_norm - bench_norm) ** 2)))
    corr = float(np.corrcoef(sim_norm, bench_norm)[0, 1])

    # ── Plot ──────────────────────────────────────────────────────────────────
    steps = np.arange(len(mean))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: raw cell counts
    ax = axes[0]
    ax.fill_between(steps, mean - std, mean + std,
                    color="#1D9E75", alpha=0.2, label="Simulation ±1 std")
    ax.plot(steps, mean, color="#1D9E75", linewidth=2, label="Simulation mean")
    ax.plot(benchmark["steps"], bench_counts,
            color="#E24B4A", linewidth=2, linestyle="--", label="EAB benchmark")
    ax.set_xlabel("Time step (1 step = 1 active week)")
    ax.set_ylabel("Infested cells")
    ax.set_title("Simulation vs EAB benchmark", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Right: normalised (% of grid)
    ax2 = axes[1]
    ax2.fill_between(steps,
                     (mean - std) / (n * n) * 100,
                     (mean + std) / (n * n) * 100,
                     color="#1D9E75", alpha=0.2)
    ax2.plot(steps, sim_norm * 100, color="#1D9E75",
             linewidth=2, label="Simulation mean")
    ax2.plot(benchmark["steps"], bench_norm * 100,
             color="#E24B4A", linewidth=2, linestyle="--", label="EAB benchmark")
    ax2.set_xlabel("Time step (1 step = 1 active week)")
    ax2.set_ylabel("% of grid infested")
    ax2.set_title("Normalised comparison (% of grid)", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    # Metrics annotation
    fig.text(0.5, -0.04,
             f"RMSE (normalised): {rmse:.4f}   |   Pearson correlation: {corr:.4f}   |   "
             f"Cell size: {cell_size_km} km   |   {benchmark['notes']}",
             ha="center", fontsize=7.5, color="#5F5E5A",
             wrap=True)

    plt.tight_layout()

    path = os.path.join(output_dir, "validation_comparison.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)

    # ── Export benchmark CSV ──────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "benchmark_data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "benchmark_infested", "simulation_mean",
                         "simulation_std", "benchmark_pct", "simulation_pct"])
        for i, (b, m, s) in enumerate(zip(bench_counts, mean, std)):
            writer.writerow([
                i,
                round(b, 2),
                round(m, 2),
                round(s, 2),
                round(b / (n * n) * 100, 2),
                round(m / (n * n) * 100, 2),
            ])
    print(f"  Saved: {csv_path}")

    return {
        "rmse":        rmse,
        "correlation": corr,
        "benchmark":   benchmark,
        "chart_path":  path,
        "csv_path":    csv_path,
    }


def print_comparison_summary(comparison: dict) -> None:
    """Print a readable summary of the validation results."""
    print("\nValidation summary:")
    print(f"  Pearson correlation (sim vs benchmark) : {comparison['correlation']:.4f}")
    print(f"  RMSE (normalised 0-1)                 : {comparison['rmse']:.4f}")
    print(f"\n  {comparison['benchmark']['notes']}")
    print("\n  Interpretation:")
    corr = comparison["correlation"]
    if corr > 0.85:
        print("  Correlation > 0.85 — simulation tracks benchmark closely.")
    elif corr > 0.6:
        print("  Correlation 0.6-0.85 — reasonable agreement with real-world rates.")
    else:
        print("  Correlation < 0.6 — consider adjusting cell_size_km parameter.")
