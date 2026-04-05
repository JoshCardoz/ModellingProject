"""
visualize.py
------------
Takes Monte Carlo results and produces:
  1. An infestation heatmap (final state of a chosen trial)
  2. A spread-over-time line chart (mean +/- std across all trials)
  3. A variable sensitivity bar chart
  4. A CSV export of infested area over time

Usage:
  from visualize import Visualizer
  viz = Visualizer(results, output_dir="outputs")
  viz.plot_heatmap()
  viz.plot_spread_over_time()
  viz.plot_sensitivity()
  viz.export_csv()
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


# Cell status constants (must match grid.py)
SUSCEPTIBLE = 0
INFESTED    = 1
DEPLETED    = 2


class Visualizer:
    """
    Produces all output charts and data files from Monte Carlo results.

    Parameters
    ----------
    results    : dict returned by MonteCarloSimulation.run()
    output_dir : str — folder where files are saved (created if missing)
    """

    def __init__(self, results: dict, output_dir: str = "outputs"):
        self.results    = results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Colour map for the heatmap: green=susceptible, red=infested, dark=depleted
        self.status_cmap = mcolors.ListedColormap(["#5DCAA5", "#E24B4A", "#2C2C2A"])
        self.status_norm = mcolors.BoundaryNorm([0, 1, 2, 3], 3)

    # ── 1. Heatmap ────────────────────────────────────────────────────────────

    def plot_heatmap(self, trial_index: int = 0, step: int = -1,
                     save: bool = True, show: bool = False) -> str:
        """
        Plot the infestation status grid for a single trial at a given step.

        Parameters
        ----------
        trial_index : which trial to show (default: first trial)
        step        : which time step (-1 = final state)
        """
        trial   = self.results["trials"][trial_index]
        snapshot = trial["history"][step]["snapshot"]
        actual_step = trial["history"][step]["step"]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(snapshot, cmap=self.status_cmap, norm=self.status_norm,
                  interpolation="nearest")

        legend_elements = [
            Patch(facecolor="#5DCAA5", label="Susceptible"),
            Patch(facecolor="#E24B4A", label="Infested"),
            Patch(facecolor="#2C2C2A", label="Depleted"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
        ax.set_title(f"Infestation heatmap — trial {trial_index + 1}, step {actual_step}",
                     fontsize=12)
        ax.set_xlabel("Grid column")
        ax.set_ylabel("Grid row")
        ax.tick_params(labelsize=8)

        path = os.path.join(self.output_dir, f"heatmap_trial{trial_index + 1}_step{actual_step}.png")
        if save:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")
        if show:
            plt.show()
        plt.close(fig)
        return path

    def plot_heatmap_progression(self, trial_index: int = 0,
                                  steps: list | None = None,
                                  save: bool = True, show: bool = False) -> str:
        """
        Plot a row of heatmap snapshots showing spread over time for one trial.
        """
        trial   = self.results["trials"][trial_index]
        history = trial["history"]
        t_max   = self.results["params"]["t_max"]

        if steps is None:
            # Pick 4 evenly spaced steps including start and end
            idxs = [0,
                    len(history) // 3,
                    2 * len(history) // 3,
                    len(history) - 1]
        else:
            idxs = [min(s, len(history) - 1) for s in steps]

        fig, axes = plt.subplots(1, len(idxs), figsize=(4 * len(idxs), 4))
        fig.suptitle(f"Spread progression — trial {trial_index + 1}", fontsize=12)

        for ax, idx in zip(axes, idxs):
            snap = history[idx]["snapshot"]
            step = history[idx]["step"]
            ax.imshow(snap, cmap=self.status_cmap, norm=self.status_norm,
                      interpolation="nearest")
            ax.set_title(f"Step {step}", fontsize=10)
            ax.axis("off")

        legend_elements = [
            Patch(facecolor="#5DCAA5", label="Susceptible"),
            Patch(facecolor="#E24B4A", label="Infested"),
            Patch(facecolor="#2C2C2A", label="Depleted"),
        ]
        fig.legend(handles=legend_elements, loc="lower center",
                   ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

        path = os.path.join(self.output_dir, f"progression_trial{trial_index + 1}.png")
        if save:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")
        if show:
            plt.show()
        plt.close(fig)
        return path

    # ── 2. Spread over time ───────────────────────────────────────────────────

    def plot_spread_over_time(self, save: bool = True,
                              show: bool = False) -> str:
        """
        Line chart of mean infested cells over time with +/- 1 std band.
        """
        mean = self.results["mean_infested"]
        std  = self.results["std_infested"]
        steps = np.arange(len(mean))

        fig, ax = plt.subplots(figsize=(8, 4))

        # Individual trial lines (faint)
        for trial in self.results["trials"]:
            ax.plot(trial["infested_counts"], color="#5DCAA5", alpha=0.12, linewidth=0.8)

        # Mean line + std band
        ax.fill_between(steps, mean - std, mean + std,
                        color="#1D9E75", alpha=0.2, label="±1 std dev")
        ax.plot(steps, mean, color="#1D9E75", linewidth=2, label="Mean infested cells")

        total_cells = self.results["params"]["n"] ** 2
        ax.set_ylim(0, total_cells)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Infested cells")
        ax.set_title(f"Spread over time — {self.results['params']['m']} Monte Carlo trials",
                     fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        path = os.path.join(self.output_dir, "spread_over_time.png")
        if save:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")
        if show:
            plt.show()
        plt.close(fig)
        return path

    # ── 3. Spread velocity histogram ──────────────────────────────────────────

    def plot_velocity_histogram(self, save: bool = True,
                                show: bool = False) -> str:
        """
        Histogram of spread velocity (avg new cells/step) across all trials.
        """
        velocities = self.results["velocities"]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(velocities, bins=15, color="#5DCAA5", edgecolor="#0F6E56", linewidth=0.5)
        ax.axvline(np.mean(velocities), color="#E24B4A", linewidth=1.5,
                   linestyle="--", label=f"Mean: {np.mean(velocities):.1f}")
        ax.set_xlabel("Spread velocity (cells / step)")
        ax.set_ylabel("Number of trials")
        ax.set_title("Distribution of spread velocity across trials", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        path = os.path.join(self.output_dir, "velocity_histogram.png")
        if save:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")
        if show:
            plt.show()
        plt.close(fig)
        return path

    # ── 4. Sensitivity ranking ────────────────────────────────────────────────

    def plot_sensitivity(self, save: bool = True, show: bool = False) -> str:
        """
        Horizontal bar chart of variable sensitivity scores.
        """
        sensitivity = self.results["sensitivity"]
        labels = {
            "wind_speed":        "Wind speed",
            "host_density_mean": "Host tree density",
            "base_spread_prob":  "Base spread probability",
        }

        names  = [labels.get(k, k) for k in sensitivity]
        scores = list(sensitivity.values())
        colours = ["#1D9E75" if s == max(scores) else "#5DCAA5" for s in scores]

        fig, ax = plt.subplots(figsize=(7, 3))
        bars = ax.barh(names, scores, color=colours, edgecolor="#0F6E56", linewidth=0.5)
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Sensitivity score (correlation with spread velocity)")
        ax.set_title("Variable sensitivity ranking", fontsize=12)
        ax.grid(axis="x", linestyle="--", alpha=0.4)

        path = os.path.join(self.output_dir, "sensitivity.png")
        if save:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")
        if show:
            plt.show()
        plt.close(fig)
        return path

    # ── 5. CSV export ─────────────────────────────────────────────────────────

    def export_csv(self) -> str:
        """
        Export a CSV of mean infested cells over time.
        Columns: step, mean_infested, std_infested, min_infested, max_infested
        """
        mean = self.results["mean_infested"]
        std  = self.results["std_infested"]
        all_counts = np.array([t["infested_counts"] for t in self.results["trials"]])
        mins = np.min(all_counts, axis=0)
        maxs = np.max(all_counts, axis=0)

        path = os.path.join(self.output_dir, "spread_data.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "mean_infested", "std_infested",
                             "min_infested", "max_infested"])
            for i, (m, s, mn, mx) in enumerate(zip(mean, std, mins, maxs)):
                writer.writerow([i, round(m, 2), round(s, 2),
                                 int(mn), int(mx)])

        print(f"  Saved: {path}")
        return path

    # ── Run everything at once ────────────────────────────────────────────────

    def run_all(self, show: bool = False) -> None:
        """Generate all charts and the CSV in one call."""
        print("Generating outputs...")
        self.plot_heatmap(show=show)
        self.plot_heatmap_progression(show=show)
        self.plot_spread_over_time(show=show)
        self.plot_velocity_histogram(show=show)
        self.plot_sensitivity(show=show)
        self.export_csv()
        print("Done.")
