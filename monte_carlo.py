"""
monte_carlo.py
--------------
Runs multiple independent trials of the spread simulation and aggregates results.

Each trial:
  1. Creates a fresh GridEnvironment with the same host density but re-randomised temperature
  2. Randomises wind direction and speed
  3. Runs the SpreadEngine for t_max steps
  4. Records the infestation history and key metrics

Returns a summary that visualize.py can use to produce heatmaps, charts, and
the variable sensitivity ranking.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from grid import GridEnvironment
from spread_engine import SpreadEngine


class MonteCarloSimulation:
    """
    Runs M independent trials of the invasive species spread simulation.

    Parameters
    ----------
    n               : int   — grid side length
    m               : int   — number of Monte Carlo trials
    t_max           : int   — time steps per trial
    base_spread_prob: float — base spread probability
    t_min, t_max_temp: float — viable temperature range for pest
    rho_min         : float — minimum host density to sustain infestation
    depletion_rate  : float — host tree depletion per step
    wind_speed_range: (float, float) — min/max wind speed randomised per trial
    temp_mean       : float — mean temperature across grid
    temp_std        : float — std deviation of temperature across grid
    seed            : int or None — master seed for reproducibility
    """

    def __init__(
        self,
        n: int = 40,
        m: int = 50,
        t_max: int = 60,
        base_spread_prob: float = 0.4,
        t_min: float = 10.0,
        t_max_temp: float = 30.0,
        rho_min: float = 0.2,
        depletion_rate: float = 0.1,
        wind_speed_range: tuple = (0.0, 5.0),
        temp_mean: float = 20.0,
        temp_std: float = 3.0,
        seed: int | None = 42,
    ):
        self.n                = n
        self.m                = m
        self.t_max            = t_max
        self.base_spread_prob = base_spread_prob
        self.t_min            = t_min
        self.t_max_temp       = t_max_temp
        self.rho_min          = rho_min
        self.depletion_rate   = depletion_rate
        self.wind_speed_range = wind_speed_range
        self.temp_mean        = temp_mean
        self.temp_std         = temp_std
        self.rng              = np.random.default_rng(seed)

        # Generate one shared host density map used across all trials
        # (represents the real landscape — only weather/wind changes)
        self.base_host_density = self.rng.uniform(0.3, 1.0, size=(n, n))

    # ── Trial runner ──────────────────────────────────────────────────────────

    def _run_trial(self, trial_seed: int) -> dict:
        """Run a single trial and return its results."""

        # Fresh grid — same host density, new randomised temperature
        grid = GridEnvironment(
            n=self.n,
            host_density=self.base_host_density.copy(),
            temp_mean=self.temp_mean,
            temp_std=self.temp_std,
            seed=trial_seed,
        )

        # Randomise wind for this trial
        wind_speed = float(self.rng.uniform(*self.wind_speed_range))
        wind_angle = float(self.rng.uniform(0, 2 * np.pi))
        wind_vector = (
            wind_speed * np.cos(wind_angle),
            wind_speed * np.sin(wind_angle),
        )

        # Set up engine
        engine = SpreadEngine(
            grid=grid,
            t_min=self.t_min,
            t_max=self.t_max_temp,
            rho_min=self.rho_min,
            base_spread_prob=self.base_spread_prob,
            depletion_rate=self.depletion_rate,
            wind_vector=wind_vector,
            seed=trial_seed + 1000,
        )

        # Run and collect history
        history = engine.run(t_max=self.t_max)

        # Spread velocity = average new cells infested per step
        infested_counts = [h["infested"] for h in history]
        deltas = [max(0, infested_counts[i] - infested_counts[i - 1])
                  for i in range(1, len(infested_counts))]
        spread_velocity = float(np.mean(deltas)) if deltas else 0.0

        return {
            "trial_seed":      trial_seed,
            "wind_speed":      wind_speed,
            "wind_angle":      wind_angle,
            "wind_vector":     wind_vector,
            "history":         history,
            "infested_counts": infested_counts,
            "spread_velocity": spread_velocity,
            "final_infested":  history[-1]["infested"],
            "final_depleted":  history[-1]["depleted"],
            "final_snapshot":  history[-1]["snapshot"],
        }

    # ── Main runner ───────────────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> dict:
        """
        Run all M trials and return aggregated results.

        Returns
        -------
        results : dict with keys
            - trials          : list of individual trial result dicts
            - mean_infested   : (t_max+1,) array — mean infested cells per step
            - std_infested    : (t_max+1,) array — std dev across trials
            - velocities      : (M,) array — spread velocity per trial
            - sensitivity     : dict — correlation of each variable with velocity
            - params          : dict — simulation parameters used
        """
        trials = []

        for i in range(self.m):
            if verbose:
                print(f"  Trial {i + 1:>3} / {self.m}", end="\r")
            trial_seed = int(self.rng.integers(0, 999999))
            result = self._run_trial(trial_seed)
            trials.append(result)

        if verbose:
            print(f"  Completed {self.m} trials.          ")

        # ── Aggregate infested counts over time ───────────────────────────────
        all_counts = np.array([t["infested_counts"] for t in trials])
        mean_infested = np.mean(all_counts, axis=0)
        std_infested  = np.std(all_counts, axis=0)

        # ── Spread velocities ─────────────────────────────────────────────────
        velocities = np.array([t["spread_velocity"] for t in trials])

        # ── Variable sensitivity ──────────────────────────────────────────────
        # Pearson correlation between each input variable and spread velocity.
        # Higher absolute correlation = more influential variable.
        wind_speeds  = np.array([t["wind_speed"] for t in trials])
        final_counts = np.array([t["final_infested"] for t in trials])

        def safe_corr(x, y):
            if np.std(x) == 0 or np.std(y) == 0:
                return 0.0
            return float(np.corrcoef(x, y)[0, 1])

        sensitivity = {
            "wind_speed":      abs(safe_corr(wind_speeds, velocities)),
            "base_spread_prob": 1.0,   # constant across trials — acts as baseline
            "host_density_mean": abs(safe_corr(
                [np.mean(self.base_host_density)] * self.m, velocities
            )),
        }

        # Normalise sensitivity scores to [0, 1]
        max_s = max(sensitivity.values()) or 1.0
        sensitivity = {k: round(v / max_s, 3) for k, v in sensitivity.items()}

        return {
            "trials":        trials,
            "mean_infested": mean_infested,
            "std_infested":  std_infested,
            "velocities":    velocities,
            "sensitivity":   sensitivity,
            "params": {
                "n":                self.n,
                "m":                self.m,
                "t_max":            self.t_max,
                "base_spread_prob": self.base_spread_prob,
                "rho_min":          self.rho_min,
                "temp_mean":        self.temp_mean,
                "temp_std":         self.temp_std,
                "wind_speed_range": self.wind_speed_range,
            },
        }
