"""
data_loader.py
--------------
Loads the Kaggle Forest Cover Type dataset and converts it into a host
density grid suitable for use in the simulation.

Dataset: https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset
Each row = one 30x30m patch of forest in Roosevelt National Forest, Colorado.

How we derive host density:
  - Elevation: invasive pests thrive at mid-range elevations. Very high or
    very low elevation = lower host suitability.
  - Slope: flat terrain (low slope) allows easier spread. High slope = lower.
  - Soil type: certain soil types support denser host tree growth.
  - Wilderness area: more remote areas have higher undisturbed host density.

All factors are combined into a single host_density value in [0, 1] per cell,
which is then reshaped into an NxN grid for the simulation.

Usage:
  from data_loader import load_host_density
  host_density = load_host_density("test.csv", n=40)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd


# Cover types that are most susceptible to invasive forest pests
# (deciduous and mixed forest types are generally more vulnerable)
# 1=Spruce/Fir, 2=Lodgepole Pine, 3=Ponderosa Pine,
# 4=Cottonwood/Willow, 5=Aspen, 6=Douglas-fir, 7=Krummholz
SUSCEPTIBILITY = {
    1: 0.5,   # Spruce/Fir — moderate
    2: 0.6,   # Lodgepole Pine — moderate-high
    3: 0.8,   # Ponderosa Pine — high
    4: 0.9,   # Cottonwood/Willow — very high (riparian, easy spread)
    5: 0.85,  # Aspen — high
    6: 0.75,  # Douglas-fir — high
    7: 0.2,   # Krummholz — low (harsh high-altitude environment)
}


def load_host_density(
    filepath: str,
    n: int = 40,
    seed: int = 42,
) -> np.ndarray:
    """
    Load the forest cover CSV and return an (n x n) host density array.

    Parameters
    ----------
    filepath : str  — path to train.csv or test.csv
    n        : int  — grid side length to produce
    seed     : int  — for reproducible random sampling

    Returns
    -------
    host_density : (n x n) float array with values in [0, 1]
    """
    print(f"  Loading dataset: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Rows loaded: {len(df):,}")

    # ── Step 1: Elevation factor ──────────────────────────────────────────────
    # Pests thrive at mid-range elevations (2000-2800m in this dataset)
    # Normalise elevation to [0,1] then apply an inverted-U curve
    elev = df["Elevation"].values.astype(float)
    elev_min, elev_max = elev.min(), elev.max()
    elev_norm = (elev - elev_min) / (elev_max - elev_min)  # 0 to 1
    # Peak suitability at mid elevation (~0.45 normalised)
    elev_factor = 1.0 - np.abs(elev_norm - 0.45) * 1.5
    elev_factor = np.clip(elev_factor, 0.1, 1.0)

    # ── Step 2: Slope factor ──────────────────────────────────────────────────
    # Flatter terrain = easier spread = higher host density score
    slope = df["Slope"].values.astype(float)
    slope_norm = (slope - slope.min()) / (slope.max() - slope.min())
    slope_factor = 1.0 - slope_norm * 0.6   # flat=1.0, steep=0.4

    # ── Step 3: Wilderness area factor ───────────────────────────────────────
    # More remote wilderness = more undisturbed forest = higher host density
    wilderness_cols = ["Wilderness_Area1", "Wilderness_Area2",
                       "Wilderness_Area3", "Wilderness_Area4"]
    # Areas 1 and 3 (Rawah, Comanche Peak) are larger/denser
    wilderness_weights = np.array([0.9, 0.6, 0.85, 0.5])
    wilderness_matrix = df[wilderness_cols].values  # shape (rows, 4)
    wilderness_factor = wilderness_matrix @ wilderness_weights  # dot product

    # ── Step 4: Cover type susceptibility (train.csv only) ───────────────────
    if "Cover_Type" in df.columns:
        cover_factor = df["Cover_Type"].map(SUSCEPTIBILITY).values.astype(float)
    else:
        # test.csv has no Cover_Type — use soil types as a proxy
        # Soil types 1-6 tend to correlate with more susceptible tree types
        soil_cols = [f"Soil_Type{i}" for i in range(1, 7)]
        available = [c for c in soil_cols if c in df.columns]
        cover_factor = df[available].sum(axis=1).clip(0, 1).values.astype(float)
        cover_factor = 0.4 + cover_factor * 0.5   # shift to [0.4, 0.9]

    # ── Step 5: Combine all factors ───────────────────────────────────────────
    host_density_raw = (
        0.35 * elev_factor +
        0.20 * slope_factor +
        0.20 * wilderness_factor +
        0.25 * cover_factor
    )

    # Normalise to [0.1, 1.0]
    hd_min = host_density_raw.min()
    hd_max = host_density_raw.max()
    host_density_norm = (host_density_raw - hd_min) / (hd_max - hd_min)
    host_density_norm = 0.1 + host_density_norm * 0.9   # scale to [0.1, 1.0]

    # ── Step 6: Sample n*n rows and reshape into grid ─────────────────────────
    rng = np.random.default_rng(seed)
    n_cells = n * n

    if len(host_density_norm) >= n_cells:
        # Sample without replacement
        indices = rng.choice(len(host_density_norm), size=n_cells, replace=False)
    else:
        # Dataset smaller than grid — sample with replacement
        indices = rng.choice(len(host_density_norm), size=n_cells, replace=True)

    grid = host_density_norm[indices].reshape(n, n)

    print(f"  Grid shape: {grid.shape}")
    print(f"  Host density range: [{grid.min():.3f}, {grid.max():.3f}]")
    print(f"  Host density mean:  {grid.mean():.3f}")

    return grid


def load_and_preview(filepath: str, n: int = 40) -> np.ndarray:
    """Load and print a summary — useful for quick checks."""
    grid = load_host_density(filepath, n=n)
    print(f"\n  Sample grid corner (top-left 5x5):")
    print(np.round(grid[:5, :5], 2))
    return grid
