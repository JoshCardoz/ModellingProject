"""
grid.py
-------
Defines the GridEnvironment class, which represents the NxN forest landscape.

Each cell stores:
  - host_density  : float [0, 1]  — how many host trees are present
  - temperature   : float         — average temperature for this cell (°C)
  - status        : int           — 0 = Susceptible, 1 = Infested, 2 = Depleted
"""

import numpy as np


# ── Cell status constants ──────────────────────────────────────────────────────
SUSCEPTIBLE = 0
INFESTED    = 1
DEPLETED    = 2


class GridEnvironment:
    """
    Represents the NxN forest grid.

    Parameters
    ----------
    n              : int   — grid side length (n x n cells)
    host_density   : ndarray or None
        If None, host density is randomly generated from a uniform [0.3, 1.0]
        distribution.  Pass your own (n x n) numpy array to use real data.
    temp_mean      : float — mean cell temperature (°C)
    temp_std       : float — std-dev for random temperature variation across cells
    seed           : int or None — RNG seed for reproducibility
    """

    def __init__(
        self,
        n: int = 50,
        host_density: np.ndarray | None = None,
        temp_mean: float = 20.0,
        temp_std: float = 3.0,
        seed: int | None = None,
    ):
        self.n    = n
        self.rng  = np.random.default_rng(seed)

        # ── Host density ──────────────────────────────────────────────────────
        if host_density is not None:
            assert host_density.shape == (n, n), "host_density must be (n x n)"
            self.host_density = host_density.astype(float)
        else:
            self.host_density = self.rng.uniform(0.3, 1.0, size=(n, n))

        # ── Temperature (static per cell for now) ─────────────────────────────
        self.temperature = self.rng.normal(temp_mean, temp_std, size=(n, n))

        # ── Infestation status grid ───────────────────────────────────────────
        self.status = np.full((n, n), SUSCEPTIBLE, dtype=int)

    # ── Convenience helpers ───────────────────────────────────────────────────

    def set_ignition(self, row: int, col: int) -> None:
        """Mark a single cell as the starting infestation point."""
        self.status[row, col] = INFESTED

    def set_ignition_random(self) -> tuple[int, int]:
        """Pick a random cell and infest it. Returns the (row, col)."""
        r = int(self.rng.integers(0, self.n))
        c = int(self.rng.integers(0, self.n))
        self.set_ignition(r, c)
        return r, c

    def reset_status(self) -> None:
        """Clear all infestation status back to Susceptible."""
        self.status[:] = SUSCEPTIBLE

    def infested_count(self) -> int:
        return int(np.sum(self.status == INFESTED))

    def depleted_count(self) -> int:
        return int(np.sum(self.status == DEPLETED))

    def susceptible_count(self) -> int:
        return int(np.sum(self.status == SUSCEPTIBLE))

    def __repr__(self) -> str:
        return (
            f"GridEnvironment(n={self.n}, "
            f"infested={self.infested_count()}, "
            f"depleted={self.depleted_count()})"
        )