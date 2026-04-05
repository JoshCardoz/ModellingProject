"""
spread_engine.py
----------------
Handles the stochastic spread logic for one simulation trial.

At each time step:
  1. Every INFESTED cell attempts to spread to each of its 4 neighbours.
  2. Spread probability P_spread is computed from host density, temperature,
     and the wind bonus for that neighbour direction.
  3. Infested cells deplete their host trees over time; once host density
     drops below rho_min the cell transitions to DEPLETED.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from grid import GridEnvironment, SUSCEPTIBLE, INFESTED, DEPLETED


# ── Direction helpers ──────────────────────────────────────────────────────────
# Cardinal directions as (delta_row, delta_col) and readable names
DIRECTIONS = {
    "N": (-1,  0),
    "S": ( 1,  0),
    "E": ( 0,  1),
    "W": ( 0, -1),
}


def _wind_bonus(wind_vector: tuple[float, float], direction: tuple[int, int]) -> float:
    """
    Returns an extra spread probability [0, max_bonus] when the wind is blowing
    toward `direction`.

    wind_vector : (dx, dy) where dx = East component, dy = North component
    direction   : (dr, dc) — the neighbour offset we are testing
    """
    wind_speed = np.sqrt(wind_vector[0] ** 2 + wind_vector[1] ** 2)
    if wind_speed == 0:
        return 0.0

    # Convert neighbour offset to a unit vector in (E, N) space
    # dr is row offset (positive = South), dc is col offset (positive = East)
    dr, dc = direction
    # Neighbour direction in (East, North) coordinates
    neigh_east  =  dc          # East is positive column
    neigh_north = -dr          # North is negative row (row 0 is top)

    # Dot product of normalised wind and neighbour direction
    wind_unit  = np.array(wind_vector) / wind_speed
    neigh_unit = np.array([neigh_east, neigh_north])
    neigh_mag  = np.sqrt(neigh_east**2 + neigh_north**2)
    if neigh_mag == 0:
        return 0.0
    neigh_unit = neigh_unit / neigh_mag

    alignment = float(np.dot(wind_unit, neigh_unit))  # [-1, 1]
    alignment = max(0.0, alignment)                    # only positive (downwind)

    # Scale: max bonus = 0.15 at full wind alignment and high wind speed
    max_bonus = 0.15
    speed_factor = min(wind_speed / 10.0, 1.0)        # normalise speed to [0,1]
    return max_bonus * alignment * speed_factor


class SpreadEngine:
    """
    Runs a single Monte Carlo trial on a GridEnvironment.

    Parameters
    ----------
    grid            : GridEnvironment
    t_min, t_max    : float  — viable temperature range for the pest (°C)
    rho_min         : float  — minimum host density to sustain infestation
    base_spread_prob: float  — base spread probability under ideal conditions
    depletion_rate  : float  — how fast a cell's host density decreases per step
    wind_vector     : (dx, dy) — constant wind for this trial (East, North)
    seed            : int or None
    """

    def __init__(
        self,
        grid: GridEnvironment,
        t_min: float = 10.0,
        t_max: float = 30.0,
        rho_min: float = 0.2,
        base_spread_prob: float = 0.4,
        depletion_rate: float = 0.1,
        wind_vector: tuple[float, float] = (0.0, 0.0),
        seed: int | None = None,
    ):
        self.grid             = grid
        self.t_min            = t_min
        self.t_max            = t_max
        self.rho_min          = rho_min
        self.base_spread_prob = base_spread_prob
        self.depletion_rate   = depletion_rate
        self.wind_vector      = wind_vector
        self.rng              = np.random.default_rng(seed)

    # ── Core probability formula ───────────────────────────────────────────────

    def _spread_prob(self, row: int, col: int, direction: tuple[int, int]) -> float:
        """
        Compute the probability that an infested cell at (row, col) spreads
        to the neighbour in `direction`.

        Formula:
          P = base * temp_factor * host_factor + wind_bonus
          clamped to [0, 1]
        """
        nr, nc = row + direction[0], col + direction[1]

        # ── Temperature suitability for the TARGET cell ───────────────────────
        temp = self.grid.temperature[nr, nc]
        if temp < self.t_min or temp > self.t_max:
            return 0.0  # outside viable range — no spread

        # Linear scale: probability peaks at mid-range temperature
        t_mid   = (self.t_min + self.t_max) / 2
        t_range = (self.t_max - self.t_min) / 2
        temp_factor = 1.0 - abs(temp - t_mid) / t_range   # [0, 1]

        # ── Host density suitability ──────────────────────────────────────────
        host = self.grid.host_density[nr, nc]
        if host < self.rho_min:
            return 0.0  # not enough trees to sustain infestation

        host_factor = (host - self.rho_min) / (1.0 - self.rho_min)  # [0, 1]

        # ── Wind bonus ────────────────────────────────────────────────────────
        bonus = _wind_bonus(self.wind_vector, direction)

        p = self.base_spread_prob * temp_factor * host_factor + bonus
        return float(np.clip(p, 0.0, 1.0))

    # ── Single time-step ───────────────────────────────────────────────────────

    def step(self) -> None:
        """
        Advance the simulation by one time step.
        Uses a copy of the current status so spread within a step is
        based only on the state at the START of that step.
        """
        n      = self.grid.n
        old_status = self.grid.status.copy()
        new_status = self.grid.status.copy()

        for r in range(n):
            for c in range(n):
                if old_status[r, c] != INFESTED:
                    continue

                # ── Try to spread to each neighbour ───────────────────────────
                for name, (dr, dc) in DIRECTIONS.items():
                    nr, nc = r + dr, c + dc
                    # Boundary check
                    if not (0 <= nr < n and 0 <= nc < n):
                        continue
                    # Only spread to Susceptible cells
                    if old_status[nr, nc] != SUSCEPTIBLE:
                        continue
                    # Stochastic spread
                    p = self._spread_prob(r, c, (dr, dc))
                    if self.rng.random() < p:
                        new_status[nr, nc] = INFESTED

                # ── Deplete host trees in this cell ───────────────────────────
                self.grid.host_density[r, c] = max(
                    0.0,
                    self.grid.host_density[r, c] - self.depletion_rate,
                )
                if self.grid.host_density[r, c] < self.rho_min:
                    new_status[r, c] = DEPLETED

        self.grid.status = new_status

    # ── Run a full trial ───────────────────────────────────────────────────────

    def run(self, t_max: int, ignition: tuple[int, int] | None = None) -> list[dict]:
        """
        Run the simulation for t_max steps.

        Parameters
        ----------
        t_max    : int            — number of time steps
        ignition : (row, col) or None
            Specific ignition cell. If None, a random cell is chosen.

        Returns
        -------
        history : list of dicts with keys
            - 'step'        : time step index
            - 'infested'    : count of INFESTED cells
            - 'depleted'    : count of DEPLETED cells
            - 'susceptible' : count of SUSCEPTIBLE cells
            - 'snapshot'    : (n x n) int array — full status grid
        """
        # Set ignition
        if ignition is not None:
            self.grid.set_ignition(*ignition)
        else:
            ignition = self.grid.set_ignition_random()

        history = []

        for t in range(t_max):
            snapshot = self.grid.status.copy()
            history.append({
                "step":        t,
                "infested":    self.grid.infested_count(),
                "depleted":    self.grid.depleted_count(),
                "susceptible": self.grid.susceptible_count(),
                "snapshot":    snapshot,
            })
            self.step()

        # Record final state
        history.append({
            "step":        t_max,
            "infested":    self.grid.infested_count(),
            "depleted":    self.grid.depleted_count(),
            "susceptible": self.grid.susceptible_count(),
            "snapshot":    self.grid.status.copy(),
        })

        return history
