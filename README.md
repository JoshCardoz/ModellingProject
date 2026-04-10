# Invasive Species Spread Simulator
Monte Carlo simulation of forest pest spread — SOFE 4820U Modelling & Simulation, Winter 2026

## Overview
This project simulates the spread of an invasive forest pest (modelled after the Emerald Ash Borer)
across an N×N forest grid using Monte Carlo simulation. Host density is derived from the
[Kaggle Forest Cover Type dataset](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset)
(Roosevelt National Forest, Colorado). Spread probability at each step is influenced by host tree
density, cell temperature, and wind speed/direction. Results are validated against the published
EAB spread rate of ~40 km/year (Orlova-Bienkowskaja 2018).

## Project Structure
```
.
├── app.py              # Streamlit web app (main entry point for the UI)
├── run_simulation.py   # CLI entry point — runs the full pipeline and saves outputs
├── monte_carlo.py      # Monte Carlo runner (M independent trials, aggregation)
├── spread_engine.py    # Stochastic spread logic for a single trial
├── grid.py             # NxN forest grid environment
├── data_loader.py      # Loads and converts the Kaggle CSV into a host density grid
├── validation.py       # Compares simulation output against real-world EAB benchmark
├── visualize.py        # Chart and CSV generation
├── requirements.txt    # Python dependencies
├── train.csv           # Kaggle Forest Cover Type training data (place here)
└── test.csv            # Kaggle Forest Cover Type test data (place here)
```

## Requirements
- Python 3.10 or higher
- Dependencies listed in `requirements.txt`

## Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset (optional but recommended)
Download `train.csv` and/or `test.csv` from the
[Kaggle Forest Cover Type dataset](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset)
and place them in the project root. If the files are not present, the simulation falls back to
randomly generated host density.

## Running the App

### Streamlit web app (recommended)
```bash
streamlit run app.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

Use the sidebar to adjust parameters (grid size, number of trials, wind speed, temperature, etc.)
and click **Run simulation**.

### Command-line pipeline
Runs the full simulation, saves all charts to `outputs/`, and prints a validation summary:
```bash
python run_simulation.py
```

Output files saved to `outputs/`:
| File | Description |
|---|---|
| `spread_over_time.png` | Mean infested cells per step across all trials |
| `heatmap_trial1_step60.png` | Final infestation heatmap for trial 1 |
| `progression_trial1.png` | Spread progression snapshots for trial 1 |
| `velocity_histogram.png` | Distribution of spread velocity across trials |
| `sensitivity.png` | Variable sensitivity ranking |
| `spread_data.csv` | Mean/std/min/max infested cells per step |
| `validation_comparison.png` | Simulation vs EAB benchmark chart |
| `benchmark_data.csv` | Benchmark and simulation values per step |

## How It Works
1. **Grid** — The forest is an N×N grid. Each cell has a host tree density (from Kaggle data or random) and a temperature drawn from a normal distribution.
2. **Ignition** — A random cell is marked as the initial infestation point.
3. **Spread** — At each time step, every infested cell attempts to spread to its 4 neighbours. The probability is: `P = base_prob × temp_factor × host_factor + wind_bonus`, clamped to [0, 1].
4. **Depletion** — Infested cells lose host density each step. Once below `rho_min`, the cell transitions to Depleted (no further spread).
5. **Monte Carlo** — Steps 1–4 are repeated M times with re-randomised wind conditions. Results are averaged to produce mean/std spread curves.
6. **Validation** — The mean spread curve is compared against a synthetic EAB benchmark using RMSE and Pearson correlation.
