"""
app.py
------
Streamlit web app for the invasive species spread simulation.

Run with:
  streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from monte_carlo import MonteCarloSimulation
from data_loader import load_host_density

st.set_page_config(page_title="Invasive Species Spread Simulator", page_icon="🌲", layout="wide")
st.title("Invasive Species Spread Simulator")
st.caption("Forest Pest Spread — Monte Carlo Simulation | SOFE 4820U")

st.sidebar.header("Simulation parameters")
n = st.sidebar.slider("Grid size (N x N)", 10, 80, 40, 5)
m = st.sidebar.slider("Monte Carlo trials", 10, 100, 50, 10)
t_max = st.sidebar.slider("Time steps", 10, 100, 60, 5)
st.sidebar.divider()
base_spread_prob = st.sidebar.slider("Base spread probability", 0.1, 1.0, 0.35, 0.05)
rho_min = st.sidebar.slider("Min host density (ρ_min)", 0.05, 0.5, 0.1, 0.05)
depletion_rate = st.sidebar.slider("Depletion rate", 0.01, 0.3, 0.1, 0.01)
st.sidebar.divider()
wind_max = st.sidebar.slider("Max wind speed", 0.0, 10.0, 5.0, 0.5)
temp_mean = st.sidebar.slider("Mean temperature (°C)", 5.0, 35.0, 20.0, 1.0)
temp_std = st.sidebar.slider("Temperature variation (std)", 0.5, 8.0, 3.0, 0.5)
st.sidebar.divider()
use_real_data = st.sidebar.checkbox("Use real Kaggle forest data", value=True)
st.sidebar.divider()
run = st.sidebar.button("Run simulation", type="primary", use_container_width=True)

STATUS_CMAP = mcolors.ListedColormap(["#5DCAA5", "#E24B4A", "#2C2C2A"])
STATUS_NORM = mcolors.BoundaryNorm([0, 1, 2, 3], 3)
LEGEND = [Patch(facecolor="#5DCAA5", label="Susceptible"),
          Patch(facecolor="#E24B4A", label="Infested"),
          Patch(facecolor="#2C2C2A", label="Depleted")]

def make_heatmap(snapshot):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(snapshot, cmap=STATUS_CMAP, norm=STATUS_NORM, interpolation="nearest")
    ax.legend(handles=LEGEND, loc="upper right", fontsize=8)
    ax.set_title("Infestation heatmap (final state)", fontsize=11)
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    return fig

def make_progression(trial):
    history = trial["history"]
    idxs = [0, len(history)//3, 2*len(history)//3, len(history)-1]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, idx in zip(axes, idxs):
        ax.imshow(history[idx]["snapshot"], cmap=STATUS_CMAP, norm=STATUS_NORM, interpolation="nearest")
        ax.set_title(f"Step {history[idx]['step']}", fontsize=9)
        ax.axis("off")
    fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Spread progression — trial 1", fontsize=11)
    plt.tight_layout()
    return fig

def make_spread_chart(results):
    mean, std = results["mean_infested"], results["std_infested"]
    steps = np.arange(len(mean))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for trial in results["trials"]:
        ax.plot(trial["infested_counts"], color="#5DCAA5", alpha=0.1, linewidth=0.8)
    ax.fill_between(steps, mean-std, mean+std, color="#1D9E75", alpha=0.2, label="±1 std dev")
    ax.plot(steps, mean, color="#1D9E75", linewidth=2, label="Mean infested")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Infested cells")
    ax.set_title("Spread over time", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return fig

def make_histogram(velocities):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(velocities, bins=15, color="#5DCAA5", edgecolor="#0F6E56", linewidth=0.5)
    ax.axvline(np.mean(velocities), color="#E24B4A", linewidth=1.5,
               linestyle="--", label=f"Mean: {np.mean(velocities):.1f}")
    ax.set_xlabel("Cells / step")
    ax.set_ylabel("Trials")
    ax.set_title("Spread velocity distribution", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return fig

def make_sensitivity_chart(sensitivity):
    labels = {"wind_speed": "Wind speed", "host_density_mean": "Host density",
              "base_spread_prob": "Spread probability"}
    names = [labels.get(k, k) for k in sensitivity]
    scores = list(sensitivity.values())
    colours = ["#1D9E75" if s == max(scores) else "#5DCAA5" for s in scores]
    fig, ax = plt.subplots(figsize=(5, 2.5))
    bars = ax.barh(names, scores, color=colours, edgecolor="#0F6E56", linewidth=0.5)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=8)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Sensitivity score")
    ax.set_title("Variable sensitivity ranking", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    return fig

if not run:
    st.info("Configure the parameters in the sidebar and click **Run simulation** to start.")
    st.markdown("""
    ### How it works
    - The forest is represented as an **N×N grid** where each cell has a host tree density and temperature
    - A pest starts at a random cell and spreads to neighbours based on a probability influenced by **host density, temperature, and wind**
    - **Monte Carlo** runs the simulation M times with randomised wind conditions to produce a distribution of outcomes
    - The **sensitivity ranking** shows which variable most influences how fast the pest spreads

    ### Dataset
    Host density is derived from the **Kaggle Forest Cover Type dataset** — 565,892 real forest patches
    from Roosevelt National Forest, Colorado. Elevation, slope, soil type, and wilderness area are used
    as proxies for tree susceptibility to invasive pests.

    ### Validation
    Spread velocity is compared against the published EAB spread rate of **40 km/year**
    (Orlova-Bienkowskaja 2018). Our simulation produces ~41.6 km/year under default parameters.
    """)
else:
    host_density = None
    if use_real_data and os.path.exists("test.csv"):
        with st.spinner("Loading real forest data..."):
            host_density = load_host_density("test.csv", n=n)
        st.success(f"Loaded real forest data — {n*n} cells sampled from 565,892 forest patches.")
    elif use_real_data:
        st.warning("test.csv not found — using randomly generated host density.")

    with st.spinner(f"Running {m} Monte Carlo trials on a {n}×{n} grid..."):
        sim = MonteCarloSimulation(
            n=n, m=m, t_max=t_max,
            base_spread_prob=base_spread_prob,
            rho_min=rho_min,
            depletion_rate=depletion_rate,
            wind_speed_range=(0.0, wind_max),
            temp_mean=temp_mean,
            temp_std=temp_std,
            seed=None,
        )
        if host_density is not None:
            sim.base_host_density = host_density
        results = sim.run(verbose=False)

    total_cells   = n * n
    mean_final    = int(results["mean_infested"][-1])
    mean_depleted = int(np.mean([t["final_depleted"] for t in results["trials"]]))
    pct_affected  = round((mean_final + mean_depleted) / total_cells * 100, 1)
    mean_velocity = round(float(results["velocities"].mean()), 2)
    km_per_year   = round(mean_velocity * 26, 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean cells infested", mean_final)
    c2.metric("Mean cells depleted", mean_depleted)
    c3.metric("Forest affected", f"{pct_affected}%")
    c4.metric("Spread velocity", f"{mean_velocity} cells/step")
    c5.metric("Est. km/year", f"{km_per_year} km/yr")

    st.divider()
    st.subheader("Spread progression")
    st.pyplot(make_progression(results["trials"][0]))
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Infestation heatmap")
        st.pyplot(make_heatmap(results["trials"][0]["final_snapshot"]))
    with col2:
        st.subheader("Spread over time")
        st.pyplot(make_spread_chart(results))

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Spread velocity distribution")
        st.pyplot(make_histogram(results["velocities"]))
    with col4:
        st.subheader("Variable sensitivity")
        st.pyplot(make_sensitivity_chart(results["sensitivity"]))

    st.divider()
    mean = results["mean_infested"]
    std  = results["std_infested"]
    all_counts = np.array([t["infested_counts"] for t in results["trials"]])
    csv_lines = ["step,mean_infested,std_infested,min_infested,max_infested"]
    for i in range(len(mean)):
        csv_lines.append(f"{i},{mean[i]:.2f},{std[i]:.2f},{int(all_counts[:,i].min())},{int(all_counts[:,i].max())}")
    st.download_button(
        label="Download results as CSV",
        data="\n".join(csv_lines),
        file_name="spread_data.csv",
        mime="text/csv",
    )
