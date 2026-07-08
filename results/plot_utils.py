"""
results/plot_utils.py

Shared styling + save helpers for the two chart-generation scripts
(generate_rl_charts.py, generate_cnn_charts.py). Nothing in rl/, cnn/, or
integration/ is imported or modified here -- this module only touches
matplotlib/seaborn presentation and the results/ output folder.

Output layout created under the project root when the chart scripts run:

    results/
    ├── figures/   <- one .png per chart (publication-quality, 300 dpi)
    ├── tables/    <- one .csv per chart with the exact numbers plotted,
    │                 so figures are reproducible/auditable without re-running
    │                 the (stochastic) training loop
    ├── rl_checkpoints/    <- written by generate_rl_charts.py's training run
    └── cnn_checkpoints/   <- written by generate_cnn_charts.py's training run
"""

import os
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    HAS_SEABORN = True
except ImportError:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    HAS_SEABORN = False

FIGURES_DIR = os.path.join("results", "figures")
TABLES_DIR = os.path.join("results", "tables")

# A single, consistent color per concept, reused across every chart in both
# scripts so a reader can tell "DQN" / "conventional" / condition-severity
# apart at a glance without re-reading each legend.
COLORS = {
    "dqn": "#1f77b4",
    "conventional": "#d62728",
    "train": "#1f77b4",
    "val": "#ff7f0e",
    "test": "#2ca02c",
    "healthy": "#2ca02c",
    "moderate_stress": "#ff9f1c",
    "severe_stress": "#d62728",
    "action_0": "#9ecae1",
    "action_1": "#4292c6",
    "action_2": "#08519c",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})


def ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)


def savefig(fig, name):
    """Saves a figure as results/figures/<name>.png and closes it (so a
    script that generates a dozen charts in one run doesn't hold a dozen
    figures open in memory)."""
    ensure_dirs()
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def save_table(df, name):
    """Saves the exact data behind a chart as results/tables/<name>.csv."""
    ensure_dirs()
    path = os.path.join(TABLES_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  wrote {path}")
    return path
