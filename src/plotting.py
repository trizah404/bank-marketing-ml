"""Shared chart styling so every figure looks consistent."""
import matplotlib
matplotlib.use("Agg")  # render to file, no screen needed
import matplotlib.pyplot as plt

# Fixed categorical order: series 1 is always blue, 2 orange, 3 aqua, ...
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "font.size": 10, "legend.frameon": False, "figure.dpi": 150,
    })


def rate_bars(ax, labels, rates, title, xlabel, baseline=None):
    """Bar chart of a rate per group, with value labels and optional baseline."""
    ax.bar(labels, rates, color=SERIES[0], width=0.6, zorder=2)
    for i, r in enumerate(rates):
        ax.text(i, r, f"{r:.0%}", ha="center", va="bottom", color=INK2, fontsize=9)
    if baseline is not None:
        ax.axhline(baseline, color=MUTED, ls="--", lw=1, zorder=1)
        ax.text(len(labels) - 0.5, baseline, f"overall {baseline:.0%}",
                ha="right", va="bottom", color=MUTED, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("subscription rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="x", visible=False)
