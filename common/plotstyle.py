"""One plotting style for every report figure: a validated categorical palette in a fixed order,
thin marks, recessive grid, text in ink colours (never the series colour), vector PDF output."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
NEUTRAL = "#8a8984"

# Task 1: colour follows the method everywhere
T1_COLORS = {"baseline": NEUTRAL, "full_mid": PALETTE[0], "full": PALETTE[0], "l2sp": PALETTE[1],
             "lora4": PALETTE[2], "lora16": PALETTE[3], "oplora": PALETTE[4]}


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.4, "legend.frameon": False,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
        "axes.titlelocation": "left", "axes.titleweight": "bold",
    })


def save(fig, path_stem):
    from pathlib import Path

    p = Path(path_stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".pdf"))
    fig.savefig(p.with_suffix(".png"), dpi=200)
    plt.close(fig)
