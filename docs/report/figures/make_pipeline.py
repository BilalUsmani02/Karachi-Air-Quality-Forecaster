"""Draw the end-to-end pipeline diagram for the technical report.

Reproducible, matplotlib-only (no graphviz): boxes + arrows, ~16:9, 200 dpi,
white background. Run with:

    uv run python docs/report/figures/make_pipeline.py

Writes both docs/report/figures/pipeline.svg and docs/report/figures/pipeline.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

FIGURES_DIR = Path(__file__).resolve().parent

# Colours per course emphasis (kept light so black text stays readable).
PAI = "#dbe9f6"      # PAI: from-scratch network + comparison
AIES = "#fde3cf"     # AIES CCP/Lab: classification + expert system
SHARED = "#e8e8e8"   # shared by all three courses
EDGE = "#333333"

# Each stage: (label, x, y, width, height, fill colour)
STAGES = [
    ("Data sources\nD1 OpenAQ sensors\nD2 ERA5 weather\nD3 CAMS air quality\nD4 EPA/WHO guidance\nD5 Consulate (context)", 0.5, 5.0, 2.3, 2.0, SHARED),
    ("Cleaning\nsrc/data/clean.py\nflag + fix faults,\nfill short gaps", 3.2, 5.0, 2.1, 2.0, SHARED),
    ("Daily features\nsrc/data/features.py\nlags, rolling stats,\ncalendar, weather", 5.7, 5.0, 2.1, 2.0, SHARED),
    ("Splits\nsrc/data/split.py\nrolling-origin CV\n+ final test", 8.2, 5.0, 2.0, 2.0, SHARED),
    ("Models\npersistence, Ridge, RF,\nHistGB, NumPy MLP\n+ PyTorch twin, direct clf.", 10.6, 5.0, 2.3, 2.0, PAI),
    ("Tuning + ensemble\ntune_classical, tune_mlp\nHistGB + NumPy MLP avg", 10.6, 2.2, 2.3, 2.0, PAI),
    ("Band classification\nsrc/classify/bands.py\nEPA 24-h breakpoints", 8.2, 2.2, 2.0, 2.0, AIES),
    ("Expert system\nsrc/expert/\nrules R01-R22,\nforward chaining", 5.7, 2.2, 2.1, 2.0, AIES),
    ("SHAP reasons\nsrc/explain/\ntop factors per\nforecast", 3.2, 2.2, 2.1, 2.0, AIES),
    ("Streamlit app\napp/\nforecast, bands, advice,\nreasons, comparison", 0.5, 2.2, 2.3, 2.0, SHARED),
]

# Arrows: index pairs into STAGES, in pipeline order (plus the live-data loop).
ARROWS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
]


def draw_box(ax, label, x, y, w, h, fc):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.12",
        linewidth=1.4, edgecolor=EDGE, facecolor=fc, zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center", fontsize=9.3, color="#111111",
        linespacing=1.35, zorder=3,
    )


def draw_arrow(ax, a, b):
    xa, ya, wa, ha = a[1], a[2], a[3], a[4]
    xb, yb, wb, hb = b[1], b[2], b[3], b[4]
    # Decide start/end points on box edges, allowing for either direction.
    if abs(ya - yb) < 0.01:  # same row -> left/right edge, whichever faces the other box
        if xb > xa:
            start = (xa + wa, ya + ha / 2)
            end = (xb, yb + hb / 2)
        else:
            start = (xa, ya + ha / 2)
            end = (xb + wb, yb + hb / 2)
    else:  # different row -> top/bottom edge, whichever faces the other box
        if yb < ya:
            start = (xa + wa / 2, ya)
            end = (xb + wb / 2, yb + hb)
        else:
            start = (xa + wa / 2, ya + ha)
            end = (xb + wb / 2, yb)
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=16,
        linewidth=1.6, color=EDGE, zorder=5, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arrow)


def make_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=200)  # 16:9
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for stage in STAGES:
        draw_box(ax, *stage)

    for i, j in ARROWS:
        draw_arrow(ax, STAGES[i], STAGES[j])

    # Live/cache/demo loop feeding back into the app (dashed, labelled).
    live_box = STAGES[9]
    ax.annotate(
        "",
        xy=(live_box[1] + live_box[3] / 2, live_box[2] + live_box[4] + 0.05),
        xytext=(STAGES[0][1] + STAGES[0][3] / 2, STAGES[0][2] - 0.35),
        arrowprops=dict(arrowstyle="-|>", linestyle="--", linewidth=1.3, color="#777777"),
    )
    ax.text(
        1.6, 1.75,
        "live / cache / demo snapshot (src/live/) feeds the app each run",
        fontsize=8.3, color="#555555", ha="center", style="italic",
    )

    ax.text(
        6.9, 7.72,
        "Karachi Air-Quality Forecaster — end-to-end pipeline",
        fontsize=15, fontweight="bold", ha="center",
    )
    ax.text(
        6.9, 7.3,
        "Boxes flow left to right, top row then bottom row; the dashed arrow is the live-refresh loop.",
        fontsize=9.3, ha="center", color="#444444",
    )

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=12, markerfacecolor=PAI, markeredgecolor=EDGE, label="PAI emphasis (from-scratch network + comparison)"),
        Line2D([0], [0], marker="s", linestyle="", markersize=12, markerfacecolor=AIES, markeredgecolor=EDGE, label="AIES CCP/Lab emphasis (classification + expert system)"),
        Line2D([0], [0], marker="s", linestyle="", markersize=12, markerfacecolor=SHARED, markeredgecolor=EDGE, label="Shared by all three courses"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=3, frameon=False, fontsize=8.6,
    )

    ax.set_xlim(0, 13.33)
    ax.set_ylim(0, 8.1)
    ax.axis("off")
    fig.tight_layout(pad=0.6)
    return fig


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    svg_path = FIGURES_DIR / "pipeline.svg"
    png_path = FIGURES_DIR / "pipeline.png"
    fig.savefig(svg_path, format="svg", facecolor="white")
    fig.savefig(png_path, format="png", facecolor="white", dpi=200)
    plt.close(fig)
    print(f"wrote {svg_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
