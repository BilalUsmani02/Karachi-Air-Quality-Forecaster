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

# Colours by role (kept light so black text stays readable).
MODEL = "#dbe9f6"    # models, tuning and the ensemble
FOCUS = "#fde3cf"    # band classification, expert system and explanations
SHARED = "#e8e8e8"   # data handling and the interface
EDGE = "#333333"

# Each stage: (label, x, y, width, height, fill colour)
STAGES = [
    ("Data sources\nD1 OpenAQ sensors\nD2 ERA5 weather\nD3 CAMS air quality\nD4 EPA/WHO guidance\nD5 Consulate (context)", 0.5, 5.0, 2.3, 2.0, SHARED),
    ("Cleaning\nsrc/data/clean.py\nflag + fix faults,\nfill short gaps", 3.2, 5.0, 2.1, 2.0, SHARED),
    ("Daily features\nsrc/data/features.py\nlags, rolling stats,\ncalendar, weather", 5.7, 5.0, 2.1, 2.0, SHARED),
    ("Splits\nsrc/data/split.py\nrolling-origin CV\n+ final test", 8.2, 5.0, 2.0, 2.0, SHARED),
    ("Models\npersistence, Ridge, RF,\nHistGB, NumPy MLP\n+ PyTorch twin, direct clf.", 10.6, 5.0, 2.3, 2.0, MODEL),
    ("Tuning + ensemble\ntune_classical, tune_mlp\nHistGB + NumPy MLP avg", 10.6, 1.35, 2.3, 2.0, MODEL),
    ("Forecast + band\nPM2.5 forecast, then\nEPA 24-h breakpoints\nsrc/classify/bands.py", 7.6, 1.35, 2.4, 2.0, FOCUS),
    ("Expert system\nsrc/expert/\nrules R01-R22, forward chaining", 3.9, 2.65, 3.0, 1.45, FOCUS),
    ("SHAP explanations\nsrc/explain/\ntop factors per forecast", 3.9, 0.6, 3.0, 1.45, FOCUS),
    ("Streamlit app\napp/\nforecast, band, advice,\nreasons, comparison", 0.5, 1.35, 2.5, 2.0, SHARED),
]

# Arrows: index pairs into STAGES. The forecast feeds the expert system and the SHAP
# explainer in parallel (6 -> 7 and 6 -> 8); both feed the app.
ARROWS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (6, 8), (7, 9), (8, 9),
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
    if xa + wa < xb or xb + wb < xa:  # side by side (same row or fanned out) -> facing side edges
        if xb > xa:
            start = (xa + wa + 0.06, ya + ha / 2)
            end = (xb - 0.06, yb + hb / 2)
        else:
            start = (xa - 0.06, ya + ha / 2)
            end = (xb + wb + 0.06, yb + hb / 2)
    else:  # stacked -> top/bottom edge, whichever faces the other box
        if yb < ya:
            start = (xa + wa / 2, ya - 0.06)
            end = (xb + wb / 2, yb + hb + 0.06)
        else:
            start = (xa + wa / 2, ya + ha + 0.06)
            end = (xb + wb / 2, yb - 0.06)
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

    # Live/cache/demo loop feeding the app (dashed, labelled).
    data_box, app_box = STAGES[0], STAGES[9]
    x_loop = data_box[1] + data_box[3] / 2
    ax.annotate(
        "",
        xy=(x_loop, app_box[2] + app_box[4] + 0.1),
        xytext=(x_loop, data_box[2] - 0.1),
        arrowprops=dict(arrowstyle="-|>", linestyle="--", linewidth=1.3, color="#777777"),
    )
    ax.text(
        x_loop + 0.15, 4.2,
        "live / cache / demo\nsnapshot (src/live/)\nfeeds the app each run",
        fontsize=8.3, color="#555555", ha="left", va="center", style="italic",
    )

    ax.text(
        6.9, 7.72,
        "Karachi Air-Quality Forecaster — end-to-end pipeline",
        fontsize=15, fontweight="bold", ha="center",
    )
    ax.text(
        6.9, 7.3,
        "Top row left to right, then bottom row right to left. "
        "The expert system and SHAP both use the forecast, in parallel.",
        fontsize=9.3, ha="center", color="#444444",
    )

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=12, markerfacecolor=MODEL, markeredgecolor=EDGE, label="Models, tuning and ensemble"),
        Line2D([0], [0], marker="s", linestyle="", markersize=12, markerfacecolor=FOCUS, markeredgecolor=EDGE, label="Band classification, expert system and explanations"),
        Line2D([0], [0], marker="s", linestyle="", markersize=12, markerfacecolor=SHARED, markeredgecolor=EDGE, label="Data handling and interface"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=3, frameon=False, fontsize=8.6,
    )

    ax.set_xlim(0, 13.33)
    ax.set_ylim(-0.3, 8.1)
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
