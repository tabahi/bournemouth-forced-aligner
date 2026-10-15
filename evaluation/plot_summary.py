"""
The two comparison figures in docs/evaluation.md, one panel per dataset:

    summary_f1.png        boundary F1 at each tolerance, per method (from the results CSVs)
    summary_distance.png  share of hand-labelled boundaries with a predicted onset within
                          t ms (from the `all_distances` of each run's scoring JSON)

    python evaluation/plot_summary.py
    python evaluation/plot_summary.py --plots-dir tmp/plots --out evaluation/plots

`compute_metrics.py` writes each run's JSON and histogram to tmp/plots/; the copies the docs
use live in evaluation/plots/, which is where this reads from by default. Only full-corpus
runs (`cfg_subset` -1) with the published decoder settings are drawn: one row per method.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("timit", "buckeye")
DATASET_TITLES = {"timit": "TIMIT (read speech, 6,300 clips)",
                  "buckeye": "Buckeye (conversational, 255 recordings)"}

# Fixed order and colour per method, the same in every panel. Line style is the family:
# solid p3cupe (ph66), dashed p4mbfa (std), dotted MFA.
METHODS = [
    ("CUPE-3-en", "CUPE-3-en (ph66)", "#2a78d6", "-"),
    ("CUPE-3-en-base", "CUPE-3-en-base (ph66)", "#eb6834", "-"),
    ("CUPE-3-mls7", "CUPE-3-mls7 (ph66)", "#1baf7a", "-"),
    ("mbfa-latin", "mbfa-latin (std)", "#eda100", "--"),
    ("mbfa-latin-benchmark", "mbfa-latin-benchmark (std)", "#e87ba4", "--"),
    ("mfa-english_us_arpa", "MFA english_us_arpa", "#008300", ":"),
]
TOLERANCES = (10, 25, 50)

INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#ffffff"


def load_rows(results_dir):
    """{dataset: {model: row}} for the full-corpus runs."""
    out = {}
    for ds in DATASETS:
        path = results_dir / f"results_{ds}.csv"
        if not path.is_file():
            sys.exit(f"{path} not found; run compute_metrics.py first")
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("cfg_subset", "-1") in ("-1", "")]
        out[ds] = {r["cfg_model"]: r for r in rows}
    return out


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)


def plot_f1(rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
    n = len(METHODS)
    width = 0.8 / n
    x = np.arange(len(TOLERANCES))
    for ax, ds in zip(axes, DATASETS):
        style_axes(ax)
        ax.grid(axis="x", visible=False)
        for i, (model, label, color, _) in enumerate(METHODS):
            row = rows[ds].get(model)
            if row is None:
                continue
            vals = [float(row[f"f1_{t}ms"]) for t in TOLERANCES]
            pos = x - 0.4 + width * (i + 0.5)
            ax.bar(pos, vals, width * 0.86, color=color, label=label, zorder=2)
        ax.set_xticks(x, [f"within {t} ms" for t in TOLERANCES])
        ax.set_ylim(40, 95)
        ax.set_title(DATASET_TITLES[ds], color=INK, fontsize=11, loc="left")
    axes[0].set_ylabel("Boundary F1 (%)", color=MUTED, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_distance(rows, plots_dir, out_path, t_max=60):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
    grid_t = np.arange(0, t_max + 1)
    for ax, ds in zip(axes, DATASETS):
        style_axes(ax)
        for model, label, color, ls in METHODS:
            row = rows[ds].get(model)
            if row is None:
                continue
            path = plots_dir / f"{row['experiment']}.json"
            if not path.is_file():
                print(f"skip {ds}/{model}: {path} not found")
                continue
            with open(path) as f:
                dist = np.sort(np.asarray(json.load(f)["all_distances"], dtype=float))
            share = np.searchsorted(dist, grid_t, side="right") / len(dist) * 100
            ax.plot(grid_t, share, color=color, linestyle=ls, linewidth=2, label=label,
                    solid_capstyle="round", dash_capstyle="round")
        for t in TOLERANCES:
            ax.axvline(t, color=GRID, linewidth=1, zorder=0)
        ax.set_xlim(0, t_max)
        ax.set_ylim(0, 100)
        ax.set_xlabel("distance to the nearest predicted onset (ms)", color=MUTED, fontsize=10)
        ax.set_title(DATASET_TITLES[ds], color=INK, fontsize=11, loc="left")
    axes[0].set_ylabel("hand-labelled boundaries within (%)", color=MUTED, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "evaluation" / "results"))
    ap.add_argument("--plots-dir", default=str(REPO_ROOT / "evaluation" / "plots"),
                    help="where the scoring JSONs are (compute_metrics.py writes tmp/plots/)")
    ap.add_argument("--out", default=str(REPO_ROOT / "evaluation" / "plots"))
    args = ap.parse_args()

    rows = load_rows(Path(args.results_dir))
    out = Path(args.out)
    os.makedirs(out, exist_ok=True)
    plot_f1(rows, out / "summary_f1.png")
    plot_distance(rows, Path(args.plots_dir), out / "summary_distance.png")
    print(f"wrote {out / 'summary_f1.png'} and {out / 'summary_distance.png'}")


if __name__ == "__main__":
    main()
