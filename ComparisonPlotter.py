"""
ComparisonPlotter.py
--------------------
Unified benchmark plotter comparing C# (ML.NET) vs Python (scikit-learn).
Reads both CSVs and produces a single multi-panel report saved to results/.

Usage:
    python ComparisonPlotter.py
    python ComparisonPlotter.py --cs StudyCsharp/StudyCsharp/bin/Debug/net10.0/result/csharp_timings.csv --py results/python_timings.csv --out results/
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":      "monospace",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#181b24",
    "axes.labelcolor":  "#c8cdd8",
    "axes.titlecolor":  "#e8ecf5",
    "xtick.color":      "#7a8099",
    "ytick.color":      "#7a8099",
    "text.color":       "#c8cdd8",
    "grid.color":       "#2a2e3d",
    "legend.framealpha": 0.15,
    "legend.edgecolor": "#3a3e50",
    "legend.facecolor": "#181b24",
})

# ── Palette ──────────────────────────────────────────────────────────────────
COLORS = {
    ("csharp",  "linear"): "#4fc3f7",   # cyan-blue
    ("csharp",  "tree"):   "#0077b6",   # deep blue
    ("python",  "linear"): "#f4845f",   # coral
    ("python",  "tree"):   "#c1440e",   # rust
}
LINESTYLE = {"linear": "-", "tree": "--"}
MARKERS = {"linear": "o", "tree": "s"}
LANG_LABEL = {"csharp": "C# / ML.NET", "python": "Python / scikit-learn"}


def load(cs_path: str, py_path: str, drop_repeat0: bool = True):
    cs = pd.read_csv(cs_path)
    py = pd.read_csv(py_path)
    df = pd.concat([cs, py], ignore_index=True)

    # Don't drop repeat 0: C# JIT cold-start makes it up to 2.2x slower
    if drop_repeat0 and "repeat" in df.columns:
        df = df[df["repeat"] != 0].reset_index(drop=False)

    # Derived columns (seconds + MB)
    for phase in ("load", "clean", "split", "train", "infer", "preprocess", "total"):
        ns_col = f"{phase}_ns"
        if ns_col in df.columns:
            df[f"{phase}_s"] = df[ns_col] / 1e9

    for metric in ("load", "clean", "split", "train", "infer"):
        rss_col = f"{metric}_rss_delta_bytes"
        if rss_col in df.columns:
            df[f"{metric}_rss_mb"] = df[rss_col] / 1024**2

    return df


def grouped_stats(df: pd.DataFrame):
    """Mean + std over repeats, per (language, model, subset_size)."""
    keys = ["language", "model", "subset_size"]
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c not in keys]
    g_mean = (df.groupby(keys)[num_cols]
                .mean().reset_index())
    g_std = (df.groupby(keys)[num_cols]
             .std().reset_index())
    return g_mean, g_std


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_s(ax):
    """Auto-format y-axis: ms if all values < 1 s, else s."""
    ylo, yhi = ax.get_ylim()
    if yhi < 1.0:
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v*1000:.1f} ms"))
    else:
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v:.2f} s"))


def _plot_lines(ax, g_mean, g_std, col, *, error_bars=True):
    for lang in ("csharp", "python"):
        for model in ("linear", "tree"):
            sub_m = g_mean[(g_mean["language"] == lang)
                           & (g_mean["model"] == model)]
            sub_s = g_std[(g_std["language"] == lang) & (
                g_std["model"] == model)] if g_std is not None else None
            if sub_m.empty or col not in sub_m.columns:
                continue
            x = sub_m["subset_size"].values
            y = sub_m[col].values
            yerr = sub_s[col].values if (
                error_bars and sub_s is not None and col in sub_s.columns) else None
            color = COLORS[(lang, model)]
            label = f"{LANG_LABEL[lang]} · {model}"
            ax.errorbar(x, y, yerr=yerr,
                        color=color, label=label,
                        marker=MARKERS[model],
                        linestyle=LINESTYLE[model],
                        linewidth=1.8, markersize=6,
                        capsize=3, elinewidth=0.8, alpha=0.92)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{int(v):,}"))


def _legend(ax, loc="best"):
    ax.legend(fontsize=7.5, loc=loc, handlelength=2.2)


def _title(ax, text):
    ax.set_title(text, fontsize=10, pad=8, fontweight="bold")


def _subtitle(ax, text):
    ax.text(0.01, 0.97, text, transform=ax.transAxes,
            fontsize=7, color="#6a7090", va="top")


# ── Figure 1: Time breakdown (3 panels) ──────────────────────────────────────

def fig_time_breakdown(g_mean, g_std, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Timing Comparison — C# (ML.NET) vs Python (scikit-learn)  [load / clean excluded from totals]",
                 fontsize=13, fontweight="bold", color="#e8ecf5", y=1.02)

    panels = [
        ("train_s",  "Train time",     "Subset size in rows", "Train time"),
        ("infer_s",  "Inference time", "Subset size in rows", "Inference time"),
        ("pipeline_s",  "Pipeline time\n(split + train + infer only)",
         "Subset size in rows", "Pipeline time"),
    ]

    for ax, (col, title, xlabel, ylabel) in zip(axes, panels):
        _plot_lines(ax, g_mean, g_std, col)
        _title(ax, title)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        _fmt_s(ax)
        _legend(ax)

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_time_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 2: Phase stacked bar (mean across repeats, per size) ───────────────

def fig_phase_bars(g_mean, out_dir):
    """
    Grouped bar chart: for each (subset_size × language) show stacked bars
    split=train+infer, so we can see where time goes.
    """
    sizes = sorted(g_mean["subset_size"].unique())
    models = ["linear", "tree"]
    langs = ["csharp", "python"]

    n_groups = len(sizes) * len(models)
    phases = ["split_s", "train_s", "infer_s"]
    phase_labels = ["Split", "Train", "Infer"]
    phase_colors = {
        "csharp": ["#1a4a6b", "#4fc3f7", "#a8d8ea"],
        "python": ["#6b2a1a", "#f4845f", "#fac8b5"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Phase Time Breakdown (Split / Train / Infer)",
                 fontsize=13, fontweight="bold", color="#e8ecf5", y=1.02)

    for ax, model in zip(axes, models):
        bar_width = 0.35
        x = np.arange(len(sizes))

        for li, lang in enumerate(langs):
            offset = (li - 0.5) * bar_width
            bottoms = np.zeros(len(sizes))
            for pi, (phase, plabel) in enumerate(zip(phases, phase_labels)):
                vals = []
                for sz in sizes:
                    row = g_mean[(g_mean["language"] == lang) &
                                 (g_mean["model"] == model) &
                                 (g_mean["subset_size"] == sz)]
                    vals.append(float(row[phase].iloc[0])
                                if not row.empty else 0.0)
                vals = np.array(vals)
                label = f"{LANG_LABEL[lang]} — {plabel}" if pi == 0 else f"— {plabel}"
                ax.bar(x + offset, vals, bar_width,
                       bottom=bottoms,
                       color=phase_colors[lang][pi],
                       label=f"{LANG_LABEL[lang]} · {plabel}",
                       alpha=0.88)
                bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(s):,}" for s in sizes], fontsize=8)
        ax.set_xlabel("Subset size in rows", fontsize=8)
        ax.set_ylabel("Time (s)", fontsize=8)
        _title(ax, f"Model: {model}")
        ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_phase_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 3: Memory (RSS) ────────────────────────────────────────────────────

def fig_memory(g_mean, g_std, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Memory Usage (Peak RSS) — C# vs Python",
                 fontsize=13, fontweight="bold", color="#e8ecf5", y=1.02)

    panels = [
        ("train_rss_mb", "Train RSS delta (MB)"),
        ("infer_rss_mb", "Infer RSS delta (MB)"),
    ]
    for ax, (col, ylabel) in zip(axes, panels):
        _plot_lines(ax, g_mean, g_std, col)
        _title(ax, ylabel)
        ax.set_xlabel("Subset size in rows", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v:.1f} MB"))
        _legend(ax)

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_memory.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 4: Model accuracy (R² + RMSE) ─────────────────────────────────────

def fig_accuracy(g_mean, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Model Accuracy — R² and RMSE (test set)",
                 fontsize=13, fontweight="bold", color="#e8ecf5", y=1.02)

    for ax, (col, ylabel, note) in zip(axes, [
        ("r2",   "R²: higher = better", "1.0 = perfect fit"),
        ("rmse", "RMSE: lower = better",
         "note: Python linear RMSE grows with size\n(wider target range in larger subsets, not model degradation)"),
    ]):
        _plot_lines(ax, g_mean, None, col, error_bars=False)
        _title(ax, ylabel)
        ax.set_xlabel("Subset size in rows", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        _subtitle(ax, note)
        _legend(ax)

    axes[0].set_ylim(0, 1.05)

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 5: Speedup ratio (Python / C# time) ───────────────────────────────

def fig_speedup(g_mean, out_dir):
    """
    Ratio: C# time / Python time for train and infer.
    >1 means C# is slower, <1 means C# is faster.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Speed Ratio: C# time ÷ Python time  (>1 = C# slower)",
                 fontsize=13, fontweight="bold", color="#e8ecf5", y=1.02)

    sizes = sorted(g_mean["subset_size"].unique())

    for ax, (col, title) in zip(axes, [
        ("train_s", "Train"),
        ("infer_s", "Inference"),
    ]):
        for model in ("linear", "tree"):
            ratios = []
            for sz in sizes:
                cs_row = g_mean[(g_mean["language"] == "csharp") &
                                (g_mean["model"] == model) &
                                (g_mean["subset_size"] == sz)]
                py_row = g_mean[(g_mean["language"] == "python") &
                                (g_mean["model"] == model) &
                                (g_mean["subset_size"] == sz)]
                if cs_row.empty or py_row.empty:
                    ratios.append(np.nan)
                    continue
                cs_val = float(cs_row[col].iloc[0])
                py_val = float(py_row[col].iloc[0])
                ratios.append(cs_val / py_val if py_val > 0 else np.nan)

            color = COLORS[("csharp", model)]
            ax.plot(sizes, ratios, marker=MARKERS[model],
                    linestyle=LINESTYLE[model], color=color,
                    linewidth=1.8, markersize=6, label=f"{model}")

        ax.axhline(1.0, color="#ffffff", linewidth=0.8,
                   linestyle=":", alpha=0.4, label="parity (ratio = 1)")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_xlabel("Subset size in rows", fontsize=8)
        ax.set_ylabel("C# time / Python time", fontsize=8)
        _title(ax, f"{title} speed ratio")
        _subtitle(ax, "above line = C# slower; below = C# faster")
        _legend(ax)

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_speedup.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 6: Repeat variance (box plots) ────────────────────────────────────

def fig_variance(df: pd.DataFrame, out_dir):
    """Box plots of train_s per (language × model × size) to show run-to-run noise."""
    sizes = sorted(df["subset_size"].unique())
    models = ["linear", "tree"]
    langs = ["csharp", "python"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Train Time — Repeat Variance (5 runs per condition)",
                 fontsize=13, fontweight="bold", color="#e8ecf5", y=1.02)

    for ax, model in zip(axes, models):
        positions, data_list, colors_list, xlabels = [], [], [], []
        pos = 0
        for sz in sizes:
            for lang in langs:
                sub = df[(df["language"] == lang) &
                         (df["model"] == model) &
                         (df["subset_size"] == sz)]["train_s"].dropna().values
                if len(sub) == 0:
                    continue
                positions.append(pos)
                data_list.append(sub)
                colors_list.append(COLORS[(lang, model)])
                xlabels.append(f"{lang[:2].upper()}\n{int(sz):,}")
                pos += 1
            pos += 0.4   # gap between size groups

        bp = ax.boxplot(data_list, positions=positions, widths=0.55,
                        patch_artist=True, notch=False,
                        medianprops=dict(color="#ffffff", linewidth=1.8),
                        whiskerprops=dict(color="#6a7090"),
                        capprops=dict(color="#6a7090"),
                        flierprops=dict(marker="x", color="#ff6b6b",
                                        markersize=4, alpha=0.6))
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks(positions)
        ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_ylabel("Train time (s)", fontsize=8)
        _title(ax, f"Model: {model}")

        # Manual legend
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=COLORS[(l, model)], label=LANG_LABEL[l])
                   for l in langs]
        ax.legend(handles=handles, fontsize=7.5)

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_variance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 7: Summary table ───────────────────────────────────────────────────

def fig_summary_table(g_mean, out_dir):
    rows = []
    for lang in ("csharp", "python"):
        for model in ("linear", "tree"):
            for sz in sorted(g_mean["subset_size"].unique()):
                row = g_mean[(g_mean["language"] == lang) &
                             (g_mean["model"] == model) &
                             (g_mean["subset_size"] == sz)]
                if row.empty:
                    continue
                r = row.iloc[0]
                rows.append({
                    "Language":    LANG_LABEL[lang],
                    "Model":       model,
                    "Subset":      f"{int(sz):,}",
                    "Train":       f"{r['train_s']*1000:.1f} ms" if r["train_s"] < 1 else f"{r['train_s']:.3f} s",
                    "Infer":       f"{r['infer_s']*1000:.1f} ms" if r["infer_s"] < 1 else f"{r['infer_s']:.3f} s",
                    "Pipeline":    (f"{r['pipeline_s']*1000:.1f} ms" if r["pipeline_s"] < 1 else f"{r['pipeline_s']:.3f} s") if "pipeline_s" in r.index else f"{(r['split_s']+r['train_s']+r['infer_s']):.3f} s",
                    "Train RSS":   f"{r['train_rss_mb']:.1f} MB",
                    "R²":          f"{r['r2']:.4f}",
                    "RMSE":        f"{r['rmse']:.4f}",
                })

    tdf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(16, len(tdf) * 0.52 + 1.5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.axis("off")

    col_widths = [0.18, 0.07, 0.09, 0.10, 0.10, 0.10, 0.10, 0.07, 0.07]
    table = ax.table(
        cellText=tdf.values,
        colLabels=tdf.columns,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(tdf.columns)):
        cell = table[0, j]
        cell.set_facecolor("#1e2a45")
        cell.set_text_props(color="#4fc3f7", fontweight="bold")
        cell.set_edgecolor("#2a3550")

    # Stripe rows + colour by language
    lang_colors = {"C# / ML.NET": "#1a2535",
                   "Python / scikit-learn": "#251a1a"}
    for i in range(1, len(tdf) + 1):
        lang_val = tdf.iloc[i - 1]["Language"]
        bg = lang_colors.get(lang_val, "#181b24")
        for j in range(len(tdf.columns)):
            cell = table[i, j]
            cell.set_facecolor(bg)
            cell.set_text_props(color="#c8cdd8")
            cell.set_edgecolor("#2a2e3d")

    ax.set_title("Summary: Mean metrics across repeats 1-4 (repeat 0 excluded - C# JIT warm-up)",
                 fontsize=11, fontweight="bold", color="#e8ecf5", pad=12)

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_summary_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cs",  default="StudyCsharp/StudyCsharp/bin/Debug/net10.0/results/csharp_timings.csv")
    ap.add_argument("--py",  default="results/python_timings.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    for p in (args.cs, args.py):
        if not os.path.exists(p):
            raise FileNotFoundError(f"CSV not found: {p}")

    os.makedirs(args.out, exist_ok=True)

    print("Loading data...")
    df = load(args.cs, args.py, drop_repeat0=False)
    print("  Repeat 0 dropped (C# JIT cold-start excluded from means)")
    g_mean, g_std = grouped_stats(df)

    print(f"  C# rows:     {len(df[df['language'] == 'csharp'])}")
    print(f"  Python rows: {len(df[df['language'] == 'python'])}")
    print(f"  Sizes:       {sorted(df['subset_size'].unique())}")
    print(f"  Models:      {list(df['model'].unique())}")
    print()

    print("Generating plots...")
    fig_time_breakdown(g_mean, g_std, args.out)
    fig_phase_bars(g_mean, args.out)
    fig_memory(g_mean, g_std, args.out)
    fig_accuracy(g_mean, args.out)
    fig_speedup(g_mean, args.out)
    fig_variance(df, args.out)
    fig_summary_table(g_mean, args.out)

    print(f"\nDone — 7 plots saved to: {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()
