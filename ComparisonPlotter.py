"""
ComparisonPlotter.py
--------------------
Unified benchmark plotter comparing C# (ML.NET) vs Python (scikit-learn).
Reads both CSVs and produces a single multi-panel report saved to results/.

Usage:
    python ComparisonPlotter.py
    python ComparisonPlotter.py --cs StudyCsharp/StudyCsharp/bin/Debug/net10.0/results/csharp_timings.csv --py results/python_timings.csv --out results/
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#181b24",
    "axes.labelcolor":   "#c8cdd8",
    "axes.titlecolor":   "#e8ecf5",
    "xtick.color":       "#7a8099",
    "ytick.color":       "#7a8099",
    "text.color":        "#c8cdd8",
    "grid.color":        "#2a2e3d",
    "legend.framealpha": 0.15,
    "legend.edgecolor":  "#3a3e50",
    "legend.facecolor":  "#181b24",
})

# Palette
COLORS = {
    ("csharp", "linear"): "#4fc3f7",
    ("csharp", "tree"):   "#0077b6",
    ("python", "linear"): "#f4845f",
    ("python", "tree"):   "#c1440e",
}
LINESTYLE = {"linear": "-",  "tree": "--"}
MARKERS = {"linear": "o",  "tree": "s"}
LANG_LABEL = {"csharp": "C# / ML.NET", "python": "Python / scikit-learn"}


def load(cs_path: str, py_path: str) -> pd.DataFrame:
    # The C# CSV may be missing the wall_clock_ns column header (one extra data
    # value per row with no matching header).  Detect and insert the missing
    # header before reading so all columns align correctly.
    import io as _io
    with open(cs_path, encoding="utf-8") as fh:
        raw_cs = fh.read()
    cs_lines = raw_cs.splitlines()
    cs_header = cs_lines[0].split(",")
    n_header = len(cs_header)
    n_data = len(cs_lines[1].split(",")) if len(cs_lines) > 1 else n_header
    if n_data == n_header + 1 and "wall_clock_ns" not in cs_header:
        # Insert the missing wall_clock_ns header before r2
        insert_pos = cs_header.index("r2") if "r2" in cs_header else n_header
        cs_header.insert(insert_pos, "wall_clock_ns")
        cs_lines[0] = ",".join(cs_header)
        raw_cs = "\n".join(cs_lines)
    cs = pd.read_csv(_io.StringIO(raw_cs))
    py = pd.read_csv(py_path)
    df = pd.concat([cs, py], ignore_index=True)

    # Normalise language labels — C# CSV writes 'ml.net', plotter expects 'csharp'
    if "language" in df.columns:
        df["language"] = df["language"].replace({"ml.net": "csharp"})

    for phase in ("load", "clean", "split", "train", "infer",
                  "preprocess", "pipeline", "total", "wall_clock"):
        ns_col = f"{phase}_ns"
        if ns_col in df.columns:
            df[f"{phase}_s"] = df[ns_col] / 1e9

    if "pipeline_s" not in df.columns and all(
            c in df.columns for c in ("split_s", "train_s", "infer_s")):
        df["pipeline_s"] = df["split_s"] + df["train_s"] + df["infer_s"]

    for metric in ("load", "clean", "split", "train", "infer"):
        rss_col = f"{metric}_rss_delta_bytes"
        if rss_col in df.columns:
            df[f"{metric}_rss_mb"] = df[rss_col] / 1024 ** 2

    return df


def grouped_stats(df: pd.DataFrame):
    keys = ["language", "model", "subset_size"]
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c not in keys]
    g_mean = df.groupby(keys)[num_cols].mean().reset_index()
    g_std = df.groupby(keys)[num_cols].std().reset_index()
    return g_mean, g_std


def _fmt_s(ax):
    lo, hi = ax.get_ylim()
    if hi < 1.0:
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
            sub_s = g_std[(g_std["language"] == lang) & (g_std["model"] == model)] \
                if g_std is not None else None
            if sub_m.empty or col not in sub_m.columns:
                continue
            x = sub_m["subset_size"].values
            y = sub_m[col].values
            yerr = sub_s[col].values if (
                error_bars and sub_s is not None and col in sub_s.columns) else None
            ax.errorbar(x, y, yerr=yerr,
                        color=COLORS[(lang, model)],
                        label=f"{LANG_LABEL[lang]} · {model}",
                        marker=MARKERS[model], linestyle=LINESTYLE[model],
                        linewidth=1.8, markersize=6,
                        capsize=3, elinewidth=0.8, alpha=0.92)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))


def _legend(ax, loc="best"):
    ax.legend(fontsize=7.5, loc=loc, handlelength=2.2)


def _title(ax, text):
    ax.set_title(text, fontsize=10, pad=8, fontweight="bold")


def _subtitle(ax, text):
    ax.text(0.01, 0.97, text, transform=ax.transAxes,
            fontsize=7, color="#6a7090", va="top")


def fig_time_breakdown(g_mean, g_std, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        "Timing Comparison — C# (ML.NET) vs Python (scikit-learn)"
        "  [load / clean excluded from totals]",
        fontsize=13, fontweight="bold", color="#e8ecf5", y=1.02)

    for ax, (col, title, ylabel) in zip(axes, [
        ("train_s",    "Train time",                             "Train time"),
        ("infer_s",    "Inference time",                         "Inference time"),
        ("pipeline_s", "Pipeline time\n(split+train+infer only)", "Pipeline time"),
    ]):
        _plot_lines(ax, g_mean, g_std, col)
        _title(ax, title)
        ax.set_xlabel("Subset size (rows)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        _fmt_s(ax)
        _legend(ax)

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_time_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_phase_bars(g_mean, out_dir):
    sizes = sorted(g_mean["subset_size"].unique())
    phase_colors = {
        "csharp": ["#1a4a6b", "#4fc3f7", "#a8d8ea"],
        "python": ["#6b2a1a", "#f4845f", "#fac8b5"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Phase Time Breakdown (Split / Train / Infer)",
                 fontsize=13, fontweight="bold", color="#e8ecf5", y=1.02)

    for ax, model in zip(axes, ["linear", "tree"]):
        x = np.arange(len(sizes))
        for li, lang in enumerate(["csharp", "python"]):
            offset = (li - 0.5) * 0.35
            bottoms = np.zeros(len(sizes))
            for pi, (phase, plabel) in enumerate(zip(
                    ["split_s", "train_s", "infer_s"], ["Split", "Train", "Infer"])):
                vals = np.array([
                    float(g_mean[(g_mean["language"] == lang) &
                                 (g_mean["model"] == model) &
                                 (g_mean["subset_size"] == sz)][phase].iloc[0])
                    if not g_mean[(g_mean["language"] == lang) &
                                  (g_mean["model"] == model) &
                                  (g_mean["subset_size"] == sz)].empty else 0.0
                    for sz in sizes])
                ax.bar(x + offset, vals, 0.35, bottom=bottoms,
                       color=phase_colors[lang][pi],
                       label=f"{LANG_LABEL[lang]} · {plabel}", alpha=0.88)
                bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(s):,}" for s in sizes], fontsize=8)
        ax.set_xlabel("Subset size (rows)", fontsize=8)
        ax.set_ylabel("Time (s)", fontsize=8)
        _title(ax, f"Model: {model}")
        ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_phase_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_memory(g_mean, g_std, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Memory Usage (Peak RSS) — C# vs Python",
                 fontsize=13, fontweight="bold", color="#e8ecf5", y=1.02)

    for ax, (col, ylabel) in zip(axes, [
        ("train_rss_mb", "Train peak RSS (MB)"),
        ("infer_rss_mb", "Infer peak RSS (MB)"),
    ]):
        _plot_lines(ax, g_mean, g_std, col)
        _title(ax, ylabel)
        ax.set_xlabel("Subset size (rows)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v:,.0f} MB"))
        _subtitle(ax, "peak working set polled every 5 ms during phase")
        _legend(ax)

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_memory.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_accuracy(g_mean, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Model Accuracy — R² and RMSE (test set)",
                 fontsize=13, fontweight="bold", color="#e8ecf5", y=1.02)

    for ax, (col, ylabel, note) in zip(axes, [
        ("r2",   "R²  (higher = better)", "1.0 = perfect fit"),
        ("rmse", "RMSE (lower = better)",
         "Python linear RMSE grows with size:\nwider target range in larger subsets"),
    ]):
        _plot_lines(ax, g_mean, None, col, error_bars=False)
        _title(ax, ylabel)
        ax.set_xlabel("Subset size (rows)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        _subtitle(ax, note)
        _legend(ax)

    axes[0].set_ylim(0, 1.05)
    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_speedup(g_mean, out_dir):
    sizes = sorted(g_mean["subset_size"].unique())
    cs_data = g_mean[g_mean["language"] == "csharp"]
    py_data = g_mean[g_mean["language"] == "python"]
    if cs_data.empty or py_data.empty:
        print("  Skipped fig_speedup: need both languages")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Speed Ratio: C# time ÷ Python time  (>1 = C# slower)",
                 fontsize=13, fontweight="bold", color="#e8ecf5", y=1.02)

    for ax, (col, title) in zip(axes, [
        ("train_s", "Train speed ratio"),
        ("infer_s", "Inference speed ratio"),
    ]):
        for model in ("linear", "tree"):
            ratios = []
            for sz in sizes:
                cs_row = cs_data[(cs_data["model"] == model)
                                 & (cs_data["subset_size"] == sz)]
                py_row = py_data[(py_data["model"] == model)
                                 & (py_data["subset_size"] == sz)]
                if cs_row.empty or py_row.empty:
                    ratios.append(np.nan)
                    continue
                cv = float(cs_row[col].iloc[0])
                pv = float(py_row[col].iloc[0])
                ratios.append(cv / pv if pv > 0 else np.nan)

            ax.plot(sizes, ratios,
                    marker=MARKERS[model], linestyle=LINESTYLE[model],
                    color=COLORS[("csharp", model)],
                    linewidth=1.8, markersize=6, label=model)

        ax.axhline(1.0, color="#ffffff", linewidth=0.8,
                   linestyle=":", alpha=0.4, label="parity (ratio = 1)")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_xlabel("Subset size (rows)", fontsize=8)
        ax.set_ylabel("C# time / Python time", fontsize=8)
        _title(ax, title)
        _subtitle(ax, "above line = C# slower; below = C# faster")
        _legend(ax)

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_speedup.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_variance(df: pd.DataFrame, out_dir):
    sizes = sorted(df["subset_size"].unique())
    langs = ["csharp", "python"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Train Time — Repeat Variance (all runs per condition)",
                 fontsize=13, fontweight="bold", color="#e8ecf5", y=1.02)

    for ax, model in zip(axes, ["linear", "tree"]):
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
            pos += 0.4

        if not data_list:
            continue

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

        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=COLORS[(l, model)], label=LANG_LABEL[l])
                           for l in langs], fontsize=7.5)

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_variance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_wall_clock(g_mean, g_std, out_dir):
    if "wall_clock_s" not in g_mean.columns:
        print("  Skipped fig_wall_clock: wall_clock_ns not in CSV yet — re-run benchmarks")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey="row")
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        "Wall-Clock Time vs Measured Pipeline Time\n"
        "Gap between rows = framework overhead outside individual phase timers",
        fontsize=12, fontweight="bold", color="#e8ecf5", y=1.02)

    row_configs = [
        ("wall_clock_s", "Wall-clock (true end-to-end)",
         "Single timer around entire repeat — includes all framework overhead"),
        ("pipeline_s",   "Pipeline (split + train + infer only)",
         "Sum of individual phase timers — excludes framework overhead"),
    ]

    for row_idx, (col, row_title, row_note) in enumerate(row_configs):
        for col_idx, model in enumerate(["linear", "tree"]):
            ax = axes[row_idx][col_idx]
            _plot_lines(ax, g_mean, g_std, col)
            ax.set_xlabel("Subset size (rows)", fontsize=8)
            ax.set_ylabel("Time (s)", fontsize=8)
            _fmt_s(ax)
            _legend(ax)
            _subtitle(ax, row_note)
            if row_idx == 0:
                _title(ax, f"Model: {model}")
            if col_idx == 0:
                ax.set_ylabel(f"{row_title}\n\nTime (s)", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_wall_clock.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_summary_table(g_mean, out_dir):
    has_wc = "wall_clock_s" in g_mean.columns
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

                def _fmt(v):
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        return "n/a"
                    return f"{v*1000:.1f} ms" if abs(v) < 1 else f"{v:.3f} s"

                pl = r["pipeline_s"] if "pipeline_s" in r.index else (
                    r["split_s"] + r["train_s"] + r["infer_s"])
                wc = r["wall_clock_s"] if has_wc else None

                entry = {
                    "Language": LANG_LABEL[lang],
                    "Model":    model,
                    "Subset":   f"{int(sz):,}",
                    "Train":    _fmt(r["train_s"]),
                    "Infer":    _fmt(r["infer_s"]),
                    "Pipeline": _fmt(pl),
                }
                if has_wc:
                    entry["Wall-clock"] = _fmt(wc)
                    entry["Overhead"] = _fmt(
                        (wc - pl) if wc is not None else None)
                entry["Train RSS"] = f"{r['train_rss_mb']:.1f} MB"
                entry["R²"] = f"{r['r2']:.4f}"
                entry["RMSE"] = f"{r['rmse']:.4f}"
                rows.append(entry)

    tdf = pd.DataFrame(rows)
    n_cols = len(tdf.columns)

    fig, ax = plt.subplots(figsize=(16, len(tdf) * 0.52 + 1.5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.axis("off")

    table = ax.table(cellText=tdf.values, colLabels=tdf.columns,
                     cellLoc="center", loc="center",
                     colWidths=[1.0 / n_cols] * n_cols)
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)

    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#1e2a45")
        cell.set_text_props(color="#4fc3f7", fontweight="bold")
        cell.set_edgecolor("#2a3550")

    lang_colors = {"C# / ML.NET": "#1a2535",
                   "Python / scikit-learn": "#251a1a"}
    for i in range(1, len(tdf) + 1):
        bg = lang_colors.get(tdf.iloc[i - 1]["Language"], "#181b24")
        for j in range(n_cols):
            cell = table[i, j]
            cell.set_facecolor(bg)
            cell.set_text_props(color="#c8cdd8")
            cell.set_edgecolor("#2a2e3d")

    ax.set_title("Summary: Mean metrics across all repeats (repeats 0-4)",
                 fontsize=11, fontweight="bold", color="#e8ecf5", pad=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_summary_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 9: All five phases — individual line charts ───────────────────────

def fig_all_phases(g_mean, g_std, out_dir):
    """
    Two separate figures:
      1. Load + Clean   — one-time I/O and preprocessing phases (flat lines expected)
      2. Split + Train + Infer — repeatable ML pipeline phases
    Each panel has its own y-axis scale so small phases remain readable.
    """

    # ── Figure A: Load and Clean (one-time phases) ─────────────────────────
    io_phases = [
        ("load_s",  "Load",  "Time to read CSV from disk"),
        ("clean_s", "Clean", "Type coercion · NaN/Inf drop · dedup"),
    ]
    fig_io, axes_io = plt.subplots(1, 2, figsize=(10, 5))
    fig_io.patch.set_facecolor("#0f1117")
    fig_io.suptitle(
        "I/O & Cleaning Phases — C# (ML.NET) vs Python (scikit-learn)\n"
        "(measured once per subset — flat lines expected across repeats)",
        fontsize=12, fontweight="bold", color="#e8ecf5", y=1.04)

    for ax, (col, phase_name, note) in zip(axes_io, io_phases):
        _plot_lines(ax, g_mean, g_std, col)
        _title(ax, phase_name)
        ax.set_xlabel("Subset size (rows)", fontsize=8)
        ax.set_ylabel("Time (s)", fontsize=8)
        _fmt_s(ax)
        _subtitle(ax, note)
        _legend(ax)

    fig_io.tight_layout()
    path_io = os.path.join(out_dir, "comparison_phases_io.png")
    fig_io.savefig(path_io, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig_io)
    print(f"  Saved: {path_io}")

    # ── Figure B: Split, Train, Infer (repeatable pipeline phases) ──────────
    ml_phases = [
        ("split_s", "Split", "Shuffle + partition (pure index split)"),
        ("train_s", "Train", "Model fitting on training set"),
        ("infer_s", "Infer", "Prediction on test set"),
    ]
    fig_ml, axes_ml = plt.subplots(1, 3, figsize=(15, 5))
    fig_ml.patch.set_facecolor("#0f1117")
    fig_ml.suptitle(
        "Pipeline Phases — C# (ML.NET) vs Python (scikit-learn)\n"
        "(split + train + infer — repeated 10× per configuration)",
        fontsize=12, fontweight="bold", color="#e8ecf5", y=1.04)

    for ax, (col, phase_name, note) in zip(axes_ml, ml_phases):
        _plot_lines(ax, g_mean, g_std, col)
        _title(ax, phase_name)
        ax.set_xlabel("Subset size (rows)", fontsize=8)
        ax.set_ylabel("Time (s)", fontsize=8)
        _fmt_s(ax)
        _subtitle(ax, note)
        _legend(ax)

    fig_ml.tight_layout()
    path_ml = os.path.join(out_dir, "comparison_phases_pipeline.png")
    fig_ml.savefig(path_ml, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig_ml)
    print(f"  Saved: {path_ml}")


# ── Figure 10: Phases vs wall-clock (stacked area + overlay) ─────────────────

def fig_phases_vs_wallclock(g_mean, g_std, out_dir):
    """
    For each language × model combination, shows a stacked bar of the five
    measured phases (load + clean + split + train + infer) next to a marker
    for pipeline_s (split+train+infer) and — if available — wall_clock_s.
    This makes the relationship between the summed phases and the true
    end-to-end time immediately visible.

    Layout: 2 rows (linear / tree) × 2 columns (C# / Python).
    """
    has_wc = "wall_clock_s" in g_mean.columns
    sizes = sorted(g_mean["subset_size"].unique())
    langs = ["csharp", "python"]
    models = ["linear", "tree"]

    # Colours for each phase segment (same for both languages, distinguished by hatch)
    seg_colors = {
        "load_s":  "#2e4a6b",
        "clean_s": "#1e6b5a",
        "split_s": "#6b3a1e",
        "train_s": "#4fc3f7",
        "infer_s": "#a8d8ea",
    }
    seg_labels = {
        "load_s":  "Load",
        "clean_s": "Clean",
        "split_s": "Split",
        "train_s": "Train",
        "infer_s": "Infer",
    }
    hatches = {"csharp": "", "python": "//"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
    fig.patch.set_facecolor("#0f1117")
    title_suffix = " (wall-clock shown as ✕)" if has_wc else ""
    fig.suptitle(
        f"Measured Phase Sum vs Total Time{title_suffix}\n"
        "Bars = stacked phases  |  ◆ = pipeline (split+train+infer)"
        + ("  |  ✕ = wall-clock (true end-to-end)" if has_wc else ""),
        fontsize=11, fontweight="bold", color="#e8ecf5", y=1.02)

    for row_idx, model in enumerate(models):
        for col_idx, lang in enumerate(langs):
            ax = axes[row_idx][col_idx]
            x = np.arange(len(sizes))
            bw = 0.5

            bottoms = np.zeros(len(sizes))
            added_labels = set()

            for phase in ["load_s", "clean_s", "split_s", "train_s", "infer_s"]:
                vals = []
                for sz in sizes:
                    row = g_mean[(g_mean["language"] == lang) &
                                 (g_mean["model"] == model) &
                                 (g_mean["subset_size"] == sz)]
                    vals.append(float(row[phase].iloc[0])
                                if not row.empty and phase in row.columns else 0.0)
                vals = np.array(vals)
                lbl = seg_labels[phase] if phase not in added_labels else "_nolegend_"
                ax.bar(x, vals, bw, bottom=bottoms,
                       color=seg_colors[phase],
                       hatch=hatches[lang],
                       label=lbl, alpha=0.88,
                       edgecolor="#0f1117", linewidth=0.4)
                bottoms += vals
                added_labels.add(phase)

            # Pipeline marker (diamond)
            pipe_vals = []
            for sz in sizes:
                row = g_mean[(g_mean["language"] == lang) &
                             (g_mean["model"] == model) &
                             (g_mean["subset_size"] == sz)]
                if not row.empty and "pipeline_s" in row.columns:
                    pipe_vals.append(float(row["pipeline_s"].iloc[0]))
                else:
                    pipe_vals.append(np.nan)
            ax.plot(x, pipe_vals, marker="D", linestyle="none",
                    color="#f4e04d", markersize=7, zorder=5,
                    label="Pipeline (split+train+infer)")

            # Wall-clock marker (X) if available
            if has_wc:
                wc_vals = []
                for sz in sizes:
                    row = g_mean[(g_mean["language"] == lang) &
                                 (g_mean["model"] == model) &
                                 (g_mean["subset_size"] == sz)]
                    if not row.empty and "wall_clock_s" in row.columns:
                        wc_vals.append(float(row["wall_clock_s"].iloc[0]))
                    else:
                        wc_vals.append(np.nan)
                ax.plot(x, wc_vals, marker="x", linestyle="none",
                        color="#ff6b6b", markersize=9, markeredgewidth=2,
                        zorder=6, label="Wall-clock (true end-to-end)")

            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(s):,}" for s in sizes],
                               fontsize=7, rotation=20, ha="right")
            ax.set_xlabel("Subset size (rows)", fontsize=8)
            ax.set_ylabel("Time (s)", fontsize=8)
            _title(ax, f"{LANG_LABEL[lang]} · {model}")
            ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_phases_vs_wallclock.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cs",  default="StudyCsharp/StudyCsharp/bin/Debug/net10.0/results/csharp_timings.csv")
    ap.add_argument("--py",  default="results/python_timings.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve_csv(given_path, candidates):
        """Return the first existing path from given_path + fallback candidates."""
        for p in [given_path] + candidates:
            if p and os.path.exists(p):
                return p
        tried = "\n  ".join([given_path] + candidates)
        raise FileNotFoundError(
            f"CSV not found. Tried:\n  {tried}\n"
            "Pass the correct path with --cs / --py."
        )

    args.cs = resolve_csv(args.cs, [
        os.path.join(script_dir, "csharp_timings.csv"),
        os.path.join(script_dir, "results", "csharp_timings.csv"),
        "csharp_timings.csv",
    ])
    args.py = resolve_csv(args.py, [
        os.path.join(script_dir, "python_timings.csv"),
        os.path.join(script_dir, "results", "python_timings.csv"),
        "python_timings.csv",
    ])

    os.makedirs(args.out, exist_ok=True)

    print("Loading data...")
    df = load(args.cs, args.py)
    g_mean, g_std = grouped_stats(df)

    print(f"  C# rows:     {len(df[df['language'] == 'csharp'])}")
    print(f"  Python rows: {len(df[df['language'] == 'python'])}")
    print(f"  Sizes:       {sorted(df['subset_size'].unique())}")
    print(f"  Models:      {sorted(df['model'].unique())}")
    print(f"  Repeats:     {sorted(df['repeat'].unique())}")
    print()

    print("Generating plots...")
    fig_time_breakdown(g_mean, g_std, args.out)
    fig_phase_bars(g_mean, args.out)
    fig_memory(g_mean, g_std, args.out)
    fig_accuracy(g_mean, args.out)
    fig_speedup(g_mean, args.out)
    fig_variance(df, args.out)
    fig_wall_clock(g_mean, g_std, args.out)
    fig_all_phases(g_mean, g_std, args.out)
    fig_phases_vs_wallclock(g_mean, g_std, args.out)
    fig_summary_table(g_mean, args.out)

    n_plots = 7 + (1 if "wall_clock_s" in g_mean.columns else 0) + 2
    print(f"\nDone — {n_plots} plots saved to: {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()
