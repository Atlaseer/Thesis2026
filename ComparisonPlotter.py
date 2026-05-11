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
    import io as _io
    with open(cs_path, encoding="utf-8") as fh:
        raw_cs = fh.read()
    cs_lines = raw_cs.splitlines()
    cs_header = cs_lines[0].split(",")
    n_header = len(cs_header)
    n_data = len(cs_lines[1].split(",")) if len(cs_lines) > 1 else n_header
    if n_data == n_header + 1 and "wall_clock_ns" not in cs_header:
        insert_pos = cs_header.index("r2") if "r2" in cs_header else n_header
        cs_header.insert(insert_pos, "wall_clock_ns")
        cs_lines[0] = ",".join(cs_header)
        raw_cs = "\n".join(cs_lines)
    cs = pd.read_csv(_io.StringIO(raw_cs))
    py = pd.read_csv(py_path)
    df = pd.concat([cs, py], ignore_index=True)

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


def fig_all_phases(g_mean, g_std, out_dir):
    """
    Two separate figures:
      1. Load + Clean   — one-time I/O and preprocessing phases (flat lines expected)
      2. Split + Train + Infer — repeatable ML pipeline phases
    """
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


def fig_phases_vs_wallclock(g_mean, g_std, out_dir):
    """
    For each language × model combination, shows a stacked bar of the five
    measured phases (load + clean + split + train + infer) next to a marker
    for pipeline_s (split+train+infer) and — if available — wall_clock_s.
    """
    has_wc = "wall_clock_s" in g_mean.columns
    sizes = sorted(g_mean["subset_size"].unique())
    langs = ["csharp", "python"]
    models = ["linear", "tree"]

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


# ── NEW: Figure RQ1 — Normalised phase contribution (% of wall-clock) ────────

def fig_normalized_phase_contribution(g_mean, out_dir):
    """
    RQ1 — For each ecosystem × model × dataset size, shows each phase as a
    percentage of total wall-clock time.  Makes immediately visible how the
    composition of time changes between C# and Python and across dataset sizes.

    Layout: 2 rows (C# / Python) × 2 columns (linear / tree).
    Each bar sums to 100 %.
    """
    has_wc = "wall_clock_s" in g_mean.columns
    phases = ["load_s", "clean_s", "split_s", "train_s", "infer_s"]
    phase_labels = ["Load", "Clean", "Split", "Train", "Infer"]
    phase_colors = ["#2e4a6b", "#1e6b5a", "#8b4513", "#4fc3f7", "#a8d8ea"]
    sizes = sorted(g_mean["subset_size"].unique())
    x = np.arange(len(sizes))
    bw = 0.6

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        "Phase Composition — % of Total Time per Ecosystem & Model  (RQ1)\n"
        "Each bar sums to 100 % of wall-clock (or phase-sum if wall-clock unavailable)",
        fontsize=12, fontweight="bold", color="#e8ecf5", y=1.02)

    for row_idx, lang in enumerate(["csharp", "python"]):
        for col_idx, model in enumerate(["linear", "tree"]):
            ax = axes[row_idx][col_idx]
            bottoms = np.zeros(len(sizes))

            # Compute denominator: wall-clock if available, else sum of all phases
            denoms = []
            for sz in sizes:
                row = g_mean[(g_mean["language"] == lang) &
                             (g_mean["model"] == model) &
                             (g_mean["subset_size"] == sz)]
                if row.empty:
                    denoms.append(np.nan)
                    continue
                r = row.iloc[0]
                if has_wc and not np.isnan(r.get("wall_clock_s", np.nan)):
                    denoms.append(float(r["wall_clock_s"]))
                else:
                    denoms.append(sum(
                        float(r[p]) for p in phases if p in r.index
                        and not np.isnan(r[p])))
            denoms = np.array(denoms)

            for phase, label, color in zip(phases, phase_labels, phase_colors):
                vals = []
                for sz in sizes:
                    row = g_mean[(g_mean["language"] == lang) &
                                 (g_mean["model"] == model) &
                                 (g_mean["subset_size"] == sz)]
                    if row.empty or phase not in row.columns:
                        vals.append(0.0)
                        continue
                    v = float(row[phase].iloc[0])
                    vals.append(v)
                vals = np.array(vals)
                pct = np.where(denoms > 0, vals / denoms * 100, 0.0)
                ax.bar(x, pct, bw, bottom=bottoms,
                       color=color, label=label, alpha=0.88,
                       edgecolor="#0f1117", linewidth=0.4)
                # Annotate segments > 5 %
                for xi, (p, b) in enumerate(zip(pct, bottoms)):
                    if p > 5:
                        ax.text(xi, b + p / 2, f"{p:.0f}%",
                                ha="center", va="center",
                                fontsize=6.5, color="#e8ecf5", fontweight="bold")
                bottoms += pct

            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(s):,}" for s in sizes], fontsize=7,
                               rotation=20, ha="right")
            ax.set_xlabel("Subset size (rows)", fontsize=8)
            ax.set_ylabel("% of total time", fontsize=8)
            ax.set_ylim(0, 110)
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
            _title(ax, f"{LANG_LABEL[lang]} · {model}")
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7.5, loc="upper right",
                          bbox_to_anchor=(1.0, 0.98))

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_normalized_phases.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── NEW: Figure RQ2 — Waterfall of signed phase differences ──────────────────

def fig_waterfall_phase_diff(g_mean, out_dir):
    """
    RQ2 — For each model × dataset size, shows the signed difference
    (C# time − Python time) per phase as horizontal bars.
    Bars right of zero = C# slower for that phase.
    Bars left  of zero = C# faster for that phase.

    Layout: one figure per model (linear / tree), subset sizes on y-axis,
    phases as grouped bars within each size.
    """
    phases = ["load_s", "clean_s", "split_s", "train_s", "infer_s"]
    phase_labels = ["Load", "Clean", "Split", "Train", "Infer"]
    # Distinct colors per phase so each bar is identifiable
    phase_colors = ["#4fc3f7", "#1e9e82", "#f4e04d", "#f4845f", "#c1440e"]

    sizes = sorted(g_mean["subset_size"].unique())
    cs_data = g_mean[g_mean["language"] == "csharp"]
    py_data = g_mean[g_mean["language"] == "python"]

    if cs_data.empty or py_data.empty:
        print("  Skipped fig_waterfall_phase_diff: need both languages")
        return

    for model in ["linear", "tree"]:
        n_phases = len(phases)
        n_sizes = len(sizes)
        fig, ax = plt.subplots(figsize=(12, max(6, n_sizes * 1.4)))
        fig.patch.set_facecolor("#0f1117")

        # Build y positions: group phases within each size block
        group_height = n_phases + 1.0   # gap between size groups
        yticks, yticklabels = [], []

        for si, sz in enumerate(reversed(sizes)):   # largest size at top
            base_y = si * group_height
            for pi, (phase, label, color) in enumerate(
                    zip(phases, phase_labels, phase_colors)):
                cs_row = cs_data[(cs_data["model"] == model) &
                                 (cs_data["subset_size"] == sz)]
                py_row = py_data[(py_data["model"] == model) &
                                 (py_data["subset_size"] == sz)]
                if cs_row.empty or py_row.empty:
                    continue
                if phase not in cs_row.columns or phase not in py_row.columns:
                    continue
                diff = float(cs_row[phase].iloc[0]) - \
                    float(py_row[phase].iloc[0])
                y_pos = base_y + pi * 0.9
                bar_color = "#ff6b6b" if diff > 0 else "#51cf66"
                ax.barh(y_pos, diff, 0.75,
                        color=bar_color, alpha=0.82,
                        edgecolor="#0f1117", linewidth=0.4)
                # Phase label on bar
                ax.text(diff + (0.002 if diff >= 0 else -0.002),
                        y_pos, label,
                        va="center",
                        ha="left" if diff >= 0 else "right",
                        fontsize=6.5, color="#c8cdd8")

            # Size label on y-axis at group midpoint
            mid_y = base_y + (n_phases - 1) * 0.9 / 2
            yticks.append(mid_y)
            yticklabels.append(f"{int(sz):,}")

        ax.axvline(0, color="#ffffff", linewidth=1.0, alpha=0.5)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=8)
        ax.set_xlabel("C# time − Python time  (s)\n"
                      "← C# faster  |  C# slower →", fontsize=8)

        # Format x-axis in ms or s
        def _xfmt(v, _):
            if abs(v) < 1.0:
                return f"{v*1000:.0f} ms"
            return f"{v:.1f} s"
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_xfmt))

        ax.set_title(
            f"Phase-Level Time Difference: C# − Python  (model: {model})  (RQ2)\n"
            "Red = C# slower  |  Green = C# faster",
            fontsize=10, fontweight="bold", color="#e8ecf5", pad=10)
        ax.set_facecolor("#181b24")
        fig.tight_layout()
        path = os.path.join(out_dir, f"comparison_waterfall_{model}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
        plt.close(fig)
        print(f"  Saved: {path}")


# ── NEW: Figure RQ3 — Log-log scaling plot ───────────────────────────────────

def fig_loglog_scaling(g_mean, out_dir):
    """
    RQ3 — Log-log plot of execution time vs dataset size for train and infer.
    On a log-log axis the slope reveals the scaling exponent:
      slope ≈ 1.0  →  O(n)   linear scaling
      slope ≈ 1.5  →  O(n^1.5)
      slope ≈ 2.0  →  O(n²)  quadratic
    A reference O(n) line is drawn for comparison.

    Annotates the fitted slope for each series so the reader can immediately
    compare scaling behaviour between ecosystems and algorithms.
    """
    from numpy.polynomial.polynomial import polyfit as nppolyfit

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        "Log-Log Scaling Plot — Execution Time vs Dataset Size  (RQ3)\n"
        "Slope on log-log axes ≈ scaling exponent  (slope 1.0 = linear O(n))",
        fontsize=12, fontweight="bold", color="#e8ecf5", y=1.02)

    for ax, (col, title) in zip(axes, [
        ("train_s", "Train time"),
        ("infer_s", "Inference time"),
    ]):
        # Reference O(n) line anchored to smallest C# linear value
        ref_sizes = sorted(g_mean["subset_size"].unique())
        ref_row = g_mean[(g_mean["language"] == "csharp") &
                         (g_mean["model"] == "linear") &
                         (g_mean["subset_size"] == ref_sizes[0])]
        if not ref_row.empty and col in ref_row.columns:
            ref_y0 = float(ref_row[col].iloc[0])
            ref_x0 = ref_sizes[0]
            ref_ys = [ref_y0 * (sz / ref_x0) for sz in ref_sizes]
            ax.plot(ref_sizes, ref_ys,
                    color="#ffffff", linewidth=0.8, linestyle=":",
                    alpha=0.35, label="O(n) reference", zorder=1)

        for lang in ("csharp", "python"):
            for model in ("linear", "tree"):
                sub = g_mean[(g_mean["language"] == lang) &
                             (g_mean["model"] == model)]
                if sub.empty or col not in sub.columns:
                    continue
                x_vals = sub["subset_size"].values.astype(float)
                y_vals = sub[col].values.astype(float)
                # Remove non-positive values before log
                mask = (x_vals > 0) & (y_vals > 0)
                if mask.sum() < 2:
                    continue
                xm, ym = x_vals[mask], y_vals[mask]

                ax.plot(xm, ym,
                        color=COLORS[(lang, model)],
                        marker=MARKERS[model],
                        linestyle=LINESTYLE[model],
                        linewidth=1.8, markersize=6, alpha=0.92,
                        label=f"{LANG_LABEL[lang]} · {model}",
                        zorder=3)

                # Fit log-log slope
                log_x = np.log10(xm)
                log_y = np.log10(ym)
                try:
                    coeffs = np.polyfit(log_x, log_y, 1)
                    slope = coeffs[0]
                    # Annotate slope near last point
                    ax.annotate(
                        f"  slope={slope:.2f}",
                        xy=(xm[-1], ym[-1]),
                        fontsize=6.5,
                        color=COLORS[(lang, model)],
                        va="center", alpha=0.85)
                except Exception:
                    pass

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _:
                                 f"{v*1000:.1f} ms" if v < 1 else f"{v:.2f} s"))
        ax.set_xlabel("Subset size (rows)  [log scale]", fontsize=8)
        ax.set_ylabel("Time  [log scale]", fontsize=8)
        _title(ax, title)
        _subtitle(ax, "Parallel lines = same exponent; steeper = worse scaling")
        _legend(ax, loc="upper left")

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_loglog_scaling.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── NEW: Figure RQ4 — Grouped ratio bars (algorithm × phase) ─────────────────

def fig_algorithm_ratio_bars(g_mean, out_dir):
    """
    RQ4 — For each dataset size, shows the C#/Python speed ratio as grouped
    bars where the grouping is:  algorithm (linear / tree) × phase (train / infer).

    This puts linear and tree directly side by side so the reader can immediately
    see whether the performance advantage is consistent across algorithm families
    or depends on the algorithm.

    One figure per dataset size (to avoid overcrowding), plus a summary figure
    showing all sizes as a multi-panel grid.
    """
    sizes = sorted(g_mean["subset_size"].unique())
    phases = [("train_s", "Train"), ("infer_s", "Infer"),
              ("pipeline_s", "Pipeline")]
    models = ["linear", "tree"]
    cs_data = g_mean[g_mean["language"] == "csharp"]
    py_data = g_mean[g_mean["language"] == "python"]

    if cs_data.empty or py_data.empty:
        print("  Skipped fig_algorithm_ratio_bars: need both languages")
        return

    n_sizes = len(sizes)
    n_cols = min(3, n_sizes)
    n_rows = int(np.ceil(n_sizes / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 5 * n_rows),
                             sharey=False)
    if n_sizes == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        "C# ÷ Python Speed Ratio by Algorithm & Phase  (RQ4)\n"
        ">1 = C# slower  |  <1 = C# faster  |  Grouped by phase, split by model",
        fontsize=12, fontweight="bold", color="#e8ecf5", y=1.02)

    # Bar positions: for each phase group, two bars side by side (linear / tree)
    n_phases = len(phases)
    group_width = 1.0
    bar_width = 0.35
    group_centers = np.arange(n_phases) * (group_width + 0.4)

    for si, sz in enumerate(sizes):
        row_idx = si // n_cols
        col_idx = si % n_cols
        ax = axes[row_idx][col_idx]

        for mi, model in enumerate(models):
            offset = (mi - 0.5) * bar_width
            ratios = []
            for col, _ in phases:
                cs_row = cs_data[(cs_data["model"] == model) &
                                 (cs_data["subset_size"] == sz)]
                py_row = py_data[(py_data["model"] == model) &
                                 (py_data["subset_size"] == sz)]
                if cs_row.empty or py_row.empty or col not in cs_row.columns:
                    ratios.append(np.nan)
                    continue
                cv = float(cs_row[col].iloc[0])
                pv = float(py_row[col].iloc[0])
                ratios.append(cv / pv if pv > 0 else np.nan)

            bar_colors = [
                "#ff6b6b" if (not np.isnan(r) and r > 1) else "#51cf66"
                for r in ratios
            ]
            bars = ax.bar(group_centers + offset, ratios, bar_width,
                          color=bar_colors,
                          label=f"{model}",
                          alpha=0.82, edgecolor="#0f1117", linewidth=0.4)

            # Value labels on bars
            for bar, ratio in zip(bars, ratios):
                if not np.isnan(ratio):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.05,
                            f"{ratio:.1f}×",
                            ha="center", va="bottom",
                            fontsize=6.5, color="#e8ecf5")

        ax.axhline(1.0, color="#ffffff", linewidth=0.9,
                   linestyle=":", alpha=0.5)
        ax.set_xticks(group_centers)
        ax.set_xticklabels([lbl for _, lbl in phases], fontsize=8)
        ax.set_ylabel("C# time / Python time", fontsize=8)
        ax.set_xlabel("Phase", fontsize=8)
        _title(ax, f"Subset size: {int(sz):,} rows")
        _subtitle(ax, "Red = C# slower  |  Green = C# faster")

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor=COLORS[("csharp", m)], label=m) for m in models
        ], fontsize=7.5, loc="upper right")

    # Hide unused subplots
    for si in range(n_sizes, n_rows * n_cols):
        axes[si // n_cols][si % n_cols].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_algorithm_ratio_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── NEW: Figure RQ4 — Interaction plot (algorithm × ecosystem) ───────────────

def fig_interaction_plot(g_mean, out_dir):
    """
    RQ4 — Classic interaction plot.
    X-axis: algorithm family (linear / tree).
    Y-axis: C#/Python speed ratio.
    Separate lines for each dataset size.

    Crossing lines are the canonical visual for an interaction effect —
    they show that the performance advantage depends on the algorithm,
    not just the ecosystem.  One panel per phase (train / infer / pipeline).
    """
    phases = [("train_s", "Train"), ("infer_s", "Infer"),
              ("pipeline_s", "Pipeline")]
    sizes = sorted(g_mean["subset_size"].unique())
    cs_data = g_mean[g_mean["language"] == "csharp"]
    py_data = g_mean[g_mean["language"] == "python"]

    if cs_data.empty or py_data.empty:
        print("  Skipped fig_interaction_plot: need both languages")
        return

    cmap = plt.cm.get_cmap("plasma", len(sizes))
    x_positions = [0, 1]
    x_labels = ["linear", "tree"]

    fig, axes = plt.subplots(1, len(phases), figsize=(5 * len(phases), 6))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        "Interaction Plot — C#/Python Ratio across Algorithm Families  (RQ4)\n"
        "Crossing lines = algorithm-dependent advantage  |  Each line = one dataset size",
        fontsize=11, fontweight="bold", color="#e8ecf5", y=1.02)

    for ax, (col, phase_label) in zip(axes, phases):
        for si, sz in enumerate(sizes):
            ratios = []
            for model in ["linear", "tree"]:
                cs_row = cs_data[(cs_data["model"] == model) &
                                 (cs_data["subset_size"] == sz)]
                py_row = py_data[(py_data["model"] == model) &
                                 (py_data["subset_size"] == sz)]
                if cs_row.empty or py_row.empty or col not in cs_row.columns:
                    ratios.append(np.nan)
                    continue
                cv = float(cs_row[col].iloc[0])
                pv = float(py_row[col].iloc[0])
                ratios.append(cv / pv if pv > 0 else np.nan)

            color = cmap(si)
            ax.plot(x_positions, ratios,
                    marker="o", linewidth=1.8, markersize=7,
                    color=color,
                    label=f"{int(sz):,} rows", alpha=0.88)
            # Label the end point
            if not np.isnan(ratios[-1]):
                ax.annotate(f"{int(sz/1e6):.1f}M",
                            xy=(1, ratios[-1]),
                            xytext=(1.05, ratios[-1]),
                            fontsize=6, color=color, va="center")

        ax.axhline(1.0, color="#ffffff", linewidth=0.9,
                   linestyle=":", alpha=0.5, label="parity")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=9, fontweight="bold")
        ax.set_xlabel("Algorithm family", fontsize=8)
        ax.set_ylabel("C# time / Python time\n(>1 = C# slower)", fontsize=8)
        _title(ax, f"{phase_label} — ratio by algorithm")
        _subtitle(
            ax, "Crossing lines → interaction effect (advantage depends on algorithm)")
        _legend(ax, loc="upper left")

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_interaction_plot.png")
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

    print("Generating original plots...")
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

    print("\nGenerating new RQ-targeted plots...")
    # RQ1 — How does time composition differ across ecosystems?
    fig_normalized_phase_contribution(g_mean, args.out)

    # RQ2 — Which phases drive the differences? (signed waterfall)
    fig_waterfall_phase_diff(g_mean, args.out)

    # RQ3 — How does scaling behaviour differ? (log-log)
    fig_loglog_scaling(g_mean, args.out)

    # RQ4 — Is the advantage consistent across algorithm families?
    fig_algorithm_ratio_bars(g_mean, args.out)
    fig_interaction_plot(g_mean, args.out)

    n_original = 10
    n_new = 5  # normalized + 2×waterfall + loglog + ratio_bars + interaction
    print(f"\nDone — {n_original} original + {n_new} new RQ-targeted plots "
          f"saved to: {os.path.abspath(args.out)}/")
    print("\nNew plots summary:")
    print("  comparison_normalized_phases.png    → RQ1: phase % of total time")
    print("  comparison_waterfall_linear.png     → RQ2: signed phase diff (linear)")
    print("  comparison_waterfall_tree.png       → RQ2: signed phase diff (tree)")
    print("  comparison_loglog_scaling.png       → RQ3: scaling exponents")
    print("  comparison_algorithm_ratio_bars.png → RQ4: ratio by algorithm & phase")
    print("  comparison_interaction_plot.png     → RQ4: interaction effect")


if __name__ == "__main__":
    main()
