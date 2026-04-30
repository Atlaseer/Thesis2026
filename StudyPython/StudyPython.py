from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

# Import for measurements
import tracemalloc
import psutil
import os

# Imports for ML models
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Target column name.
TARGET = "compatibility_score"
# print("Numpy version is: ", np.__version__)

# Numeric feature columns I allow the model to use.
# I list them explicitly so I don’t accidentally include IDs/text columns or any extra fields.
BASE_FEATURES = [
    "skill_match_score",
    "skill_complementarity_score",
    "network_value_a_to_b",
    "network_value_b_to_a",
    "career_alignment_score",
    "experience_gap",
    "industry_match",
    "geographic_score",
    "seniority_match",
]


def now_ns() -> int:
    """High-resolution monotonic time in nanoseconds (good for timing phases)."""
    return perf_counter_ns()


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def _rss() -> int:
    """OS physical RAM for this process (bytes). Best cross-language metric:
    captures both Python heap and NumPy/C-extension native buffers."""
    return psutil.Process(os.getpid()).memory_info().rss


def _measure(fn):
    """
    Run fn() while recording:
      - elapsed wall-clock nanoseconds
      - tracemalloc heap peak (Python-managed allocations only;
        undercounts NumPy/C buffers — use rss_delta for the full picture)
      - RSS delta (OS physical RAM change)
    Returns (result, elapsed_ns, heap_peak_bytes, rss_delta_bytes).
    """
    rss_before = _rss()
    tracemalloc.start()
    t0 = now_ns()

    result = fn()

    elapsed_ns = now_ns() - t0
    _, heap_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_delta = _rss() - rss_before

    return result, elapsed_ns, heap_peak, rss_delta


def detect_constant_numeric_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns and df[c].nunique(dropna=False) <= 1]


"""
    Find numeric columns with no variance (nunique <= 1).

    I drop these because they carry no signal and can create noise or weird edge cases
    in other toolchains. (In my dataset check, skill_complementarity_score was constant.)
    """


def stratified_subsample_by_target(
    df: pd.DataFrame,
    n: int,
    target: str,
    *,
    n_bins: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Subsample n rows while roughly preserving the target distribution.

    I use this for scaling tests so that smaller subsets still “look like” the full dataset.
    It’s deterministic (fixed seed) so I can rerun and get the same subset every time.
    """
    if n >= len(df):
        return df.reset_index(drop=True)

    # Bin the target into quantiles so each bin has roughly equal mass.
    bins = pd.qcut(df[target], q=n_bins, duplicates="drop")

    # Allocate sample counts per bin proportional to bin sizes.
    counts = bins.value_counts(sort=False)
    props = counts / counts.sum()
    alloc = (props * n).round().astype(int)

    # Fix rounding drift so I return exactly n rows (still deterministic).
    drift = int(n - alloc.sum())
    if drift != 0:
        order = props.sort_values(ascending=False).index.tolist()
        i = 0
        while drift != 0 and i < len(order) * 10:
            b = order[i % len(order)]
            if drift > 0:
                alloc.loc[b] += 1
                drift -= 1
            elif alloc.loc[b] > 0:
                alloc.loc[b] -= 1
                drift += 1
            i += 1

    # Sample within each bin.
    parts: list[pd.DataFrame] = []
    for b, k in alloc.items():
        k = int(k)
        if k <= 0:
            continue
        bin_df = df.loc[bins == b]
        k_eff = min(k, len(bin_df))
        if k_eff <= 0:
            continue
        parts.append(bin_df.sample(n=k_eff, random_state=seed))

    out = pd.concat(parts, axis=0)

    # If I’m short due to bin constraints, top up from remaining rows.
    if len(out) < n:
        rest = df.drop(index=out.index)
        out = pd.concat(
            [out, rest.sample(n=n - len(out), random_state=seed)])

    # Shuffle deterministically so concatenation order doesn’t matter.
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_clean_and_select_features(
    csv_path: Path,
    *,
    derive_network_asymmetry: bool,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Returns (clean_df, features, phase_metrics).
    phase_metrics contains timing and memory for the load and clean phases.
    """
    # --- Load phase ---
    (df,), load_ns, load_heap_peak, load_rss_delta = _measure(
        lambda: (pd.read_csv(csv_path),)
    )

    # --- Clean phase ---
    def _clean():
        nonlocal df
        df = df.drop_duplicates(ignore_index=True)

        needed = set(BASE_FEATURES + [TARGET])
        missing = sorted(c for c in needed if c not in df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        model_cols = BASE_FEATURES + [TARGET]
        df_num = df[model_cols].apply(pd.to_numeric, errors="coerce")
        df_ = df.loc[df_num.notna().all(axis=1)].copy()
        df_[model_cols] = df_num.loc[df_.index]

        features = BASE_FEATURES.copy()
        if derive_network_asymmetry:
            df_["network_asymmetry"] = (
                df_["network_value_a_to_b"] - df_["network_value_b_to_a"]
            ).astype(float)
            features.append("network_asymmetry")

        const = detect_constant_numeric_cols(df_, features)
        features = [c for c in features if c not in const]
        return df_.reset_index(drop=True), features

    (clean_df, features), clean_ns, clean_heap_peak, clean_rss_delta = _measure(_clean)

    phase_metrics = {
        "load_ns":              float(load_ns),
        "load_heap_peak_bytes": float(load_heap_peak),
        "load_rss_delta_bytes": float(load_rss_delta),
        "clean_ns":             float(clean_ns),
        "clean_heap_peak_bytes": float(clean_heap_peak),
        "clean_rss_delta_bytes": float(clean_rss_delta),
    }
    return clean_df, features, phase_metrics


# ---------------------------------------------------------------------------
# Split → Train → Infer (timed per repeat)
# ---------------------------------------------------------------------------

def split_train_infer(
    sub: pd.DataFrame,
    features: list[str],
    *,
    test_size: float,
    seed: int,
    model_type: str,
) -> dict:
    """
    Measures time and memory for three phases:

    Heap (tracemalloc): Python-managed allocations only.
      For scikit-learn this undercounts because NumPy arrays and C extensions
      allocate outside Python's heap. Treat as a language-runtime overhead indicator.

    RSS delta (psutil): OS physical RAM change.
      Captures Python heap + NumPy buffers. Best cross-language comparison metric.
    """

    # --- Split phase ---
    def _split():
        X = sub[features].to_numpy(dtype=np.float32, copy=False)
        y = sub[TARGET].to_numpy(dtype=np.float32, copy=False)
        return train_test_split(X, y, test_size=test_size, random_state=seed)

    (X_train, X_test, y_train, _), split_ns, split_heap, split_rss = _measure(_split)

    # --- Train phase ---
    if model_type == "linear":
        mdl = LinearRegression()
    elif model_type == "tree":
        mdl = DecisionTreeRegressor(random_state=seed)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    _, train_ns, train_heap, train_rss = _measure(
        lambda: mdl.fit(X_train, y_train))

    # --- Infer phase ---
    _, infer_ns, infer_heap, infer_rss = _measure(lambda: mdl.predict(X_test))

    return {
        "split_ns":             float(split_ns),
        "split_heap_peak_bytes": float(split_heap),
        "split_rss_delta_bytes": float(split_rss),
        "train_ns":             float(train_ns),
        "train_heap_peak_bytes": float(train_heap),
        "train_rss_delta_bytes": float(train_rss),
        "infer_ns":             float(infer_ns),
        "infer_heap_peak_bytes": float(infer_heap),
        "infer_rss_delta_bytes": float(infer_rss),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Python regression timing + memory experiment."
    )
    ap.add_argument("--csv",  type=str, default="data/compatibility_pairs.csv")
    ap.add_argument("--out",  type=str, default="results/python_timings.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size",  type=float, default=0.2)
    ap.add_argument("--repeats",    type=int,   default=5)
    ap.add_argument("--warmup",     type=int,   default=1)
    ap.add_argument("--stratify-by-target",      action="store_true")
    ap.add_argument("--derive-network-asymmetry", action="store_true")
    ap.add_argument("--vary-split-per-repeat",    action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load + clean once — their cost is constant across repeats
    clean_df, features, preproc_metrics = load_clean_and_select_features(
        csv_path,
        derive_network_asymmetry=args.derive_network_asymmetry,
    )
    total_rows = len(clean_df)

    load_ns = preproc_metrics["load_ns"]
    clean_ns = preproc_metrics["clean_ns"]

    print(f"Rows after cleaning: {total_rows}, features: {features}")
    print(
        f"  load : {load_ns/1e6:.0f} ms | heap peak {preproc_metrics['load_heap_peak_bytes']/1024:.0f} KB | RSS Δ {preproc_metrics['load_rss_delta_bytes']/1024:.0f} KB")
    print(
        f"  clean: {clean_ns/1e6:.0f} ms | heap peak {preproc_metrics['clean_heap_peak_bytes']/1024:.0f} KB | RSS Δ {preproc_metrics['clean_rss_delta_bytes']/1024:.0f} KB")

    sizes = [
        # int(total_rows * 1.00),
        # int(total_rows * 0.75),
        # int(total_rows * 0.50),
        int(total_rows * 0.25),
        int(total_rows * 0.10),
    ]
    if not sizes or any(s <= 0 for s in sizes):
        raise ValueError("All subset sizes must be positive.")

    models = ["linear", "tree"]
    rows: list[dict] = []

    for n in sizes:
        n_eff = min(int(n), len(clean_df))

        if args.stratify_by_target:
            sub = stratified_subsample_by_target(
                clean_df, n_eff, TARGET, seed=args.seed)
        else:
            sub = clean_df.sample(
                n=n_eff, random_state=args.seed).reset_index(drop=True)

        for model in models:
            print(f"  subset={n_eff}  model={model}")

            # Warmup
            for w in range(args.warmup):
                split_train_infer(
                    sub, features,
                    test_size=args.test_size,
                    seed=args.seed + w if args.vary_split_per_repeat else args.seed,
                    model_type=model,
                )

            # Measured repeats
            for r in range(args.repeats):
                split_seed = args.seed + r if args.vary_split_per_repeat else args.seed

                t = split_train_infer(
                    sub, features,
                    test_size=args.test_size,
                    seed=split_seed,
                    model_type=model,
                )

                preprocess_ns = load_ns + clean_ns + t["split_ns"]
                total_ns = preprocess_ns + t["train_ns"] + t["infer_ns"]

                rows.append({
                    "language":  "python",
                    "library":   "scikit-learn",
                    "model":     model,
                    "subset_size": int(n_eff),
                    "repeat":    int(r),
                    "n_features": int(len(features)),
                    "seed":      int(args.seed),
                    "split_seed": int(split_seed),
                    "test_size": float(args.test_size),
                    # Load phase
                    "load_ns":              float(load_ns),
                    "load_heap_peak_bytes": float(preproc_metrics["load_heap_peak_bytes"]),
                    "load_rss_delta_bytes": float(preproc_metrics["load_rss_delta_bytes"]),
                    # Clean phase
                    "clean_ns":              float(clean_ns),
                    "clean_heap_peak_bytes": float(preproc_metrics["clean_heap_peak_bytes"]),
                    "clean_rss_delta_bytes": float(preproc_metrics["clean_rss_delta_bytes"]),
                    # Split phase
                    "split_ns":              float(t["split_ns"]),
                    "split_heap_peak_bytes": float(t["split_heap_peak_bytes"]),
                    "split_rss_delta_bytes": float(t["split_rss_delta_bytes"]),
                    # Aggregate
                    "preprocess_ns": float(preprocess_ns),
                    # Train phase
                    "train_ns":              float(t["train_ns"]),
                    "train_heap_peak_bytes": float(t["train_heap_peak_bytes"]),
                    "train_rss_delta_bytes": float(t["train_rss_delta_bytes"]),
                    # Infer phase
                    "infer_ns":              float(t["infer_ns"]),
                    "infer_heap_peak_bytes": float(t["infer_heap_peak_bytes"]),
                    "infer_rss_delta_bytes": float(t["infer_rss_delta_bytes"]),
                    # Total
                    "total_ns": float(total_ns),
                    # Flags
                    "stratify_by_target":      bool(args.stratify_by_target),
                    "derive_network_asymmetry": bool(args.derive_network_asymmetry),
                    "vary_split_per_repeat":    bool(args.vary_split_per_repeat),
                })

    pd.DataFrame(rows).to_csv(out_path, index=False)

    meta = {
        "csv":              str(csv_path.resolve()),
        "sizes_requested":  sizes,
        "repeats":          int(args.repeats),
        "warmup":           int(args.warmup),
        "seed":             int(args.seed),
        "test_size":        float(args.test_size),
        "stratify_by_target":       bool(args.stratify_by_target),
        "derive_network_asymmetry": bool(args.derive_network_asymmetry),
        "vary_split_per_repeat":    bool(args.vary_split_per_repeat),
        "target":                   TARGET,
        "base_features_requested":  BASE_FEATURES,
        "features_used":            features,
        "rows_after_cleaning":      int(total_rows),
        "notes": {
            "preprocess_ns":       "load_ns + clean_ns + split_ns",
            "total_ns":            "load_ns + clean_ns + split_ns + train_ns + infer_ns",
            "load_clean_once":     "Load and clean are timed once; their values are constant across all repeats.",
            "heap_peak_bytes":     "tracemalloc peak during the phase. Python-managed heap only — undercounts NumPy/C buffers.",
            "rss_delta_bytes":     "psutil RSS change during the phase. OS physical RAM. Best cross-language comparison metric.",
            "constant_features":   "Features with nunique<=1 are dropped (e.g. skill_complementarity_score).",
        },
    }

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path.resolve()}")
    print(f"Wrote: {meta_path.resolve()}")


if __name__ == "__main__":
    main()
