from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants — must match Program.cs exactly
# ---------------------------------------------------------------------------

TARGET = "compatibility_score"

# Exactly 9 base features. No engineered features are added in either
# language so both models train on identical inputs.
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

# Dataset-size fractions — must match Program.cs
PERCENTAGES = [1.00, 0.75, 0.50, 0.25, 0.10]


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def now_ns() -> int:
    """High-resolution monotonic time in nanoseconds."""
    return perf_counter_ns()


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def detect_constant_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Return columns whose unique-value count (including NaN) is <= 1."""
    return [c for c in cols if c in df.columns and df[c].nunique(dropna=False) <= 1]


def load_clean_and_select_features(
    csv_path: Path,
) -> tuple[pd.DataFrame, list[str], int, int]:
    """
    Load the CSV and apply deterministic cleaning.

    Returns
    -------
    clean_df   : cleaned DataFrame (reset index)
    features   : list of feature column names actually used
    load_ns    : nanoseconds spent reading the CSV from disk
    clean_ns   : nanoseconds spent on all cleaning steps
    """
    # --- Load phase ---
    t0 = now_ns()
    df = pd.read_csv(csv_path)
    load_ns = now_ns() - t0

    # --- Clean phase ---
    t1 = now_ns()

    # 1. Drop exact duplicate rows (deterministic, keeps first occurrence).
    df = df.drop_duplicates(ignore_index=True)

    # 2. Validate schema — fail fast if the dataset changes.
    needed = set(BASE_FEATURES + [TARGET])
    missing = sorted(c for c in needed if c not in df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 3. Coerce model columns to float64; invalid strings become NaN.
    model_cols = BASE_FEATURES + [TARGET]
    df_num = df[model_cols].apply(pd.to_numeric, errors="coerce")

    # 4. Drop any row that has a NaN or Inf in a model column.
    valid_mask = df_num.notna().all(axis=1)
    for col in model_cols:
        valid_mask &= np.isfinite(df_num[col])
    df = df.loc[valid_mask].copy()
    df[model_cols] = df_num.loc[df.index]

    # 5. Drop constant features (no signal; matches C# behaviour).
    features = BASE_FEATURES.copy()
    const_cols = detect_constant_cols(df, features)
    features = [c for c in features if c not in const_cols]

    clean_ns = now_ns() - t1

    return df.reset_index(drop=True), features, load_ns, clean_ns


# ---------------------------------------------------------------------------
# Per-repeat timing: split → train → infer
# ---------------------------------------------------------------------------

def split_train_infer(
    sub: pd.DataFrame,
    features: list[str],
    *,
    test_size: float,
    seed: int,
    model_type: str,
) -> dict[str, int]:
    """
    Time split, train, and inference phases on an in-memory subset.

    All data is kept as float64 (double precision) to match C# double arrays.

    Returns dict with keys: split_ns, train_ns, infer_ns.
    """
    # --- Split phase ---
    t0 = now_ns()
    X = sub[features].to_numpy(dtype=np.float64, copy=False)
    y = sub[TARGET].to_numpy(dtype=np.float64, copy=False)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    split_ns = now_ns() - t0

    # --- Train phase ---
    if model_type == "linear":
        # OLS exact solver — matches C# OlsRegression (closed-form, no iterations).
        model = LinearRegression()
    elif model_type == "tree":
        # Single unpruned decision tree — matches C# FastForest(numberOfTrees:1).
        model = DecisionTreeRegressor(random_state=seed)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    t1 = now_ns()
    model.fit(X_train, y_train)
    train_ns = now_ns() - t1

    # --- Inference phase ---
    t2 = now_ns()
    model.predict(X_test)
    infer_ns = now_ns() - t2

    return {"split_ns": split_ns, "train_ns": train_ns, "infer_ns": infer_ns}


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Python regression timing experiment (phased, reproducible)."
    )
    ap.add_argument(
        "--csv", type=str, default="data/compatibility_pairs.csv",
        help="Path to compatibility_pairs.csv",
    )
    ap.add_argument(
        "--out", type=str, default="results/python_timings.csv",
        help="Output CSV path",
    )
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--repeats",   type=int,   default=100)
    ap.add_argument("--warmup",    type=int,   default=1)
    ap.add_argument(
        "--vary-split-per-repeat", action="store_true",
        help="If set, repeat r uses seed+r for the train/test split.",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and clean once — avoids mixing disk/cache effects with repeat timing.
    print("Loading and preprocessing data...")
    df_clean, features, load_ns, clean_ns = load_clean_and_select_features(csv_path)
    total_rows = len(df_clean)
    print(f"Preprocessing done. Rows after cleaning: {total_rows}, "
          f"features used: {features}")

    # Build subset sizes from the shared percentage list.
    sizes = [max(1, int(total_rows * p)) for p in PERCENTAGES]

    models = ["linear", "tree"]
    rows: list[dict] = []

    for n in sizes:
        n_eff = min(n, total_rows)

        # Simple random sample with fixed seed — identical logic to C# Random(seed)+shuffle.
        sub = df_clean.sample(n=n_eff, random_state=args.seed).reset_index(drop=True)

        for model_type in models:
            print(f"  subset={n_eff:>7d}  model={model_type}")

            # Warmup runs — not recorded.
            for w in range(args.warmup):
                split_train_infer(
                    sub, features,
                    test_size=args.test_size,
                    seed=args.seed + w if args.vary_split_per_repeat else args.seed,
                    model_type=model_type,
                )

            # Measured repeats.
            for r in range(args.repeats):
                split_seed = args.seed + r if args.vary_split_per_repeat else args.seed

                times = split_train_infer(
                    sub, features,
                    test_size=args.test_size,
                    seed=split_seed,
                    model_type=model_type,
                )

                # preprocess_ns and total_ns defined identically to Program.cs:
                #   preprocess_ns = load_ns + clean_ns + split_ns
                #   total_ns      = load_ns + clean_ns + split_ns + train_ns + infer_ns
                preprocess_ns = load_ns + clean_ns + times["split_ns"]
                total_ns = preprocess_ns + times["train_ns"] + times["infer_ns"]

                rows.append({
                    "language":              "python",
                    "library":               "scikit-learn",
                    "model":                 model_type,
                    "subset_size":           int(n_eff),
                    "repeat":                int(r),
                    "n_features":            int(len(features)),
                    "seed":                  int(args.seed),
                    "split_seed":            int(split_seed),
                    "test_size":             float(args.test_size),
                    "load_ns":               float(load_ns),
                    "clean_ns":              float(clean_ns),
                    "split_ns":              float(times["split_ns"]),
                    "preprocess_ns":         float(preprocess_ns),
                    "train_ns":              float(times["train_ns"]),
                    "infer_ns":              float(times["infer_ns"]),
                    "total_ns":              float(total_ns),
                    "vary_split_per_repeat": bool(args.vary_split_per_repeat),
                })

    pd.DataFrame(rows).to_csv(out_path, index=False)

    meta = {
        "csv":                    str(csv_path.resolve()),
        "percentages":            PERCENTAGES,
        "sizes_used":             sizes,
        "repeats":                int(args.repeats),
        "warmup":                 int(args.warmup),
        "seed":                   int(args.seed),
        "test_size":              float(args.test_size),
        "vary_split_per_repeat":  bool(args.vary_split_per_repeat),
        "target":                 TARGET,
        "base_features":          BASE_FEATURES,
        "features_used":          features,
        "rows_after_cleaning":    int(total_rows),
        "notes": {
            "linear_model":      "LinearRegression — OLS exact solver (no iterations). Matches C# OlsRegression.",
            "tree_model":        "DecisionTreeRegressor — single unpruned tree. Matches C# FastForest(numberOfTrees:1, numberOfLeaves:...).",
            "precision":         "All feature/target arrays are float64 (double). Matches C# double[].",
            "preprocess_ns":     "load_ns + clean_ns + split_ns",
            "total_ns":          "load_ns + clean_ns + split_ns + train_ns + infer_ns",
            "load_clean_once":   "CSV is loaded and cleaned once to avoid mixing disk/cache effects with repeat timing.",
            "no_feature_eng":    "No engineered features added. Both languages use exactly the same BASE_FEATURES columns.",
        },
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path.resolve()}")
    print(f"Wrote: {meta_path.resolve()}")


if __name__ == "__main__":
    main()
