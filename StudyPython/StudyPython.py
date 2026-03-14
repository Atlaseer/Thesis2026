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

# Target column name.
TARGET = "compatibility_score"

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


def now_ns() -> float:
    """High-resolution monotonic time in nanoseconds (good for timing phases)."""
    return perf_counter_ns()


def detect_constant_numeric_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """
    Find numeric columns with no variance (nunique <= 1).

    I drop these because they carry no signal and can create noise or weird edge cases
    in other toolchains. (In my dataset check, skill_complementarity_score was constant.)
    """
    const: list[str] = []
    for c in cols:
        if c not in df.columns:
            continue
        # Treat NaN as a value (even though I don’t expect missing values right now).
        if df[c].nunique(dropna=False) <= 1:
            const.append(c)
    return const


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
            else:
                if alloc.loc[b] > 0:
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
        remaining = n - len(out)
        rest = df.drop(index=out.index)
        out = pd.concat(
            [out, rest.sample(n=remaining, random_state=seed)], axis=0)

    # Shuffle deterministically so concatenation order doesn’t matter.
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_clean_and_select_features(
    csv_path: Path,
    *,
    derive_network_asymmetry: bool,
) -> tuple[pd.DataFrame, list[str], float, float]:
    """
    Load the CSV, apply deterministic cleaning, and return:
      (clean_df, features_used, load_ns, clean_ns)

    I time loading separately from cleaning so I can report phase timings later.
    """
    # Load timing: start before reading the CSV, stop right after it’s in memory.
    t0 = now_ns()
    df = pd.read_csv(csv_path)
    t1 = now_ns()
    load_ns = t1 - t0

    # Cleaning timing starts here.
    t2 = now_ns()

    # Drop exact duplicates (even if I don’t expect any, it keeps the pipeline explicit).
    df = df.drop_duplicates(ignore_index=True)

    # Fail fast if the schema changes.
    needed = set(BASE_FEATURES + [TARGET])
    missing = sorted([c for c in needed if c not in df.columns])
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available: {list(df.columns)}")

    # Only coerce model columns to numeric; anything invalid becomes NaN.
    model_cols = BASE_FEATURES + [TARGET]
    df_num = df[model_cols].apply(pd.to_numeric, errors="coerce")

    # Drop rows with any NaN in model columns (deterministic rule).
    df = df.loc[df_num.notna().all(axis=1)].copy()
    df[model_cols] = df_num.loc[df.index]

    # Start from the base feature list and optionally add derived features.
    features = BASE_FEATURES.copy()

    # Optional feature: network asymmetry = a_to_b - b_to_a
    if derive_network_asymmetry:
        df["network_asymmetry"] = (
            df["network_value_a_to_b"] - df["network_value_b_to_a"]).astype(float)
        features.append("network_asymmetry")

    # Drop constant features (no signal).
    const = detect_constant_numeric_cols(df, features)
    features = [c for c in features if c not in const]

    t3 = now_ns()
    clean_ns = t3 - t2

    return df.reset_index(drop=True), features, load_ns, clean_ns


def split_train_infer_times(
    sub: pd.DataFrame,
    features: list[str],
    *,
    test_size: float,
    seed: int,
) -> dict:
    """
    Time three phases on an in-memory subset:
      - split_ns: build X/y and do train/test split
      - train_ns: fit the regression model
      - infer_ns: predict on the test set
    """
    t0 = now_ns()

    # Materialize numeric arrays. float64 keeps types consistent.
    X = sub[features].to_numpy(dtype=np.float64, copy=False)
    y = sub[TARGET].to_numpy(dtype=np.float64, copy=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    t1 = now_ns()

    model = LinearRegression()
    model.fit(X_train, y_train)
    t2 = now_ns()

    _ = model.predict(X_test)
    t3 = now_ns()

    return {
        "split_ns": t1 - t0,
        "train_ns": t2 - t1,
        "infer_ns": t3 - t2,
    }


def main() -> None:
    """
    Run the timing experiment and write results to CSV + a meta JSON.

    I write one row per measured run so it’s easy to summarize later (mean/median/std)
    and compare directly against the C# implementation.
    """
    ap = argparse.ArgumentParser(
        description="Python linear regression timing experiment (phased, reproducible)."
    )

    ap.add_argument("--csv", type=str, default="data/compatibility_pairs.csv")
    ap.add_argument("--out", type=str, default="results/python_timings.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument(
        "--sizes",
        type=str,
        default="5000,10000,25000,50000",
        help="Comma-separated subset sizes",
    )
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=1)

    ap.add_argument("--stratify-by-target", action="store_true")
    ap.add_argument("--derive-network-asymmetry", action="store_true")

    ap.add_argument(
        "--vary-split-per-repeat",
        action="store_true",
        help="If set, repeat r uses seed+r for the train/test split. Otherwise all repeats share the same split.",
    )

    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    if not sizes or any(s <= 0 for s in sizes):
        raise ValueError(
            "Provide positive integers in --sizes (comma-separated).")

    # Load + clean once so I don’t mix scaling effects with repeated disk I/O/cache effects.
    df_clean, features, load_ns, clean_ns = load_clean_and_select_features(
        csv_path,
        derive_network_asymmetry=args.derive_network_asymmetry,
    )

    rows: list[dict] = []

    for n in sizes:
        n_eff = min(int(n), int(len(df_clean)))

        # Deterministic subset selection.
        if args.stratify_by_target:
            sub = stratified_subsample_by_target(
                df_clean, n_eff, TARGET, seed=args.seed)
        else:
            sub = df_clean.sample(
                n=n_eff, random_state=args.seed).reset_index(drop=True)

        # Warmup runs to reduce one-time allocation/caching noise (not recorded).
        for w in range(args.warmup):
            _ = split_train_infer_times(
                sub,
                features,
                test_size=args.test_size,
                seed=(args.seed + w) if args.vary_split_per_repeat else args.seed,
            )

        # Measured repeats.
        for r in range(args.repeats):
            split_seed = (
                args.seed + r) if args.vary_split_per_repeat else args.seed
            times = split_train_infer_times(
                sub, features, test_size=args.test_size, seed=split_seed)

            preprocess_ns = load_ns + clean_ns + times["split_ns"]
            total_ns = load_ns + clean_ns + \
                times["split_ns"] + times["train_ns"] + times["infer_ns"]

            rows.append(
                {
                    "language": "python",
                    "library": "scikit-learn",
                    "model": "LinearRegression",
                    "subset_size": int(n_eff),
                    "repeat": int(r),
                    "n_features": int(len(features)),
                    "seed": int(args.seed),
                    "split_seed": int(split_seed),
                    "test_size": float(args.test_size),
                    "load_ns": float(load_ns),
                    "clean_ns": float(clean_ns),
                    "split_ns": float(times["split_ns"]),
                    "preprocess_ns": float(preprocess_ns),
                    "train_ns": float(times["train_ns"]),
                    "infer_ns": float(times["infer_ns"]),
                    "total_ns": float(total_ns),
                    "stratify_by_target": bool(args.stratify_by_target),
                    "derive_network_asymmetry": bool(args.derive_network_asymmetry),
                    "vary_split_per_repeat": bool(args.vary_split_per_repeat),
                }
            )

    res = pd.DataFrame(rows)
    res.to_csv(out_path, index=False)

    meta = {
        "csv": str(csv_path.resolve()),
        "sizes_requested": sizes,
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "seed": int(args.seed),
        "test_size": float(args.test_size),
        "stratify_by_target": bool(args.stratify_by_target),
        "derive_network_asymmetry": bool(args.derive_network_asymmetry),
        "vary_split_per_repeat": bool(args.vary_split_per_repeat),
        "target": TARGET,
        "base_features_requested": BASE_FEATURES,
        "features_used": features,
        "rows_after_cleaning": int(len(df_clean)),
        "notes": {
            "constant_features_dropped": "I drop any numeric feature with nunique<=1 (e.g., skill_complementarity_score was constant in my checks).",
            "total_ns_definition": "total_ns = load_ns + clean_ns + split_ns + train_ns + infer_ns",
            "load_clean_timed_once": "I time loading/cleaning once per run to avoid mixing scaling results with disk/cache effects.",
        },
    }

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Wrote:", out_path.resolve())
    print("Wrote:", meta_path.resolve())


if __name__ == "__main__":
    main()
