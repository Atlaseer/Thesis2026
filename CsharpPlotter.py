import pandas as pd
import matplotlib.pyplot as plt
import os

# Load CSV and throw error if it isn't there
CSV_PATH = "StudyCsharp/StudyCsharp/bin/Debug/net10.0/results/csharp_timings.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"CSV not found at: {CSV_PATH}\n"
        "Run Program.cs first, then check the path matches where it wrote the file."
    )

df = pd.read_csv(CSV_PATH)

print(f"Loaded {len(df)} rows.")
print(
    f"Models: {df['model'].unique()}, Sizes: {sorted(df['subset_size'].unique())}")
# Convert ns -> seconds and bytes -> MB
df["total_s"] = df["total_ns"] / 1e9
df["train_s"] = df["train_ns"] / 1e9
df["infer_s"] = df["infer_ns"] / 1e9
df["split_s"] = df["split_ns"] / 1e9
df["train_rss_mb"] = df["train_rss_delta_bytes"] / 1024**2
df["infer_rss_mb"] = df["infer_rss_delta_bytes"] / 1024**2

# Average over repeats
grouped = (df.groupby(["subset_size", "model"])
             .mean(numeric_only=True)
             .reset_index()
             .sort_values("subset_size"))

os.makedirs("results", exist_ok=True)

MARKERS = {"linear": "o", "tree": "s"}

# -------- Plot 1: Total time --------
plt.figure()
for model in grouped["model"].unique():
    sub = grouped[grouped["model"] == model]
    plt.plot(sub["subset_size"], sub["total_s"],
             marker=MARKERS.get(model, "o"), label=model)
plt.xlabel("Subset size (rows)")
plt.ylabel("Total time (s)")
plt.title("C# -- Total Time vs Dataset Size")
plt.legend()
plt.xscale("log")
plt.tight_layout()
plt.savefig("results/csharp_total_time.png", dpi=150)
plt.close()

# -------- Plot 2: Train time --------
plt.figure()
for model in grouped["model"].unique():
    sub = grouped[grouped["model"] == model]
    plt.plot(sub["subset_size"], sub["train_s"],
             marker=MARKERS.get(model, "o"), label=model)
plt.xlabel("Subset size (rows)")
plt.ylabel("Train time (s)")
plt.title("C# -- Train Time vs Dataset Size")
plt.legend()
plt.xscale("log")
plt.tight_layout()
plt.savefig("results/csharp_train_time.png", dpi=150)
plt.close()

# -------- Plot 3: Inference time --------
plt.figure()
for model in grouped["model"].unique():
    sub = grouped[grouped["model"] == model]
    plt.plot(sub["subset_size"], sub["infer_s"],
             marker=MARKERS.get(model, "o"), label=model)
plt.xlabel("Subset size (rows)")
plt.ylabel("Inference time (s)")
plt.title("C# -- Inference Time vs Dataset Size")
plt.legend()
plt.xscale("log")
plt.tight_layout()
plt.savefig("results/csharp_infer_time.png", dpi=150)
plt.close()


# -------- Plot 4: Train RSS memory delta --------
plt.figure()
for model in grouped["model"].unique():
    sub = grouped[grouped["model"] == model]
    plt.plot(sub["subset_size"], sub["train_rss_mb"],
             marker=MARKERS.get(model, "o"), label=model)
plt.xlabel("Subset size (rows)")
plt.ylabel("RSS delta during training (MB)")
plt.title("C# -- Train Memory (RSS) vs Dataset Size")
plt.legend()
plt.xscale("log")
plt.tight_layout()
plt.savefig("results/csharp_train_memory.png", dpi=150)
plt.close()

print("\nSaved 4 plots to results/")
print("\nMean timings per model and subset size:")
print(df.groupby(["model", "subset_size"])[["train_s", "infer_s", "total_s"]]
        .mean().to_string())

print(df.columns)
print(df.head())
