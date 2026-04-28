import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv(
    "StudyCsharp/StudyCsharp/bin/Debug/net10.0/results/csharp_timings.csv")

# Convert ns → seconds
df["total_s"] = df["total_ns"] / 1e9
df["train_s"] = df["train_ns"] / 1e9
df["infer_s"] = df["infer_ns"] / 1e9

# Average over repeats
grouped = df.groupby(["subset_size", "model"]).mean(
    numeric_only=True).reset_index()

# -------- Plot 1: Total time --------
plt.figure()
for model in grouped["model"].unique():
    sub = grouped[grouped["model"] == model]
    plt.plot(sub["subset_size"], sub["total_s"], label=model)

plt.xlabel("Subset size")
plt.ylabel("Total time (s)")
plt.title("Total Time vs Dataset Size")
plt.legend()
plt.xscale("log")
plt.savefig("results/total_time.png")
plt.close()

# -------- Plot 2: Train time --------
plt.figure()
for model in grouped["model"].unique():
    sub = grouped[grouped["model"] == model]
    plt.plot(sub["subset_size"], sub["train_s"], label=model)

plt.xlabel("Subset size")
plt.ylabel("Train time (s)")
plt.title("Train Time vs Dataset Size")
plt.legend()
plt.xscale("log")
plt.savefig("results/train_time.png")
plt.close()

# -------- Plot 3: Inference time --------
plt.figure()
for model in grouped["model"].unique():
    sub = grouped[grouped["model"] == model]
    plt.plot(sub["subset_size"], sub["infer_s"], label=model)

plt.xlabel("Subset size")
plt.ylabel("Inference time (s)")
plt.title("Inference Time vs Dataset Size")
plt.legend()
plt.xscale("log")
plt.savefig("results/infer_time.png")
plt.close()
