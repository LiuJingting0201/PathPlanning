import pandas as pd
import matplotlib.pyplot as plt
import os

# ------------------------
# get CSV data
# ------------------------
import glob

# all the results
csv_files = sorted(glob.glob("results/results_*.csv"))
if not csv_files:
    raise FileNotFoundError("⚠️ RUN experiment_runner.py")

# pick the latest one
latest_csv = csv_files[-1]
print(f"📊 Loading data from: {latest_csv}")
df = pd.read_csv(latest_csv)


# select A* , Dijkstra, matching.
a_df = df[df["Algorithm"] == "A*"]
d_df = df[df["Algorithm"] == "Dijkstra"]

cities = df["City"].unique()


plt.style.use("ggplot")
colors = {"A*": "#ff758c", "Dijkstra": "#40a9ff"}


os.makedirs("visuals", exist_ok=True)

# ------------------------
# name
# ------------------------
def get_unique_path(base, ext=".png"):
    i = 1
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"

# ------------------------
# plot
# ------------------------
def plot_metric(metric, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(cities))
    ax.bar([i - 0.2 for i in x], d_df[metric], width=0.4, label="Dijkstra", color=colors["Dijkstra"])
    ax.bar([i + 0.2 for i in x], a_df[metric], width=0.4, label="A*", color=colors["A*"])

    # tags
    for i, val in enumerate(d_df[metric]):
        ax.text(i - 0.2, val + max(d_df[metric]) * 0.01, f"{val:.0f}", ha='center', va='bottom', fontsize=9)
    for i, val in enumerate(a_df[metric]):
        ax.text(i + 0.2, val + max(a_df[metric]) * 0.01, f"{val:.0f}", ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} Comparison Across Cities")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    base = os.path.join("visuals", f"{metric.lower().replace(' ', '_')}_comparison")
    save_path = get_unique_path(base, ".png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved: {save_path}")
    plt.close()


plot_metric("Iterations", "Iterations")
plot_metric("Visited Nodes", "Visited Nodes")
plot_metric("Distance (km)", "Path Length (km)")
plot_metric("Time (s)", "Execution Time (s)")
