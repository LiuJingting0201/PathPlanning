import os
import osmnx as ox
import random
import matplotlib.pyplot as plt
from astar_module import astar
from dijkstra_module import dijkstra
import csv
from math import sqrt
import time

# ------------------------
# city file
# ------------------------
cities = [
    "Turin, Italy",
    "Paris, France",
    "Beijing, China",
    
]

results = []

# ------------------------
# name
# ------------------------
def get_unique_path(base, ext=".png"):
    i = 1
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"

def get_unique_csv_path():
    i = 1
    while os.path.exists(f"results/results_{i}.csv"):
        i += 1
    return f"results/results_{i}.csv"

# ------------------------
# calculate the distance
# ------------------------
def euclidean_distance(y1, x1, y2, x2):
    return sqrt((x2 - x1)**2 + (y2 - y1)**2) * 111000  # 单位：米（大致）

# ------------------------
# reconstract path and optimize
# ------------------------
def reconstruct_path(G, start, end):
    path = []
    curr = end
    while curr != start:
        prev = G.nodes[curr]["previous"]
        path.append((prev, curr))
        curr = prev
    return path[::-1]

def compute_path_length(G, path_edges):
    total_length = 0
    for u, v in path_edges:
        if G.has_edge(u, v, 0):
            total_length += G.edges[(u, v, 0)].get("length", 0)
    return total_length / 1000  # km

def plot_path(G, path_edges, city, algo):
    for u, v in path_edges:
        if G.has_edge(u, v, 0):
            G.edges[(u, v, 0)]["color"] = "red"
            G.edges[(u, v, 0)]["linewidth"] = 2
    base_name = os.path.join("results", f"{city.replace(', ', '_')}_{algo}")
    filepath = get_unique_path(base_name)
    ox.plot_graph(
        G,
        edge_color=[G.edges[e].get("color", "#cccccc") for e in G.edges],
        edge_linewidth=[G.edges[e].get("linewidth", 0.5) for e in G.edges],
        node_size=0,
        bgcolor="white",
        save=True,
        show=False,
        filepath=filepath,
        dpi=300
    )

# ------------------------
# results file with memo
# ------------------------
os.makedirs("results", exist_ok=True)

# ------------------------
# main. for every city
# ------------------------
for city in cities:
    print(f"\nRunning on {city}")
    G = ox.graph_from_place(city, network_type="drive", simplify=True)

    # adding maxspeed and weight
    for u, v, data in G.edges(keys=False, data=True):
        speed = data.get("maxspeed", 50)
        if isinstance(speed, list):
            speed = min([int(s) for s in speed if str(s).isdigit()])
        elif isinstance(speed, str):
            digits = ''.join(filter(str.isdigit, speed))
            speed = int(digits) if digits else 50
        else:
            speed = 50
        data["maxspeed"] = speed
        data["weight"] = data["length"] / (speed * 1000 / 3600)

    nodes = [n for n in G.nodes if G.out_degree(n) > 0]

    # sampling org. and end. with reasonable distance
    while True:
        start, end = random.sample(nodes, 2)
        dist = euclidean_distance(
            G.nodes[start]["y"], G.nodes[start]["x"],
            G.nodes[end]["y"], G.nodes[end]["x"]
        )
        if dist > 2000:
            break

    # Dijkstra
    start_time = time.time()
    d_it, d_vis = dijkstra(G, start, end)
    d_time = time.time() - start_time
    d_path = reconstruct_path(G, start, end)
    d_len = compute_path_length(G, d_path)
    plot_path(G, d_path, city, "Dijkstra")

    # Reset edge styles
    for u, v, data in G.edges(keys=False, data=True):
        data["color"] = "#cccccc"
        data["linewidth"] = 0.5

    # A*
    start_time = time.time()
    a_it, a_vis = astar(G, start, end)
    a_time = time.time() - start_time
    a_path = reconstruct_path(G, start, end)
    a_len = compute_path_length(G, a_path)
    plot_path(G, a_path, city, "Astar")

    # Calculate improvements
    def ratio(val1, val2):
        return round((1 - val1 / val2) * 100, 2) if val2 != 0 else 0.0

    results.append((city, "A*", a_it, a_vis, len(a_path), round(a_len, 2), round(a_time, 4), start, end, "",
                    ratio(a_it, d_it), ratio(a_vis, d_vis), ratio(a_time, d_time)))
    results.append((city, "Dijkstra", d_it, d_vis, len(d_path), round(d_len, 2), round(d_time, 4), start, end, "",
                    0.0, 0.0, 0.0))

# ------------------------
# save CSV 
# ------------------------
csv_path = get_unique_csv_path()
with open(csv_path, mode="w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "City", "Algorithm", "Iterations", "Visited Nodes", "Path Steps", "Distance (km)", "Time (s)",
        "Start Node", "End Node", "", "Iter Improvement (%)", "Node Improvement (%)", "Time Improvement (%)"
    ])
    writer.writerows(results)

print(f"\n✅ Results saved to '{csv_path}'")