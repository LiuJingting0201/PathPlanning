import os
import shutil
import osmnx as ox
import random
import matplotlib.pyplot as plt
from astar_module import astar
from dijkstra_module import dijkstra
import csv

# ------------------------
# 设置：城市列表 & 文件夹
# ------------------------
cities = [
    "Turin, Italy",
    "Paris, France",
    "New York City, New York, USA",
    "Beijing, China",
    "Piedmont, California, USA"
]

results = []

# ------------------------
# 路径处理与绘图函数
# ------------------------
def reconstruct_path(G, start, end):
    path = []
    curr = end
    while curr != start:
        prev = G.nodes[curr]["previous"]
        path.append((prev, curr))
        curr = prev
    return path[::-1]

def plot_path(G, path_edges, city, algo):
    for u, v in path_edges:
        if G.has_edge(u, v, 0):
            G.edges[(u, v, 0)]["color"] = "red"
            G.edges[(u, v, 0)]["linewidth"] = 2
    ox.plot_graph(
        G,
        edge_color=[G.edges[e].get("color", "#cccccc") for e in G.edges],
        edge_linewidth=[G.edges[e].get("linewidth", 0.5) for e in G.edges],
        node_size=0,
        bgcolor="white",
        save=True,
        filepath=os.path.join("results", f"{city.replace(', ', '_')}_{algo}.png"),
        dpi=300
    )

# ------------------------
# 清空 & 创建结果文件夹
# ------------------------
if os.path.exists("results"):
    shutil.rmtree("results")
os.makedirs("results", exist_ok=True)

# ------------------------
# 主循环：每个城市测试
# ------------------------
for city in cities:
    print(f"\nRunning on {city}")
    G = ox.graph_from_place(city, network_type="drive", simplify=True)

    # 添加 maxspeed 和 weight
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

    # 采样有效且足够远的起点终点（至少2公里）
    while True:
        start, end = random.sample(nodes, 2)
        dist = ox.distance.euclidean_dist_vec(
            G.nodes[start]["y"], G.nodes[start]["x"],
            G.nodes[end]["y"], G.nodes[end]["x"]
        )
        if dist > 2000:
            break

    # Dijkstra
    d_it, d_vis = dijkstra(G, start, end)
    d_path = reconstruct_path(G, start, end)
    plot_path(G, d_path, city, "Dijkstra")

    # Reset edge styles
    for u, v, data in G.edges(keys=False, data=True):
        data["color"] = "#cccccc"
        data["linewidth"] = 0.5

    # A*
    a_it, a_vis = astar(G, start, end)
    a_path = reconstruct_path(G, start, end)
    plot_path(G, a_path, city, "Astar")

    # Save result entries
    results.append((city, "A*", a_it, a_vis, len(a_path), start, end))
    results.append((city, "Dijkstra", d_it, d_vis, len(d_path), start, end))

# ------------------------
# 保存 CSV 文件
# ------------------------
csv_path = os.path.join("results", "results.csv")
with open(csv_path, mode="w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["City", "Algorithm", "Iterations", "Visited Nodes", "Path Length", "Start Node", "End Node"])
    writer.writerows(results)

print("\n✅ Results saved to 'results/results.csv'")
