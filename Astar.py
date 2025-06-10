import osmnx as ox
import random
import heapq
import time
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# 样式设置
# --------------------------
def style_unvisited_edge(edge):        
    G.edges[edge]["color"] = "#d36206"
    G.edges[edge]["alpha"] = 0.2
    G.edges[edge]["linewidth"] = 0.5

def style_visited_edge(edge):
    G.edges[edge]["color"] = "#40a9ff"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_active_edge(edge):
    G.edges[edge]["color"] = "red"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 2

def style_path_edge(edge):
    G.edges[edge]["color"] = "white"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 5

def plot_graph():
    ox.plot_graph(
        G,
        node_size = 5,
        edge_color = [ G.edges[e]["color"] for e in G.edges ],
        edge_alpha = [ G.edges[e]["alpha"] for e in G.edges ],
        edge_linewidth = [ G.edges[e]["linewidth"] for e in G.edges ],
        node_color = "white",
        bgcolor = "#18080e"
    )

# --------------------------
# 启发式函数
# --------------------------
def heuristic(n1, n2):
    return ox.distance.great_circle(G.nodes[n1]["y"], G.nodes[n1]["x"], G.nodes[n2]["y"], G.nodes[n2]["x"])

# --------------------------
# A*算法
# --------------------------
def astar(start, end):
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["g"] = float("inf")
        G.nodes[node]["previous"] = None

    G.nodes[start]["g"] = 0
    pq = [(heuristic(start, end), 0, start)]
    iterations = 0
    visited = 0
    start_time = time.time()

    while pq:
        f, g, curr = heapq.heappop(pq)
        if curr == end:
            break
        if G.nodes[curr]["visited"]:
            continue
        G.nodes[curr]["visited"] = True
        visited += 1
        for _, neighbor, data in G.out_edges(curr, data=True):
            style_visited_edge((curr, neighbor, 0))
            tentative_g = g + data["weight"]
            if tentative_g < G.nodes[neighbor]["g"]:
                G.nodes[neighbor]["g"] = tentative_g
                G.nodes[neighbor]["previous"] = curr
                heapq.heappush(pq, (tentative_g + heuristic(neighbor, end), tentative_g, neighbor))
                style_active_edge((curr, neighbor, 0))
        iterations += 1

    exec_time = (time.time() - start_time) * 1000
    return iterations, visited, exec_time

# --------------------------
# Dijkstra算法
# --------------------------
def dijkstra(start, end):
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["distance"] = float("inf")
        G.nodes[node]["previous"] = None
    G.nodes[start]["distance"] = 0
    pq = [(0, start)]
    iterations = 0
    visited = 0
    start_time = time.time()

    while pq:
        dist, curr = heapq.heappop(pq)
        if curr == end:
            break
        if G.nodes[curr]["visited"]:
            continue
        G.nodes[curr]["visited"] = True
        visited += 1
        for _, neighbor, data in G.out_edges(curr, data=True):
            style_visited_edge((curr, neighbor, 0))
            new_dist = dist + data["weight"]
            if new_dist < G.nodes[neighbor]["distance"]:
                G.nodes[neighbor]["distance"] = new_dist
                G.nodes[neighbor]["previous"] = curr
                heapq.heappush(pq, (new_dist, neighbor))
                style_active_edge((curr, neighbor, 0))
        iterations += 1

    exec_time = (time.time() - start_time) * 1000
    return iterations, visited, exec_time

# --------------------------
# 路径重建
# --------------------------
def reconstruct_path(start, end):
    curr = end
    while curr != start:
        prev = G.nodes[curr]["previous"]
        style_path_edge((prev, curr, 0))
        curr = prev

# --------------------------
# 可视化对比图
# --------------------------
def plot_comparison(d_data, a_data):
    labels = ['Iterations', 'Visited Nodes', 'Time (ms)']
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, d_data, width, label='Dijkstra', color='#40a9ff')
    ax.bar(x + width/2, a_data, width, label='A*', color='#ff758c')

    ax.set_ylabel('Count')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.show()

# --------------------------
# 主程序
# --------------------------
if __name__ == "__main__":
    place = "Turin, Piedmont, Italy"
    G = ox.graph_from_place(place, network_type="drive", simplify=True)

    for edge in G.edges:
        maxspeed = G.edges[edge].get("maxspeed", 50)
        if isinstance(maxspeed, list):
            maxspeed = min([int(s) for s in maxspeed if str(s).isdigit()])
        elif isinstance(maxspeed, str):
            digits = ''.join(filter(str.isdigit, maxspeed))
            maxspeed = int(digits) if digits else 50
        else:
            maxspeed = 50
        G.edges[edge]["maxspeed"] = maxspeed
        G.edges[edge]["weight"] = G.edges[edge]["length"] / (maxspeed * 1000 / 3600)

    for edge in G.edges:
        style_unvisited_edge(edge)

    nodes = list(G.nodes)
    start, end = random.sample(nodes, 2)
    print(f"Start: {start}, End: {end}")

    d_it, d_vis, d_time = dijkstra(start, end)
    reconstruct_path(start, end)

    for edge in G.edges:
        style_unvisited_edge(edge)

    a_it, a_vis, a_time = astar(start, end)
    reconstruct_path(start, end)

    print("\n=== Performance Comparison ===")
    print(f"{'Metric':<15} | {'Dijkstra':<10} | {'A*':<10}")
    print("-" * 40)
    print(f"{'Iterations':<15} | {d_it:<10} | {a_it:<10}")
    print(f"{'Visited Nodes':<15} | {d_vis:<10} | {a_vis:<10}")
    print(f"{'Time (ms)':<15} | {d_time:<10.2f} | {a_time:<10.2f}")

    plot_graph()
    plot_comparison([d_it, d_vis, d_time], [a_it, a_vis, a_time])
