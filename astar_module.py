# astar_module.py
import osmnx as ox
import heapq

def heuristic(G, n1, n2):
    return ox.distance.great_circle(G.nodes[n1]['y'], G.nodes[n1]['x'], G.nodes[n2]['y'], G.nodes[n2]['x'])

def astar(G, start, end):
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["g"] = float("inf")
        G.nodes[node]["previous"] = None
    G.nodes[start]["g"] = 0
    pq = [(heuristic(G, start, end), 0, start)]
    iterations = 0
    visited = 0

    while pq:
        f, g, curr = heapq.heappop(pq)
        if curr == end:
            break
        if G.nodes[curr]["visited"]:
            continue
        G.nodes[curr]["visited"] = True
        visited += 1
        for _, neighbor, data in G.out_edges(curr, data=True):
            tentative_g = g + data["weight"]
            if tentative_g < G.nodes[neighbor]["g"]:
                G.nodes[neighbor]["g"] = tentative_g
                G.nodes[neighbor]["previous"] = curr
                heapq.heappush(pq, (tentative_g + heuristic(G, neighbor, end), tentative_g, neighbor))
        iterations += 1

    return iterations, visited
