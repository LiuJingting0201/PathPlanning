# dijkstra_module.py
import heapq

def dijkstra(G, start, end):
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["distance"] = float("inf")
        G.nodes[node]["previous"] = None
    G.nodes[start]["distance"] = 0
    pq = [(0, start)]
    iterations = 0
    visited = 0

    while pq:
        dist, curr = heapq.heappop(pq)
        if curr == end:
            break
        if G.nodes[curr]["visited"]:
            continue
        G.nodes[curr]["visited"] = True
        visited += 1
        for _, neighbor, data in G.out_edges(curr, data=True):
            new_dist = dist + data["weight"]
            if new_dist < G.nodes[neighbor]["distance"]:
                G.nodes[neighbor]["distance"] = new_dist
                G.nodes[neighbor]["previous"] = curr
                heapq.heappush(pq, (new_dist, neighbor))
        iterations += 1
    return iterations, visited
