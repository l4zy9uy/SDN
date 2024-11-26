import json
from pprint import pprint

import networkx as nx
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
from sklearn.metrics.pairwise import haversine_distances
def load_latlong(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def validate_positions(graph, latlong):
    valid_positions = {}
    for node in graph.nodes:
        if node in latlong:
            valid_positions[node] = (float(latlong[node]["Longitude"]), float(latlong[node]["Latitude"]))
        else:
            print(f"Warning: No lat/long data for node '{node}'. It will be excluded from the plot.")
    return valid_positions


# Calculate the geographic (Haversine) distance between two points
def haversine(coord1, coord2):
    r = 6371
    lon1, lat1 = radians(float(coord1['Latitude'])), radians(float(coord1['Longitude']))
    lon2, lat2 = radians(float(coord2['Latitude'])), radians(float(coord2['Longitude']))

    distance = haversine_distances([[lon1, lat1], [lon2, lat2]]) * r
    # distance is 2D array based on scikit-learn doc
    print(distance[0][1])
    return distance[0][1]


# Build a weighted graph using geographic distances
# Build the network topology and assign geographic distances as weights
def build_weighted_graph(latlong):
    g = nx.Graph()

    # Define network topology (nodes and connections)
    edges = [
        ("Vancouver", "Seattle"),
        ("Seattle", "Missoula"), ("Missoula", "Minneapolis"), ("Minneapolis", "Chicago"),
        ("Seattle", "Salt Lake City"),
        ("Seattle", "Portland"), ("Portland", "Sunnyvale"),
        ("Sunnyvale", "Salt Lake City"), ("Sunnyvale", "Los Angeles"),
        ("Los Angeles", "Salt Lake City"), ("Los Angeles", "Tucson"), ("Tucson", "El Paso"),
        ("Salt Lake City", "Denver"), ("Denver", "Albuquerque"), ("Albuquerque", "El Paso"),
        ("Denver", "Kansas City"), ("Kansas City", "Chicago"),
        ("Kansas City", "Dallas"), ("Dallas", "Houston"),
        ("El Paso", "Houston"),
        ("Houston", "Jackson"), ("Jackson", "Memphis"), ("Memphis", "Nashville"),
        ("Houston", "Baton Rouge"), ("Baton Rouge", "Jacksonville"),
        ("Chicago", "Indianapolis"), ("Indianapolis", "Louisville"), ("Louisville", "Nashville"),
        ("Nashville", "Atlanta"), ("Atlanta", "Jacksonville"),
        ("Jacksonville", "Miami"),
        ("Chicago", "Cleveland"),
        ("Cleveland", "Buffalo"), ("Buffalo", "Boston"), ("Boston", "New York"),
        ("New York", "Philadelphia"), ("Philadelphia", "Washington"),
        ("Cleveland", "Pittsburgh"), ("Pittsburgh", "Ashburn"), ("Ashburn", "Washington"),
        ("Washington", "Raleigh"), ("Raleigh", "Atlanta")
    ]

    # Add edges and compute weights based on lat/long
    for u, v in edges:
        if u in latlong and v in latlong:  # Ensure both nodes have valid coordinates
            distance = haversine(latlong[u], latlong[v])  # Calculate geographic distance
            g.add_edge(u, v, weight=distance)
        else:
            print(f"Skipping edge ({u}, {v}) due to missing lat/long data.")

    return g


# Precompute shortest path distances using weighted graph
def compute_shortest_path_distances(weighted_graph):
    return dict(nx.all_pairs_dijkstra_path_length(weighted_graph, weight="weight"))


# Select initial centroids based on node degree and distance
def initialize_centroids(graph, k, shortest_paths):
    node_degrees = dict(graph.degree())
    pprint(node_degrees)
    # Calculate node degrees
    max_degree = max(node_degrees.values())  # Highest degree value
    candidates = [node for node, degree in node_degrees.items() if degree == max_degree]

    print("candidates: ", candidates)

    print("\nCandidates with highest degree and their total shortest path distances:")
    candidate_sums = {n: sum(shortest_paths[n].values()) for n in candidates}
    for candidate, total_distance in candidate_sums.items():
        print(f"Node: {candidate}, Total Distance: {total_distance}")

    # Select the first centroid as the node with the minimum total shortest path distance
    first_centroid = min(candidates, key=lambda n: candidate_sums[n])
    print(f"\nFirst centroid selected: {first_centroid} with total distance: {candidate_sums[first_centroid]}")

    centroids = [first_centroid]

    # Select remaining centroids to maximize distance from existing centroids
    while len(centroids) < k:
        candidate_nodes = [node for node in graph.nodes if node not in centroids]
        farthest_node = max(
            candidate_nodes,
            key=lambda n: sum(shortest_paths[n][c] for c in centroids)
        )
        centroids.append(farthest_node)

    return centroids


# Assign nodes to the nearest centroid
def assign_clusters(graph, centroids, shortest_paths):
    clusters = {centroid: [] for centroid in centroids}
    for node in graph.nodes:
        nearest_centroid = min(
            centroids,
            key=lambda c: shortest_paths[node][c]
        )
        clusters[nearest_centroid].append(node)
    return clusters


# Recompute centroids based on cluster members
def recompute_centroids(clusters, shortest_paths):
    new_centroids = []
    for centroid, nodes in clusters.items():
        new_centroid = min(
            nodes,
            key=lambda n: sum(shortest_paths[n][other] for other in nodes)
        )
        new_centroids.append(new_centroid)
    return new_centroids


# Advanced K-Means Algorithm
def advanced_kmeans(graph, k, shortest_paths):
    centroids = initialize_centroids(graph, k, shortest_paths)
    while True:
        clusters = assign_clusters(graph, centroids, shortest_paths)
        new_centroids = recompute_centroids(clusters, shortest_paths)
        if set(new_centroids) == set(centroids):
            break
        centroids = new_centroids
    return centroids, clusters


def main():
    json_file_path = "latlong.json"
    latlong = load_latlong(json_file_path)

    # Build the weighted graph directly
    weighted_graph = build_weighted_graph(latlong)

    # Compute shortest path distances
    shortest_paths = compute_shortest_path_distances(weighted_graph)
    pprint(f"shortest paths: {shortest_paths}")

    # Run Advanced K-Means
    num_clusters = 4
    centroids, clusters = advanced_kmeans(weighted_graph, num_clusters, shortest_paths)

    print(f"Final Centroids: {centroids}")
    print(f"Clusters: {clusters}")

    # Draw the graph with geographic positions
    pos = {node: (float(latlong[node]["Longitude"]), float(latlong[node]["Latitude"])) for node in weighted_graph.nodes}
    plt.figure(figsize=(14, 10))
    colors = ["red", "blue", "green", "purple", "orange"]

    # Draw nodes with cluster colors
    for cluster_id, centroid in enumerate(centroids):
        cluster_nodes = clusters[centroid]
        cluster_pos = {node: pos[node] for node in cluster_nodes if node in pos}

        nx.draw_networkx_nodes(
            G=weighted_graph,
            pos=cluster_pos,
            nodelist=cluster_nodes,
            node_size=300,
            node_color=colors[cluster_id % len(colors)],
        )
        plt.scatter([], [], color=colors[cluster_id % len(colors)], s=100, label=f"Cluster {cluster_id}")

    edges_to_draw = [(u, v) for u, v in weighted_graph.edges if u in pos and v in pos]
    nx.draw_networkx_edges(weighted_graph, pos, edgelist=edges_to_draw, edge_color="gray", width=1.5)

    edge_labels = nx.get_edge_attributes(weighted_graph, "weight")  # Get the 'weight' (distance) attribute
    edge_labels = {edge: f"{distance:.1f} km" for edge, distance in edge_labels.items()}  # Format distances
    nx.draw_networkx_edge_labels(weighted_graph, pos, edge_labels=edge_labels, font_size=8, font_color="blue")

    for centroid in centroids:
        if centroid in pos:
            plt.scatter(
                pos[centroid][0], pos[centroid][1],
                color="black",
                s=800,
                marker="x",
            )

    plt.scatter([], [], color="black", s=100, marker="x", label="Centroid")
    plt.title("Advanced K-Means with Geographic Distances for SDN Controller Placement", fontsize=16)
    plt.legend(scatterpoints=1, loc="upper right")
    plt.show()

if __name__ == "__main__":
    main()
