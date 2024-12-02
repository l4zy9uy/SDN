import json
from pprint import pprint
import networkx as nx
import matplotlib.pyplot as plt
from math import radians
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
def recompute_centroids(clusters, shortest_paths, graph, degree_threshold):
    new_centroids = []
    for centroid, nodes in clusters.items():
        valid_candidates = [n for n in nodes if graph.degree[n] >= degree_threshold]
        if valid_candidates:
            new_centroid = min(
                valid_candidates,
                key=lambda n: sum(shortest_paths[n][other] for other in nodes)
            )
            new_centroids.append(new_centroid)
    return new_centroids


# Select the first centroid based on node degree and shortest path distances
def initialize_centroids(graph, shortest_paths):
    # Calculate node degrees
    node_degrees = dict(graph.degree())
    pprint(node_degrees)

    # Find nodes with the highest degree
    max_degree = max(node_degrees.values())
    candidates = [node for node, degree in node_degrees.items() if degree == max_degree]

    print("Candidates with the highest degree:", candidates)

    # Select the node with the minimum sum of shortest path distances
    candidate_sums = {n: sum(shortest_paths[n].values()) for n in candidates}
    for candidate, total_distance in candidate_sums.items():
        print(f"Node: {candidate}, Total Distance: {total_distance}")

    # First centroid is the one with the minimum total distance
    first_centroid = min(candidates, key=lambda n: candidate_sums[n])
    print(f"First centroid selected: {first_centroid}, Total Distance: {candidate_sums[first_centroid]}")

    return first_centroid

# Advanced K-Means Algorithm
def advanced_kmeans(graph, k, shortest_paths, degree_threshold):
    """
    Advanced K-Means algorithm that calculates the next centroid and
    recomputes the final centroid for each cluster in the same while loop.

    Args:
        graph (networkx.Graph): The network topology as a graph.
        k (int): Number of clusters (controllers).
        shortest_paths (dict): Precomputed shortest path distances.
        degree_threshold (int): Minimum degree required for centroid nodes.

    Returns:
        tuple: Final centroids and clusters.
    """
    # Step 1: Select the first centroid
    first_centroid = initialize_centroids(graph, shortest_paths)
    centroids = [first_centroid]  # Start with the first centroid
    print(f"Initial Centroid: {first_centroid}")

    # Step 2: Initialize clusters with the first centroid
    clusters = assign_clusters(graph, centroids, shortest_paths)
    print(f"Initial clusters: {clusters}")

    # Step 3: Iteratively calculate centroids and recompute final centroids
    while len(centroids) < k:
        # Step 3.1: Calculate the next centroid
        candidate_nodes = [node for node in graph.nodes if node not in centroids]
        next_centroid = max(
            candidate_nodes,
            key=lambda n: min(shortest_paths[n][c] for c in centroids)  # Maximize min distance to existing centroids
        )
        centroids.append(next_centroid)
        print(f"Added new centroid {next_centroid}: {centroids}")

        # Step 3.2: Assign nodes to clusters based on the updated centroids
        clusters = assign_clusters(graph, centroids, shortest_paths)
        print(f"Updated clusters with {len(centroids)} centroids: {clusters}")

        # Step 3.3: Recompute final centroids for each cluster
        for cluster_centroid in centroids:
            current_cluster = clusters[cluster_centroid]
            valid_candidates = [n for n in current_cluster if graph.degree[n] >= degree_threshold]
            if valid_candidates:
                # Recompute the final centroid as the node with the minimum sum of distances
                refined_centroid = min(
                    valid_candidates,
                    key=lambda n: sum(shortest_paths[n][other] for other in current_cluster)
                )
                # Update the centroid for this cluster
                centroids[centroids.index(cluster_centroid)] = refined_centroid
                print(f"Refined Centroid for Cluster {len(centroids)}: {refined_centroid}")

    # Step 4: Final assignment of clusters
    clusters = assign_clusters(graph, centroids, shortest_paths)

    return centroids, clusters


def calculate_degree_threshold(graph):
    """
    Calculate the degree threshold for centroid selection based on the average node degree.

    Args:
        graph (networkx.Graph): The input graph representing the network topology.

    Returns:
        int: The degree threshold (rounded average degree of all nodes).
    """
    # Compute degrees of all nodes in the graph
    degrees = dict(graph.degree())

    # Calculate the average degree
    avg_degree = sum(degrees.values()) / len(degrees)

    # Round the average degree to the nearest integer
    degree_threshold = round(avg_degree)

    print(f"Average Degree: {avg_degree}, Degree Threshold: {degree_threshold}")
    return degree_threshold

def main():
    json_file_path = "latlong.json"
    latlong = load_latlong(json_file_path)

    # Build the weighted graph directly
    weighted_graph = build_weighted_graph(latlong)

    # Compute shortest path distances
    shortest_paths = compute_shortest_path_distances(weighted_graph)
    pprint(f"shortest paths: {shortest_paths}")

    degree_threshold = calculate_degree_threshold(weighted_graph)
    # Run Advanced K-Means
    num_clusters = 4
    centroids, clusters = advanced_kmeans(weighted_graph, num_clusters, shortest_paths, degree_threshold)

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
