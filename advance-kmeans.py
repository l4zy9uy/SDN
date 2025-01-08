import json
from pprint import pprint
import networkx as nx
import matplotlib.pyplot as plt
from math import radians
from sklearn.metrics.pairwise import haversine_distances
import random

def load_latlong(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# Calculate the geographic (Haversine) distance between two points
def haversine(coord1, coord2):
    r = 6371
    lon1, lat1 = radians(float(coord1['Latitude'])), radians(float(coord1['Longitude']))
    lon2, lat2 = radians(float(coord2['Latitude'])), radians(float(coord2['Longitude']))

    distance = haversine_distances([[lon1, lat1], [lon2, lat2]]) * r
    # distance is 2D array based on scikit-learn doc
    #print(distance[0][1])
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
    return g


# Precompute shortest path distances using weighted graph
def compute_shortest_path_distances(weighted_graph):
    return dict(nx.all_pairs_dijkstra_path_length(weighted_graph, weight="weight"))

# Assign nodes to the nearest centroid
def assign_clusters(graph, centroids, shortest_paths):
    #print(f"current centroid: {centroids}")
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
    #pprint(node_degrees)

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
    iteratively recomputes the final centroids until stabilization.

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
    #print(f"Initial Centroid: {first_centroid}")

    # Step 2: Initialize clusters with the first centroid
    clusters = assign_clusters(graph, centroids, shortest_paths)
    #print(f"Initial clusters: {clusters}")

    # Step 3: Iteratively calculate centroids and recompute final centroids
    while len(centroids) < k:
        # Step 3.1: Calculate the next centroid
        last_centroid = centroids[-1]  # Focus on the most recently added centroid
        candidate_nodes = [
            node for node in graph.nodes
            if node not in centroids and graph.degree[node] >= degree_threshold
        ]
        if candidate_nodes:  # Ensure there are valid candidates
            next_centroid = max(
                candidate_nodes,
                key=lambda n: shortest_paths[n][last_centroid]  # Maximize distance from the last centroid
            )
            centroids.append(next_centroid)
            #print(f"Added new centroid {next_centroid}: {centroids}")
        else:
            #print("No valid candidates for the next centroid.")
            break  # Exit the l

        # Step 3.2: Refine clusters and centroids until stabilization
        while True:
            # Assign nodes to clusters based on the current centroids
            clusters = assign_clusters(graph, centroids, shortest_paths)
            current_cluster = clusters[next_centroid]  # Focus on the latest added centroid
            #print(f"current cluster: {next_centroid}")
            # Identify valid candidates for the current cluster
            valid_candidates = [n for n in current_cluster if graph.degree[n] >= degree_threshold]
            if valid_candidates:
                # Compute the sum of shortest path distances to all nodes in the cluster for each valid candidate
                refined_centroid = min(
                    valid_candidates,
                    key=lambda n: sum(shortest_paths[n][other] for other in current_cluster)
                )

                # Check if the centroid has stabilized
                if refined_centroid == next_centroid:
                    break  # Stop refining if the centroid is stable
                next_centroid = refined_centroid  # Update the centroid for further refinement
                centroids[-1] = refined_centroid  # Update the list of centroids
                #print(f"Refined Centroid for Cluster: {refined_centroid}")

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

    #print(f"Average Degree: {avg_degree}, Degree Threshold: {degree_threshold}")
    return degree_threshold

def calculate_average_propagation_latency(clusters, centroids, shortest_paths, total_nodes):
    """
    Calculate the average propagation latency based on the 15th formula in the paper.

    Args:
        clusters (dict): A dictionary where keys are centroids, and values are lists of nodes assigned to each centroid.
        centroids (list): A list of current centroids (controllers).
        shortest_paths (dict): A dictionary of precomputed shortest path distances between all nodes.
        total_nodes (int): Total number of nodes in the network (N).

    Returns:
        float: The average propagation latency across all clusters.
    """
    total_latency = 0  # Sum of all latencies
    num_centroids = len(centroids)  # Number of controllers (K)

    for centroid in centroids:
        cluster_nodes = clusters[centroid]
        for node in cluster_nodes:
            # Add the shortest path distance from the node to its centroid
            total_latency += shortest_paths[node][centroid]

    # Formula: Average Latency = Total Latency / (N - K)
    if total_nodes > num_centroids:
        avg_latency = total_latency / (total_nodes - num_centroids)
    else:
        avg_latency = float('inf')  # Avoid division by zero or invalid state

    return avg_latency


def test_k_values_with_maps(graph, latlong, k_values, L1):
    results = []
    shortest_paths = compute_shortest_path_distances(graph)
    degree_threshold = calculate_degree_threshold(graph)

    for k in k_values:
        #print(f"Testing for k = {k}")
        centroids, clusters = advanced_kmeans(graph, k, shortest_paths, degree_threshold)
        avg_latency = calculate_average_propagation_latency(clusters, centroids, shortest_paths, graph.number_of_nodes())
        cost_benefit_ratio = L1 / (avg_latency * k)
        results.append((k, cost_benefit_ratio))
        #print(f"k={k}, Avg Latency={avg_latency:.4f}, Cost/Benefit Ratio={cost_benefit_ratio:.4f}")

        # Draw map for each k
        pos = {node: (float(latlong[node]["Longitude"]), float(latlong[node]["Latitude"])) for node in graph.nodes}
        plt.figure(figsize=(14, 10))
        colors = ["red", "blue", "green", "purple", "orange", "yellow", "pink", "cyan"]

        for cluster_id, centroid in enumerate(centroids):
            cluster_nodes = clusters[centroid]
            cluster_pos = {node: pos[node] for node in cluster_nodes if node in pos}

            nx.draw_networkx_nodes(
                G=graph,
                pos=cluster_pos,
                nodelist=cluster_nodes,
                node_size=300,
                node_color=colors[cluster_id % len(colors)],
            )
            plt.scatter([], [], color=colors[cluster_id % len(colors)], s=100, label=f"Cluster {cluster_id}")

        edges_to_draw = [(u, v) for u, v in graph.edges if u in pos and v in pos]
        nx.draw_networkx_edges(graph, pos, edgelist=edges_to_draw, edge_color="gray", width=1.5)

        edge_labels = nx.get_edge_attributes(graph, "weight")
        edge_labels = {edge: f"{distance:.1f} km" for edge, distance in edge_labels.items()}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, font_color="blue")

        for centroid in centroids:
            if centroid in pos:
                plt.scatter(
                    pos[centroid][0], pos[centroid][1],
                    color="black",
                    s=800,
                    marker="x",
                )

        plt.scatter([], [], color="black", s=100, marker="x", label="Centroid")
        plt.title(f"Clustering Map for k={k}", fontsize=16)
        plt.legend(scatterpoints=1, loc="upper right")
        plt.show()

    return results


def plot_cost_benefit_ratio(results):
    k_values, cost_benefit_ratios = zip(*results)

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, cost_benefit_ratios, marker='o', label='Cost/Benefit Ratio')
    plt.xlabel('Number of Controllers (k)')
    plt.ylabel('Cost/Benefit Ratio')
    plt.title('Cost/Benefit Ratio vs. Number of Controllers (k)')
    plt.grid(True)
    plt.legend()
    plt.show()

def calculate_randomized_latency(graph, k, latlong, shortest_paths, total_nodes):
    """
    Calculate the average propagation latency over 9 randomizations for a given k.

    Args:
        graph (networkx.Graph): The network topology as a graph.
        k (int): Number of clusters (controllers).
        latlong (dict): Dictionary of lat/long for each node.
        shortest_paths (dict): Precomputed shortest path distances.
        total_nodes (int): Total number of nodes in the network.

    Returns:
        float: Average latency over 9 random runs.
    """
    latencies = []
    for i in range(9):
        # Choose k random centers
        random_centers = random.sample(list(graph.nodes), k)
        # Assign clusters based on random centers
        clusters = assign_clusters(graph, random_centers, shortest_paths)
        # Calculate latency
        latency = calculate_average_propagation_latency(clusters, random_centers, shortest_paths, total_nodes)
        latencies.append(latency)
        #print(f"Random run {i+1}, Latency: {latency:.4f}")
    # Return the average latency
    return sum(latencies) / len(latencies)


def test_k_values_with_randomization(graph, latlong, k_values, L1):
    """
    Test k values for both Advanced K-Means and randomized centers,
    and calculate the average propagation latency.

    Args:
        graph (networkx.Graph): The network topology as a graph.
        latlong (dict): Dictionary of lat/long for each node.
        k_values (range): Range of k values to test.
        L1 (float): Baseline latency.

    Returns:
        list: Results containing k, advanced latency, random latency, and cost-benefit ratios.
    """
    results = []
    shortest_paths = compute_shortest_path_distances(graph)
    degree_threshold = calculate_degree_threshold(graph)

    for k in k_values:
        print(f"Testing for k = {k}")
        # Advanced K-Means latency
        centroids, clusters = advanced_kmeans(graph, k, shortest_paths, degree_threshold)
        advanced_latency = calculate_average_propagation_latency(clusters, centroids, shortest_paths, graph.number_of_nodes())
        cost_benefit_ratio_advanced = L1 / (advanced_latency * k)

        # Randomized latency
        random_latency = calculate_randomized_latency(graph, k, latlong, shortest_paths, graph.number_of_nodes())
        cost_benefit_ratio_random = L1 / (random_latency * k)

        results.append((k, advanced_latency, random_latency, cost_benefit_ratio_advanced, cost_benefit_ratio_random))
        print(f"k={k}, Advanced Latency={advanced_latency:.4f}, Random Latency={random_latency:.4f}")
        print(f"Advanced Cost/Benefit Ratio={cost_benefit_ratio_advanced:.4f}, Random Cost/Benefit Ratio={cost_benefit_ratio_random:.4f}")
    return results


def plot_latency_ratio(results):
    """
    Plot the ratio of Randomized Latency to Advanced K-Means Latency as a bar graph.

    Args:
        results (list): Results containing k, advanced latency, and random latency.
    """
    k_values, advanced_latencies, random_latencies, _, _ = zip(*results)
    # Calculate the ratio of Randomized Latency to Advanced Latency
    latency_ratios = [random / advanced for random, advanced in zip(random_latencies, advanced_latencies)]

    # Plot the bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(k_values, latency_ratios, color='skyblue', edgecolor='black')
    plt.xlabel('Number of Controllers (k)')
    plt.ylabel('Ratio of Randomized to Advanced Latency')
    plt.title('Ratio of Randomized Latency to Advanced K-Means Latency')
    plt.xticks(k_values)  # Ensure x-axis shows all k values
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()




def main():
    json_file_path = "latlong.json"
    latlong = load_latlong(json_file_path)

    # Build the weighted graph directly
    weighted_graph = build_weighted_graph(latlong)

    # Define the reference latency (L1)
    L1 = 1588.075  # Example baseline value from your setup

    # Test for k values from 1 to 8
    k_values = range(1, 9)
    results = test_k_values_with_randomization(weighted_graph, latlong, k_values, L1)

    # Plot the latency comparison
    plot_latency_ratio( results)


    results = test_k_values_with_maps(weighted_graph, latlong, k_values, L1)

    # Plot the Cost/Benefit Ratio
    plot_cost_benefit_ratio(results)

if __name__ == "__main__":
    main()
