import json
import random
import sys

import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Define the OS3E Graph
def build_graph():
    g = nx.Graph()
    nx.add_path(g, ["Vancouver", "Seattle"])
    nx.add_path(g, ["Seattle", "Missoula", "Minneapolis", "Chicago"])
    nx.add_path(g, ["Seattle", "Salt Lake City"])
    nx.add_path(g, ["Seattle", "Portland", "Sunnyvale"])
    nx.add_path(g, ["Sunnyvale", "Salt Lake City"])
    nx.add_path(g, ["Sunnyvale", "Los Angeles"])
    nx.add_path(g, ["Los Angeles", "Salt Lake City"])
    nx.add_path(g, ["Los Angeles", "Tucson", "El Paso"])
    nx.add_path(g, ["Salt Lake City", "Denver"])
    nx.add_path(g, ["Denver", "Albuquerque", "El Paso"])
    nx.add_path(g, ["Denver", "Kansas City", "Chicago"])
    nx.add_path(g, ["Kansas City", "Dallas", "Houston"])
    nx.add_path(g, ["El Paso", "Houston"])
    nx.add_path(g, ["Houston", "Jackson", "Memphis", "Nashville"])
    nx.add_path(g, ["Houston", "Baton Rouge", "Jacksonville"])
    nx.add_path(g, ["Chicago", "Indianapolis", "Louisville", "Nashville"])
    nx.add_path(g, ["Nashville", "Atlanta"])
    nx.add_path(g, ["Atlanta", "Jacksonville"])
    nx.add_path(g, ["Jacksonville", "Miami"])
    nx.add_path(g, ["Chicago", "Cleveland"])
    nx.add_path(g, ["Cleveland", "Buffalo", "Boston", "New York", "Philadelphia", "Washington"])
    nx.add_path(g, ["Cleveland", "Pittsburgh", "Ashburn", "Washington"])
    nx.add_path(g, ["Washington", "Raleigh", "Atlanta"])
    return g

def load_latlong(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def validate_positions(graph, latlong):
    """Ensure that positions are defined only for nodes present in the graph."""
    valid_positions = {}
    for node in graph.nodes:
        print (node)
        if node in latlong:
            valid_positions[node] = (float(latlong[node]["Longitude"]), float(latlong[node]["Latitude"]))
        else:
            print(f"Warning: No lat/long data for node '{node}'. It will be excluded from the plot.")
    return valid_positions

def apply_kmeans(pos, num_clusters):
    """Apply K-means clustering to the node positions."""
    data = list(pos.values())
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(data)
    clusters = kmeans.labels_
    return {node: clusters[i] for i, node in enumerate(pos.keys())}


# Main function
def main():
    # Path to the JSON file
    json_file_path = "latlong.json"

    # Load latitude and longitude data
    latlong = load_latlong(json_file_path)

    # Create the graph
    graph = build_graph()

    # Validate positions
    pos = validate_positions(graph, latlong)
    print(pos)
    num_clusters = 5  # Adjust this to the desired number of clusters
    clusters = apply_kmeans(pos, num_clusters)

    data = list(pos.values())
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)
    clusters = {node: kmeans.labels_[i] for i, node in enumerate(pos.keys())}
    centroids = kmeans.cluster_centers_

    # Ensure edges are only drawn if both nodes have valid positions
    edges_to_draw = [
        (u, v) for u, v in graph.edges if u in pos and v in pos
    ]

    # Draw the graph with geographic positions
    plt.figure(figsize=(14, 10))
    colors = ["red", "blue", "green", "purple", "orange"]

    # Draw nodes with cluster colors
    for cluster_id in range(num_clusters):
        # Filter nodes belonging to this cluster and present in pos
        cluster_nodes = [node for node in clusters if clusters[node] == cluster_id]
        cluster_pos = {node: pos[node] for node in cluster_nodes}

        # Debugging output
        print(f"Cluster {cluster_id}: {cluster_pos}")

        # Draw the nodes in this cluster
        nx.draw_networkx_nodes(
            G=graph,
            pos=cluster_pos,
            nodelist=cluster_nodes,
            node_size=800,
            node_color=colors[cluster_id % len(colors)],
            label=f"Cluster {cluster_id}"
        )
        print(f"Cluster {cluster_id} drawn successfully.")

    # Draw edges
    nx.draw_networkx_edges(graph, pos, edgelist=edges_to_draw, edge_color="gray", width=1.5)

    # Draw node labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold", font_color="black")

    for idx, centroid in enumerate(centroids):
        plt.scatter(
            centroid[0], centroid[1],
            color="black",
            s=200,
            marker="x",
            label=f"Centroid {idx}"
        )


    # Title and legend
    plt.title("OS3E Network Topology with K-Means Clustering", fontsize=16)
    plt.legend(scatterpoints=1, loc="upper left")
    plt.axis("off")  # Turn off axis for a cleaner look
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
