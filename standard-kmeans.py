import json
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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
    valid_positions = {}
    for node in graph.nodes:
        print (node)
        if node in latlong:
            valid_positions[node] = (float(latlong[node]["Longitude"]), float(latlong[node]["Latitude"]))
        else:
            print(f"Warning: No lat/long data for node '{node}'. It will be excluded from the plot.")
    return valid_positions

def main():
    json_file_path = "latlong.json"
    latlong = load_latlong(json_file_path)

    graph = build_graph()
    pos = validate_positions(graph, latlong)
    print(pos)

    num_clusters = 5

    data = list(pos.values())
    print("data: ", data)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)
    clusters = {node: kmeans.labels_[i] for i, node in enumerate(pos.keys())}
    print("cluster: ", clusters)
    centroids = kmeans.cluster_centers_

    edges_to_draw = [
        (u, v) for u, v in graph.edges if u in pos and v in pos
    ]

    plt.figure(figsize=(14, 10))
    colors = ["red", "blue", "green", "purple", "orange"]

    # Draw nodes with cluster colors
    for cluster_id in range(num_clusters):
        cluster_nodes = [node for node in clusters if clusters[node] == cluster_id]
        cluster_pos = {node: pos[node] for node in cluster_nodes}

        print(f"Cluster {cluster_id}: {cluster_pos}")

        nx.draw_networkx_nodes(
            G=graph,
            pos=cluster_pos,
            nodelist=cluster_nodes,
            node_size=300,
            node_color=colors[cluster_id % len(colors)],
            label=f"Cluster {cluster_id}"
        )
        print(f"Cluster {cluster_id} drawn successfully.")

    nx.draw_networkx_edges(graph, pos, edgelist=edges_to_draw, edge_color="gray", width=1.5)

    for idx, centroid in enumerate(centroids):
        plt.scatter(
            centroid[0], centroid[1],
            color="black",
            s=100,
            marker="x",
        )

    # Title and legend
    plt.title("OS3E Network Topology with K-Means Clustering", fontsize=16)
    plt.scatter([], [], color="black", s=100, marker="x", label="Centroid")
    plt.legend(scatterpoints=1, loc="upper right")
    plt.axis()
    plt.show()

if __name__ == "__main__":
    main()
