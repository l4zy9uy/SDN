import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint

# Define the OS3E Graph
def OS3EGraph():
    g = nx.Graph()
    nx.add_path(g, ["Vancouver", "Seattle"])
    nx.add_path(g, ["Seattle", "Missoula", "Minneapolis", "Chicago"])
    nx.add_path(g, ["Seattle", "Salt Lake City"])
    nx.add_path(g, ["Seattle", "Portland", "Sunnyvale, CA"])
    nx.add_path(g, ["Sunnyvale, CA", "Salt Lake City"])
    nx.add_path(g, ["Sunnyvale, CA", "Los Angeles"])
    nx.add_path(g, ["Los Angeles", "Salt Lake City"])
    nx.add_path(g, ["Los Angeles", "Tucson", "El Paso, TX"])
    nx.add_path(g, ["Salt Lake City", "Denver"])
    nx.add_path(g, ["Denver", "Albuquerque", "El Paso, TX"])
    nx.add_path(g, ["Denver", "Kansas City, MO", "Chicago"])
    nx.add_path(g, ["Kansas City, MO", "Dallas", "Houston"])
    nx.add_path(g, ["El Paso, TX", "Houston"])
    nx.add_path(g, ["Houston", "Jackson, MS", "Memphis", "Nashville"])
    nx.add_path(g, ["Houston", "Baton Rouge", "Jacksonville"])
    nx.add_path(g, ["Chicago", "Indianapolis", "Louisville", "Nashville"])
    nx.add_path(g, ["Nashville", "Atlanta"])
    nx.add_path(g, ["Atlanta", "Jacksonville"])
    nx.add_path(g, ["Jacksonville", "Miami"])
    nx.add_path(g, ["Chicago", "Cleveland"])
    nx.add_path(g, ["Cleveland", "Buffalo", "Boston", "New York", "Philadelphia", "Washington DC"])
    nx.add_path(g, ["Cleveland", "Pittsburgh", "Ashburn, VA", "Washington DC"])
    nx.add_path(g, ["Washington DC", "Raleigh, NC", "Atlanta"])
    return g

# Create the graph
graph = OS3EGraph()

# Define the node mapping based on Table 3
node_mapping = {
    "Boston": 1, "New York": 2, "Philadelphia": 3, "Washington DC": 4, "Ashburn, VA": 5,
    "Pittsburgh": 6, "Cleveland": 7, "Buffalo": 8, "Raleigh, NC": 9, "Indianapolis": 10,
    "Chicago": 11, "Miami": 12, "Jacksonville": 13, "Atlanta": 14, "Louisville": 15,
    "Nashville": 16, "Minneapolis": 17, "Kansas City, MO": 18, "Memphis": 19, "Jackson, MS": 20,
    "Baton Rouge": 21, "Dallas": 22, "Denver": 23, "Albuquerque": 24, "El Paso, TX": 25,
    "Houston": 26, "Salt Lake City": 27, "Tucson": 28, "Los Angeles": 29, "Sunnyvale, CA": 30,
    "Portland": 31, "Seattle": 32, "Missoula": 33, "Vancouver": 34
}

# Use node numbers from Table 3 as labels
labels = {node: node_mapping[node] for node in graph.nodes()}

# Draw the graph with improved visualization settings
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(graph, seed=42)

# Draw nodes with increased size and distinct color
nx.draw_networkx_nodes(graph, pos, node_size=800, node_color="lightblue")

# Draw edges with increased width for better visibility
nx.draw_networkx_edges(graph, pos, edge_color="gray", width=1.5)

# Draw labels with Table 3 numbers
nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_weight="bold", font_color="darkblue")

# Title and display
plt.title("OS3E Network Topology with Table 3 Node Numbers", fontsize=16)
plt.axis("off")  # Turn off axis for a cleaner look
plt.show()
