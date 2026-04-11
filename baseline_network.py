import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import cKDTree

# Network Parameters
AREA_SIZE = 2000
NUM_NODES = 1500
COMM_RANGE = 80
INITIAL_ENERGY = 0.5
SINK_POS = (1000, 1000)


"""
Node Generation: * Adds the central sink node first, assigning it infinite energy so it does not act as a failure point.

Uses np.random.uniform to generate 3000 random (x, y) coordinate pairs within the 2000x2000 grid.

Adds these 3000 coordinates to the graph as sensor nodes, each tagged with 0.5 Joules of initial energy.
"""
def create_wsn_graph(seed=42):
    """Generates the static WSN graph optimized with KDTree."""
    np.random.seed(seed)
    G = nx.Graph()
    
    # 1. Add the Sink Node
    G.add_node("SINK", pos=SINK_POS, type="sink", energy=float('inf'))
    
    # 2. Generate Random Sensor Nodes
    print(f"Generating {NUM_NODES} sensor nodes...")
    positions = np.random.uniform(0, AREA_SIZE, size=(NUM_NODES, 2))
    
    for i in range(NUM_NODES):
        G.add_node(i, pos=tuple(positions[i]), type="sensor", energy=INITIAL_ENERGY)
        
    # 3. Connect Nodes within Communication Range (40m)
    print("Calculating edges using KDTree...")
    
    # Extract all node IDs and positions (including the sink)
    all_node_ids = list(G.nodes())
    pos_dict = nx.get_node_attributes(G, 'pos')
    pos_array = np.array([pos_dict[n] for n in all_node_ids])
    
    # Use KDTree to find all pairs within COMM_RANGE
    tree = cKDTree(pos_array)
    pairs = tree.query_pairs(r=COMM_RANGE)
    
    # Add the edges to the graph
    for i, j in pairs:
        n1, n2 = all_node_ids[i], all_node_ids[j]
        x1, y1 = pos_array[i]
        x2, y2 = pos_array[j]
        
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        G.add_edge(n1, n2, weight=dist)
                
    print(f"Network generated: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


"""
Extracts the stored coordinate data from the generated graph.

Applies conditional formatting: the node tagged as 'sink' is rendered as a large red point (size 100), while nodes tagged as 'sensor' are rendered as small blue points (size 5).

Executes nx.draw_networkx_nodes to display the static 2D topology map
"""
def visualize_network(G):
    """Plots a sample of the network to verify topology."""
    pos = nx.get_node_attributes(G, 'pos')
    colors = ['red' if G.nodes[n]['type'] == 'sink' else 'blue' for n in G.nodes()]
    sizes = [100 if G.nodes[n]['type'] == 'sink' else 5 for n in G.nodes()]
    
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.6)
    plt.title("WSN Topology (3000 nodes, 2000x2000m)")
    plt.show()

# Execute
if __name__ == "__main__":
    wsn_graph = create_wsn_graph()
    visualize_network(wsn_graph)