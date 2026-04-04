import networkx as nx
import random
# Added NUM_NODES and INITIAL_ENERGY to the import
from baseline_network import create_wsn_graph, SINK_POS, NUM_NODES, INITIAL_ENERGY
import itertools
import csv
import math
import networkx as nx


# Simulation Parameters
MAX_ROUNDS = 3000

# Static Protocol Parameters
K_PHANTOM_HOPS = 8
FAKE_TRAFFIC_RATIO = 0.5

# Standard WSN Physics Constants (Replaces TX_ENERGY/RX_ENERGY)
E_ELEC = 50e-9      # 50 nJ/bit for running circuitry
E_AMP = 100e-12     # 100 pJ/bit/m^2 for the amplifier
PACKET_SIZE = 2000  # bits per packet


# Static Protocol Parameters (Fixed for the entire simulation)
K_PHANTOM_HOPS = 8
FAKE_TRAFFIC_RATIO = 0.5


class Adversary:
    def __init__(self, start_node):
        self.current_node = start_node
        self.hop_count = 0
        self.is_captured = False

    def step(self, transmissions, G, source_nodes):
        # transmissions is a dict: {sender_node_id: transmission_count}
        # The adversary only hears nodes within its 80m communication range (its graph neighbors)
        neighbors = list(G.neighbors(self.current_node))
        
        heard_senders = {node: count for node, count in transmissions.items() if node in neighbors}

        if heard_senders:
            # Traffic Analysis: move to the node that transmitted the most packets
            next_node = max(heard_senders, key=heard_senders.get)
            self.current_node = next_node
            self.hop_count += 1

            if self.current_node in source_nodes:
                self.is_captured = True


def random_walk(G, start_node, steps):
    current_node = start_node
    path = [current_node]
    visited = {current_node}  # Initialized as a set for O(1) lookup
    
    for _ in range(int(steps)):
        # List comprehension filters dead nodes and visited nodes simultaneously
        valid_neighbors = [
            n for n in G.neighbors(current_node) 
            if n not in visited and G.nodes[n]['energy'] > 0
        ]
        
        if not valid_neighbors:
            break  
            
        current_node = random.choice(valid_neighbors)
        path.append(current_node)
        visited.add(current_node)
        
    return path, current_node


def route_to_sink(G, start_node):
    """Routes the packet to the sink using randomized dynamic weights for fast multipath."""
    alive_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'sink' or G.nodes[n]['energy'] > 0]
    G_alive = G.subgraph(alive_nodes)
    
    if start_node not in G_alive:
        return []
        
    # Dynamic weight function: Multiplies actual distance by a random factor between 1.0 and 1.5
    def randomized_weight(u, v, edge_data):
        return edge_data['weight'] * random.uniform(1.0, 1.5)
        
    try:
        # Uses standard Dijkstra but evaluates edges using the randomized weight
        return nx.shortest_path(G_alive, source=start_node, target="SINK", weight=randomized_weight)
    except nx.NetworkXNoPath:
        return []
    

def deduct_energy(G, path):
    """Calculates First Order Radio Energy based on physical Euclidean distance."""
    if len(path) < 2:
        return

    for i in range(len(path) - 1):
        sender = path[i]
        receiver = path[i+1]
        
        try:
            distance = G[sender][receiver]['weight']
        except KeyError:
            distance = 50.0 
            
        tx_energy = (E_ELEC * PACKET_SIZE) + (E_AMP * PACKET_SIZE * (distance ** 2))
        rx_energy = E_ELEC * PACKET_SIZE
        
        if G.nodes[sender]['type'] != 'sink':
            G.nodes[sender]['energy'] -= tx_energy
            
        if G.nodes[receiver]['type'] != 'sink':
            G.nodes[receiver]['energy'] -= rx_energy



def get_fixed_sources(G, num_sources=4):
    """
    Selects 4 source nodes based on the Scenario All Directions (SAD) geometry.
    Nodes are placed N, S, E, W at approximately 25 hops from the sink.
    """
    # Cast the generator to a dictionary
    hop_lengths = dict(nx.single_target_shortest_path_length(G, "SINK"))
    
    candidates = {'N': [], 'S': [], 'E': [], 'W': []}
    
    for n in G.nodes():
        if G.nodes[n]['type'] == 'sensor' and n in hop_lengths:
            hops = hop_lengths[n]
            
            # Filter nodes that are near the 25-hop target distance
            if 20 <= hops <= 30:
                x, y = G.nodes[n]['pos']
                
                # Calculate angle relative to the sink at (1000, 1000)
                dx = x - 1000.0
                dy = y - 1000.0
                angle = math.degrees(math.atan2(dy, dx))
                
                # Group by geometric quadrant and store deviation from exactly 25 hops
                if 45 <= angle < 135:
                    candidates['N'].append((n, abs(hops - 25)))
                elif -135 <= angle < -45:
                    candidates['S'].append((n, abs(hops - 25)))
                elif -45 <= angle < 45:
                    candidates['E'].append((n, abs(hops - 25)))
                else:
                    candidates['W'].append((n, abs(hops - 25)))
    
    sources = []
    # Select the node in each quadrant that is closest to exactly 25 hops
    for direction in ['N', 'S', 'E', 'W']:
        if candidates[direction]:
            best_node = sorted(candidates[direction], key=lambda x: x[1])[0][0]
            sources.append(best_node)
            
    return sources

def run_simulation():
    print("Initializing network...")
    G = create_wsn_graph(seed=42)
    
    print("Locating connected source nodes...")
    fixed_sources = get_fixed_sources(G, num_sources=4)
    print(f"Selected Source Nodes: {fixed_sources}")
    
    dead_nodes = 0
    rounds_survived = 0
    
    # --- CSV Setup ---
    csv_filename = "baseline_results.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Hotspot_Energy_Ratio", "Coverage_Ratio", "k", "f"])
        
    print(f"Starting simulation loop. Logging to {csv_filename}...")

    
    
    for current_round in range(MAX_ROUNDS):
        packets_delivered_this_round = 0
        
        # --- Baseline State Logging ---
        all_sensor_energies = sorted([max(0, G.nodes[n]['energy']) for n in G.nodes() if G.nodes[n]['type'] == 'sensor'])
        weakest_10_avg = sum(all_sensor_energies[:10]) / 10
        energy_ratio = weakest_10_avg / INITIAL_ENERGY
        coverage_ratio = (NUM_NODES - dead_nodes) / NUM_NODES
        
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_round, round(energy_ratio, 4), round(coverage_ratio, 4), K_PHANTOM_HOPS, FAKE_TRAFFIC_RATIO])

        # Every round, EACH fixed source generates a packet
        # [Keep the rest of your routing loop unchanged below this line]
        # --- Routing Logic ---
        for source_node in fixed_sources:
            if G.nodes[source_node]['energy'] <= 0:
                continue 
                
            phantom_path, phantom_node = random_walk(G, source_node, K_PHANTOM_HOPS)
            sink_path = route_to_sink(G, phantom_node)
            
            if not sink_path:
                continue 
                
            full_path = phantom_path[:-1] + sink_path
            
            if full_path:
                deduct_energy(G, full_path)
                packets_delivered_this_round += 1
                
            if random.random() < FAKE_TRAFFIC_RATIO:
                active_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'sensor' and G.nodes[n]['energy'] > 0]
                if active_nodes:
                    fake_source = random.choice(active_nodes)
                    fake_path = route_to_sink(G, fake_source)
                    if fake_path:
                        deduct_energy(G, fake_path)
        
        # --- Network State Check ---
        current_dead = sum(1 for n in G.nodes() if G.nodes[n]['type'] == 'sensor' and G.nodes[n]['energy'] <= 0)
        if current_dead > dead_nodes:
            dead_nodes = current_dead
            
        rounds_survived = current_round
        
        # --- Termination Conditions ---
        if packets_delivered_this_round == 0:
            print(f"Network Partitioned at round {current_round}. Sink is isolated from all active sources.")
            break
            
        if dead_nodes >= 0.2 * NUM_NODES:
            print(f"Critical failure threshold (20%) reached at round {current_round}.")
            break


if __name__ == "__main__":
    run_simulation()