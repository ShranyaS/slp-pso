import networkx as nx
import random
from baseline_network import create_wsn_graph, NUM_NODES, INITIAL_ENERGY
from pso_optimizer import run_pso
import itertools
import csv
import math
import networkx as nx


# Simulation Parameters
MAX_ROUNDS = 3000
# TX_ENERGY = 0.005  
# RX_ENERGY = 0.005  
PSO_UPDATE_INTERVAL = 1  # Recalculate k and f every 10 rounds
E_ELEC = 50e-9      # 50 nJ/bit for running transmitter/receiver circuitry
E_AMP = 100e-12     # 100 pJ/bit/m^2 for the transmit amplifier
PACKET_SIZE = 2000  # bits per packet
DUMMY_ROTATION_INTERVAL = 50


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


def run_adaptive_simulation():
    print("Initializing adaptive network...")
    G = create_wsn_graph(seed=42)
    fixed_sources = get_fixed_sources(G, num_sources=4)
    print(f"Selected Source Nodes: {fixed_sources}")
    
    dead_nodes = 0
    rounds_survived = 0
    
    current_k = 8
    current_f = 0.5
    current_dummy_sources = []
    
    csv_filename = "adaptive_results.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Hotspot_Energy_Ratio", "Coverage_Ratio", "k", "f"])
        
    print(f"Starting adaptive simulation loop. Logging to {csv_filename}...")
    
    for current_round in range(MAX_ROUNDS):
        packets_delivered = 0
        
        # --- Decoy (Dummy Source) Rotation Logic ---
        # Rotate every 50 rounds, or if the current decoys are uninitialized/dead
        if (current_round % DUMMY_ROTATION_INTERVAL == 0 or 
            not current_dummy_sources or 
            any(G.nodes[d]['energy'] <= 0 for d in current_dummy_sources)):
            
            active_sensors = [n for n in G.nodes() if G.nodes[n]['type'] == 'sensor' 
                              and G.nodes[n]['energy'] > 0 
                              and n not in fixed_sources]
            if active_sensors:
                # Select 2 distinct decoy nodes
                current_dummy_sources = random.sample(active_sensors, min(2, len(active_sensors)))

        # 1. State Evaluation & PSO Update
        if current_round % PSO_UPDATE_INTERVAL == 0:
            all_sensor_energies = sorted([max(0, G.nodes[n]['energy']) for n in G.nodes() if G.nodes[n]['type'] == 'sensor'])
            weakest_10_avg = sum(all_sensor_energies[:10]) / INITIAL_ENERGY
            energy_ratio = weakest_10_avg
            coverage_ratio = (NUM_NODES - dead_nodes) / NUM_NODES
            
            current_k, current_f = run_pso(energy_ratio, coverage_ratio)
            
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([current_round, round(energy_ratio, 4), round(coverage_ratio, 4), current_k, round(current_f, 4)])
            
        # 2. Routing Phase
        for source_node in fixed_sources:
            if G.nodes[source_node]['energy'] <= 0:
                continue
                
            # Real Traffic
            phantom_path, phantom_node = random_walk(G, source_node, current_k)
            sink_path = route_to_sink(G, phantom_node)
            
            if not sink_path:
                continue
                
            full_path = phantom_path[:-1] + sink_path
            deduct_energy(G, full_path)
            packets_delivered += 1
                
            # Targeted Fake Traffic (Decoy Scheme)
            if random.random() < current_f and current_dummy_sources:
                fake_source = random.choice(current_dummy_sources)
                if G.nodes[fake_source]['energy'] > 0:
                    fake_path = route_to_sink(G, fake_source)
                    if fake_path:
                        deduct_energy(G, fake_path)
        
        # 3. Network State Check
        current_dead = sum(1 for n in G.nodes() if G.nodes[n]['type'] == 'sensor' and G.nodes[n]['energy'] <= 0)
        if current_dead > dead_nodes:
            dead_nodes = current_dead
            
        rounds_survived = current_round
        
        # Termination Conditions
        if packets_delivered == 0:
            print(f"Network Partitioned at round {current_round}. Sink is isolated.")
            break
            
        if dead_nodes >= 0.2 * NUM_NODES:
            print(f"Critical failure threshold (20%) reached at round {current_round}.")
            break

    print(f"\n--- Adaptive Simulation Complete ---")
    print(f"Total Rounds Survived: {rounds_survived}")
    print(f"Total Dead Nodes: {dead_nodes}")


if __name__ == "__main__":
    run_adaptive_simulation()