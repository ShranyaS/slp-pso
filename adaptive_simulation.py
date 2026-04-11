import networkx as nx
import random
from baseline_network import create_wsn_graph, NUM_NODES, INITIAL_ENERGY
from pso_optimizer import run_pso
import itertools
import csv
import math
import networkx as nx
import numpy as np

# Simulation Parameters
MAX_ROUNDS = 3000
# TX_ENERGY = 0.005  
# RX_ENERGY = 0.005  
PSO_UPDATE_INTERVAL = 1  # Recalculate k and f every 10 rounds
E_ELEC = 50e-9      # 50 nJ/bit for running transmitter/receiver circuitry
E_AMP = 100e-12     # 100 pJ/bit/m^2 for the transmit amplifier
PACKET_SIZE = 2000  # bits per packet
DUMMY_ROTATION_INTERVAL = 50


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

def random_walk(G, start_node, steps, directed_steps=3):
    current_node = start_node
    path = [current_node]
    visited = {current_node}
    
    # Get the physical coordinates of the source node
    start_pos = np.array(G.nodes[start_node]['pos'])
    
    for i in range(int(steps)):
        valid_neighbors = [
            n for n in G.neighbors(current_node) 
            if n not in visited and G.nodes[n]['energy'] > 0
        ]
        
        if not valid_neighbors:
            break  
            
        if i < directed_steps:
            # Phase 1: Directed Walk (Move outward from source)
            current_dist = np.linalg.norm(np.array(G.nodes[current_node]['pos']) - start_pos)
            outward_neighbors = [
                n for n in valid_neighbors 
                if np.linalg.norm(np.array(G.nodes[n]['pos']) - start_pos) > current_dist
            ]
            
            # If an outward step is possible, take it. Otherwise, fall back to random.
            if outward_neighbors:
                current_node = random.choice(outward_neighbors)
            else:
                current_node = random.choice(valid_neighbors)
        else:
            # Phase 2: Pure Random Walk
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


def run_adaptive_simulation(seed_val=42):
    random.seed(seed_val)
    print(f"Initializing adaptive network with seed {seed_val}...")
    G = create_wsn_graph(seed=seed_val)
    
    fixed_sources = get_fixed_sources(G, num_sources=4)
    print(f"Selected Source Nodes: {fixed_sources}")
    
    dead_nodes = 0
    rounds_survived = 0
    
    current_k = 8
    current_f = 0.5
    current_dummy_sources = []
    
    csv_filename = f"results/adaptive_results_{seed_val}.csv"
    # Log to CSV using the normalized plot variable, NOT the PSO sum
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Hotspot_Energy_Ratio", "Coverage_Ratio", "k", "f"])
        
    print(f"Starting adaptive simulation loop. Logging to {csv_filename}...")

    hunter = Adversary(start_node="SINK")
    total_captures = 0
    
    for current_round in range(MAX_ROUNDS):
        packets_delivered = 0
        round_transmissions = {}
        
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
            alive_sensor_energies = sorted([G.nodes[n]['energy'] for n in G.nodes() if G.nodes[n]['type'] == 'sensor' and G.nodes[n]['energy'] > 0])
            
            if alive_sensor_energies:
                # Average of the 10 most depleted nodes
                min_energy_avg = sum(alive_sensor_energies[:10]) / 10.0
                
                # Average of the top 10% healthiest nodes
                top_count = max(1, int(len(alive_sensor_energies) * 0.10))
                max_energy_avg = sum(alive_sensor_energies[-top_count:]) / top_count
                
                # Calculate the gap (0.0 to 1.0). 0 means perfectly balanced, 1 means extreme imbalance.
                energy_gap = (max_energy_avg - min_energy_avg) / INITIAL_ENERGY
            else:
                energy_gap = 1.0

            coverage_ratio = (NUM_NODES - dead_nodes) / NUM_NODES
            
            # Pass the gap instead of the absolute ratio
            current_k, current_f = run_pso(energy_gap, coverage_ratio)
            
            # Enforce stricter minimum privacy thresholds
            # Allow deeper late-game energy savings
            current_k = max(5, current_k)  
            current_f = max(0.2, current_f)

            
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

            for i in range(len(full_path) - 1):
                sender = full_path[i]
                round_transmissions[sender] = round_transmissions.get(sender, 0) + 1
                
            # Targeted Fake Traffic (Decoy Scheme)
            if random.random() < current_f and current_dummy_sources:
                fake_source = random.choice(current_dummy_sources)
                if G.nodes[fake_source]['energy'] > 0:
                    fake_path = route_to_sink(G, fake_source)
                    if fake_path:
                        deduct_energy(G, fake_path)
                        for i in range(len(fake_path) - 1):
                            sender = fake_path[i]
                            round_transmissions[sender] = round_transmissions.get(sender, 0) + 1
        
        # --- Adversary Step ---
        hunter.step(round_transmissions, G, fixed_sources)
        if hunter.is_captured:
            total_captures += 1
            print(f"Source Node {hunter.current_node} CAPTURED at Round {current_round}! Safety Period: {hunter.hop_count} hops. Total Captures: {total_captures}")
            hunter = Adversary(start_node="SINK")  # Reset to sink
                
        
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
    print(f"Total Captures (Capture Ratio metric): {total_captures}")
    print(f"Total Dead Nodes: {dead_nodes}")
    return rounds_survived, total_captures, dead_nodes


if __name__ == "__main__":
    run_adaptive_simulation()