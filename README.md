# Adaptive Source Location Privacy in WSNs using PSO

## Overview
This project is a discrete-event simulation framework designed to evaluate Source Location Privacy (SLP) protocols in Wireless Sensor Networks (WSNs). It compares a static baseline SLP implementation against a proposed adaptive architecture that utilizes Particle Swarm Optimization (PSO). The simulation measures network survival, privacy preservation (capture ratio), network entropy, and Dempster-Shafer spatial conflict.

## Software, Tools, and Libraries Required
* **Environment:** Python 3.8 or higher.
* **Libraries:**
  * `networkx`: For generating and routing through the WSN topology.
  * `numpy`: For numerical operations and coordinate calculations.
  * `scipy`: Specifically `cKDTree` for highly efficient spatial neighbor calculations.
  * `pyswarms`: For executing the Particle Swarm Optimization (PSO) algorithm.
  * `matplotlib`: For rendering network topologies and plotting metric graphs.
  * `pandas`: For reading and processing the output CSV data.

You can install the required dependencies using pip:
bash
pip install networkx numpy scipy pyswarms matplotlib pandas

Network Simulation Framework

This project simulates and evaluates baseline and adaptive models for large-scale network privacy and energy behavior using Monte Carlo analysis.


### 1. Verify the Topology

To generate and visualize the 3000-node network topology:

bash
python baseline_network.py

### 2. Run a Single Simulation

Run either the baseline or adaptive model:

python baseline_simulation.py
python adaptive_simulation.py

Outputs:

Round-by-round logs in the terminal
Metrics saved as CSV files

### 3. Run Monte Carlo Simulation (Full Evaluation)
python monte_carlo.py
Executes both models across 10 parallel seeds
Uses multi-core CPU processing

Note: This may take 15–20 minutes depending on your CPU.

### 4. Generate Output Graphs
python plot_new_metrics.py

Generated Graphs:

Entropy
Spatial Conflict
Capture Ratio
Survival


Inputs
Topological Constraints (baseline_network.py)
Nodes: 3000
Area: 2000m × 2000m
Communication Range: 80m
Initial Energy: 0.5J

Seed Values
Single simulation: seed_val = 42
Monte Carlo: seeds 100–110


Optimization Bounds (pso_optimizer.py)
k ∈ [2, 12] → Phantom walk length
f ∈ [0.0, 0.8] → Fake traffic probability

Outputs
#### 1. Console Output

Displays real-time simulation updates:

Network generation status
Node capture events
Network partitions
20% mortality triggers

Monte Carlo also prints a comparative statistical table.

#### 2. CSV Files (results/ directory)

Example:

results/adaptive_results_100.csv


| Column Name          | Description                                         |
| -------------------- | --------------------------------------------------- |
| Round                | Current simulation round                            |
| Hotspot_Energy_Ratio | Remaining hotspot energy vs initial                 |
| Coverage_Ratio       | Percentage of alive nodes                           |
| k                    | Avg. phantom walk length                            |
| f                    | Avg. fake traffic probability                       |
| Total_Captures       | Total adversary captures                            |
| Network_Entropy      | Shannon entropy (traffic uniformity)                |
| Spatial_Conflict     | Avg. distance between real & dummy sources (meters) |


#### 3. Graphical Plots (.png files)

Generated in root directory:

comparison_entropy.png
comparison_conflict.png
comparison_capture_ratio.png
comparison_survival.png

These compare:

Baseline vs Adaptive models
Network survival
Privacy degradation
Anonymity effectiveness
