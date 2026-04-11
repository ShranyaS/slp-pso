import numpy as np
import pyswarms as ps

# Define Optimization Bounds
# x[0] = k (phantom hops): between 2 and 15
# x[1] = f (fake traffic ratio): between 0.0 and 1.0
# Increase max f bound to 0.8 (80% fake traffic chance)
BOUNDS = (np.array([2.0, 0.0]), np.array([12.0, 0.8]))

def fitness_function(particles, energy_gap, coverage_ratio):
    n_particles = particles.shape[0]
    costs = np.zeros(n_particles)
    
    for i in range(n_particles):
        k = particles[i, 0]
        f = particles[i, 1]
        
        # Normalize based on new max bounds
        k_norm = k / 12.0
        f_norm = f / 0.8
        U = (0.6 * k_norm) + (0.4 * f_norm)
        
        # Relax the penalty threshold to 40% drain
        if energy_gap <= 0.40:
            gap_penalty = 1.0
        else:
            gap_penalty = np.exp(-5.0 * (energy_gap - 0.40) * U)
            
        cr_penalty = 1.0
        if coverage_ratio < 0.85:
            cr_penalty = np.exp(-20.0 * (0.85 - coverage_ratio))
            
        fitness = U * gap_penalty * cr_penalty
        costs[i] = -fitness
        
    return costs


def run_pso(energy_gap, current_coverage_ratio):
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=15, dimensions=2, options=options, bounds=BOUNDS)
    
    # Pass energy_gap instead of energy_ratio
    cost, pos = optimizer.optimize(fitness_function, iters=20, verbose=False, 
                                   energy_gap=energy_gap, 
                                   coverage_ratio=current_coverage_ratio)
    
    optimal_k = int(np.round(pos[0]))
    optimal_f = float(pos[1])
    
    return optimal_k, optimal_f

# Test the optimizer independently
if __name__ == "__main__":
    print("Testing PSO Optimizer with 100% Battery and 100% Nodes...")
    k, f = run_pso(1.0, 1.0)
    print(f"Optimal Result -> k: {k}, f: {f:.2f}")
    
    print("\nTesting PSO Optimizer with 30% Battery and 80% Nodes...")
    k, f = run_pso(0.3, 0.8)
    print(f"Optimal Result -> k: {k}, f: {f:.2f}")