import numpy as np
import pyswarms as ps

# Define Optimization Bounds
# x[0] = k (phantom hops): between 2 and 15
# x[1] = f (fake traffic ratio): between 0.0 and 1.0
BOUNDS = (np.array([2.0, 0.0]), np.array([10.0, 0.5]))

def fitness_function(particles, energy_ratio, coverage_ratio):
    n_particles = particles.shape[0]
    costs = np.zeros(n_particles)
    
    for i in range(n_particles):
        k = particles[i, 0]
        f = particles[i, 1]
        
        # 1. Base Utility (Normalize k based on max 12.0, f based on max 0.5)
        k_norm = k / 12.0
        f_norm = f / 0.5
        
        # Utility Weights (60% importance on k, 40% on f)
        U = (0.6 * k_norm) + (0.4 * f_norm)
        
        # 2. Multiplicative Constraint Penalty
        # The particle's normalized cost must not exceed the available energy ratio
        particle_cost = U 
        
        if particle_cost <= energy_ratio:
            energy_penalty = 1.0
        else:
            # Exponentially penalize particles that demand more energy than what is safely available
            energy_penalty = np.exp(-10.0 * (particle_cost - energy_ratio))
            
        # 3. Coverage Penalty
        cr_penalty = 1.0
        if coverage_ratio < 0.85:
            cr_penalty = np.exp(-20.0 * (0.85 - coverage_ratio))
            
        # Calculate final MPM fitness
        fitness = U * energy_penalty * cr_penalty
        
        # PySwarms minimizes, so return negative fitness to maximize
        costs[i] = -fitness
        
    return costs


def run_pso(current_energy_ratio, current_coverage_ratio):
    """
    Executes the PSO algorithm to find the optimal k and f.
    """
    # PSO Hyperparameters: c1 (cognitive), c2 (social), w (inertia)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    
    # Initialize the swarm with 15 particles
    optimizer = ps.single.GlobalBestPSO(n_particles=15, dimensions=2, options=options, bounds=BOUNDS)
    
    # Run optimization for 20 iterations
    # **kwargs passes the current network state to the fitness function
    cost, pos = optimizer.optimize(fitness_function, iters=20, verbose=False, 
                                   energy_ratio=current_energy_ratio, 
                                   coverage_ratio=current_coverage_ratio)
    
    # Extract optimized variables
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