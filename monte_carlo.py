import concurrent.futures
import time
import statistics
from baseline_simulation import run_simulation as run_baseline
from adaptive_simulation import run_adaptive_simulation as run_adaptive

# Number of parallel simulations to run
NUM_RUNS = 100

def run_single_iteration(seed_val):
    # Runs both simulations for a specific seed
    base_rounds, base_captures, base_dead = run_baseline(seed_val=seed_val)
    adapt_rounds, adapt_captures, adapt_dead = run_adaptive(seed_val=seed_val)
    
    return {
        'seed': seed_val,
        'base': (base_rounds, base_captures, base_dead),
        'adapt': (adapt_rounds, adapt_captures, adapt_dead)
    }

def main():
    print(f"Starting Monte Carlo simulation with {NUM_RUNS} iterations in parallel...")
    start_time = time.time()
    
    # Generate a list of unique seeds
    seeds = list(range(100, 100 + NUM_RUNS))
    results = []

    # Execute in parallel using all available CPU cores
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for res in executor.map(run_single_iteration, seeds):
            results.append(res)
            print(f"Completed iteration for Seed: {res['seed']} | Progress: {len(results)}/{NUM_RUNS}")

    # Aggregate Data
    base_r, base_c, base_d = zip(*[r['base'] for r in results])
    adapt_r, adapt_c, adapt_d = zip(*[r['adapt'] for r in results])

    end_time = time.time()
    
    # Output the Averaged Metrics
    print("\n" + "="*50)
    print("MONTE CARLO AVERAGES ({} runs)".format(NUM_RUNS))
    print("="*50)
    print(f"Execution Time: {round(end_time - start_time, 2)} seconds")
    print("-" * 50)
    print(f"{'Metric':<20} | {'Baseline':<12} | {'Adaptive':<12}")
    print("-" * 50)
    print(f"{'Avg Rounds Survived':<20} | {statistics.mean(base_r):<12.1f} | {statistics.mean(adapt_r):<12.1f}")
    print(f"{'Avg Captures':<20} | {statistics.mean(base_c):<12.1f} | {statistics.mean(adapt_c):<12.1f}")
    print(f"{'Avg Dead Nodes':<20} | {statistics.mean(base_d):<12.1f} | {statistics.mean(adapt_d):<12.1f}")
    print("=" * 50)

if __name__ == "__main__":
    main()