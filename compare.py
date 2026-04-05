import random
from baseline_simulation import run_simulation as run_baseline
from adaptive_simulation import run_adaptive_simulation as run_adaptive

def run_comparison(seed_value=105):
    print(f"--- Running Baseline Model (Seed: {seed_value}) ---")
    random.seed(seed_value)
    base_rounds, base_captures, base_dead = run_baseline()

    print(f"\n--- Running Adaptive Model (Seed: {seed_value}) ---")
    random.seed(seed_value)
    adapt_rounds, adapt_captures, adapt_dead = run_adaptive()

    print("\n" + "="*45)
    print(f"{'Metric':<20} | {'Baseline':<10} | {'Adaptive':<10}")
    print("-" * 45)
    print(f"{'Rounds Survived':<20} | {base_rounds:<10} | {adapt_rounds:<10}")
    print(f"{'Total Captures':<20} | {base_captures:<10} | {adapt_captures:<10}")
    print(f"{'Dead Nodes':<20} | {base_dead:<10} | {adapt_dead:<10}")
    print("=" * 45)

if __name__ == "__main__":
    run_comparison()