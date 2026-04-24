import pandas as pd
import matplotlib.pyplot as plt

def plot_entropy_and_conflict(seed_val=100):
    # Load the CSV data
    try:
        base_df = pd.read_csv(f"results/baseline_results_{seed_val}.csv")
        adapt_df = pd.read_csv(f"results/adaptive_results_{seed_val}.csv")
    except FileNotFoundError:
        print(f"Results for seed {seed_val} not found. Run the simulation first.")
        return

    # 1. Plot Network Entropy
    plt.figure(figsize=(10, 6))
    plt.plot(base_df['Round'], base_df['Network_Entropy'], label='Baseline', color='red', alpha=0.7)
    plt.plot(adapt_df['Round'], adapt_df['Network_Entropy'], label='Adaptive (PSO)', color='darkblue', alpha=0.9)
    plt.title('Network Entropy vs. Simulation Round')
    plt.xlabel('Simulation Round')
    plt.ylabel('Shannon Entropy (bits)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('comparison_entropy.png')
    plt.close()  # <-- Closes the figure instead of showing it

    # 2. Plot Spatial Conflict (Evidence Theory)
    plt.figure(figsize=(10, 6))
    plt.plot(base_df['Round'], base_df['Spatial_Conflict'], label='Baseline', color='red', alpha=0.7)
    plt.plot(adapt_df['Round'], adapt_df['Spatial_Conflict'], label='Adaptive (PSO)', color='darkblue', alpha=0.9)
    plt.title("Adversary's Evidence Conflict vs. Simulation Round")
    plt.xlabel('Simulation Round')
    plt.ylabel('Spatial Conflict Distance (meters)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('comparison_conflict.png')
    plt.close()  # <-- Closes the figure instead of showing it
    
    print("Graphs successfully generated and saved to the directory.")

if __name__ == "__main__":
    plot_entropy_and_conflict(seed_val=100)
    