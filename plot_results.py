import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_monte_carlo_plots(start_seed=100, num_runs=10):
    NUM_NODES = 3000
    seeds = range(start_seed, start_seed + num_runs)
    
    df_base_list = []
    df_adapt_list = []

    # 1. Load Data
    for seed in seeds:
        try:
            df_base_list.append(pd.read_csv(f'results/baseline_results_{seed}.csv'))
            df_adapt_list.append(pd.read_csv(f'results/adaptive_results_{seed}.csv'))
        except FileNotFoundError:
            print(f"Skipping seed {seed}: File not found.")

    if not df_base_list or not df_adapt_list:
        print("Error: Could not load data. Check the results folder.")
        return

    # 2. Aggregate Data (Average across all seeds by Round)
    df_base = pd.concat(df_base_list).groupby('Round').mean().reset_index()
    df_adapt = pd.concat(df_adapt_list).groupby('Round').mean().reset_index()

    df_base['Alive_Nodes'] = (df_base['Coverage_Ratio'] * NUM_NODES).round()
    df_adapt['Alive_Nodes'] = (df_adapt['Coverage_Ratio'] * NUM_NODES).round()

    # --- Plot 1: Network Survival ---
    plt.figure(figsize=(10, 6))
    plt.plot(df_base['Round'], df_base['Alive_Nodes'], label='Baseline', color='red', linewidth=2)
    plt.plot(df_adapt['Round'], df_adapt['Alive_Nodes'], label='Adaptive', color='blue', linewidth=2)
    plt.xlabel('Simulation Round')
    plt.ylabel('Alive Sensor Nodes')
    plt.title('Monte Carlo Average: Network Survival')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('comparison_survival_mc.png')
    plt.close()

    # --- Plot 2: Hotspot Energy Drain ---
    plt.figure(figsize=(10, 6))
    plt.plot(df_base['Round'], df_base['Hotspot_Energy_Ratio'], label='Baseline', color='red', linewidth=2)
    plt.plot(df_adapt['Round'], df_adapt['Hotspot_Energy_Ratio'], label='Adaptive', color='blue', linewidth=2)
    plt.xlabel('Simulation Round')
    plt.ylabel('Global Hotspot Energy Ratio')
    plt.title('Monte Carlo Average: Hotspot Drain')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('comparison_energy_mc.png')
    plt.close()

    # --- Plot 3: Privacy Parameters (k and f) ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Simulation Round')
    ax1.set_ylabel('Average k (Phantom Hops)', color=color)
    ax1.plot(df_adapt['Round'], df_adapt['k'], label='Adaptive Avg k', color=color, linewidth=2)
    ax1.axhline(y=8, color='lightblue', linestyle='--', label='Baseline k=8')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:green'
    ax2.set_ylabel('Average f (Fake Traffic Ratio)', color=color)
    ax2.plot(df_adapt['Round'], df_adapt['f'], label='Adaptive Avg f', color=color, linewidth=2)
    ax2.axhline(y=0.5, color='lightgreen', linestyle='--', label='Baseline f=0.5')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Monte Carlo Average: Privacy Parameters')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig('comparison_privacy_mc.png')
    plt.close()

    # --- Plot 4: Adversary Captures (Conditional) ---
    # Only executes if you decide to add Total_Captures to your CSVs later
    if 'Total_Captures' in df_base.columns and 'Total_Captures' in df_adapt.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df_base['Round'], df_base['Total_Captures'], label='Baseline Captures', color='red', linewidth=2)
        plt.plot(df_adapt['Round'], df_adapt['Total_Captures'], label='Adaptive Captures', color='blue', linewidth=2)
        plt.xlabel('Simulation Round')
        plt.ylabel('Cumulative Source Captures')
        plt.title('Monte Carlo Average: Adversary Success')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('comparison_adversary_mc.png')
        plt.close()

    print("Successfully generated Monte Carlo plots in the root directory.")

if __name__ == "__main__":
    generate_monte_carlo_plots()