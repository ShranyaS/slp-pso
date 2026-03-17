import pandas as pd
import matplotlib.pyplot as plt

def generate_comparative_plots():
    # Load data
    try:
        df_base = pd.read_csv('baseline_results.csv')
        df_adapt = pd.read_csv('adaptive_results.csv')
    except FileNotFoundError:
        print("Error: Ensure both baseline_results.csv and adaptive_results.csv exist.")
        return

    NUM_NODES = 3000

    # Calculate exact alive nodes
    df_base['Alive_Nodes'] = (df_base['Coverage_Ratio'] * NUM_NODES).round()
    df_adapt['Alive_Nodes'] = (df_adapt['Coverage_Ratio'] * NUM_NODES).round()

    # --- Plot 1: Hotspot Energy Ratio over Time ---
    plt.figure(figsize=(10, 6))
    plt.plot(df_base['Round'], df_base['Hotspot_Energy_Ratio'], label='Baseline (Static)', color='red', linewidth=2)
    plt.plot(df_adapt['Round'], df_adapt['Hotspot_Energy_Ratio'], label='Adaptive (PSO)', color='blue', linewidth=2)
    plt.xlabel('Simulation Round')
    plt.ylabel('Hotspot Energy Ratio')
    plt.title('Hotspot Energy Depletion Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('comparison_energy.png')

    # --- Plot 2: Network Survival (Alive Nodes) ---
    plt.figure(figsize=(10, 6))
    plt.plot(df_base['Round'], df_base['Alive_Nodes'], label='Baseline (Static)', color='red', linewidth=2)
    plt.plot(df_adapt['Round'], df_adapt['Alive_Nodes'], label='Adaptive (PSO)', color='blue', linewidth=2)
    plt.xlabel('Simulation Round')
    plt.ylabel('Alive Sensor Nodes')
    plt.title('Network Survival and Node Mortality')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('comparison_survival.png')

    # --- Plot 3: Privacy Parameters (k and f) ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Simulation Round')
    ax1.set_ylabel('k (Phantom Hops)', color=color)
    ax1.plot(df_adapt['Round'], df_adapt['k'], label='Adaptive k', color=color, linewidth=2)
    ax1.axhline(y=8, color='lightblue', linestyle='--', label='Baseline k (Static=8)')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:green'
    ax2.set_ylabel('f (Fake Traffic Ratio)', color=color)
    ax2.plot(df_adapt['Round'], df_adapt['f'], label='Adaptive f', color=color, linewidth=2)
    ax2.axhline(y=0.5, color='lightgreen', linestyle='--', label='Baseline f (Static=0.5)')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Privacy Parameters: Adaptive vs Baseline')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.grid(True)
    plt.savefig('comparison_privacy.png')

    print("Plots generated successfully: comparison_energy.png, comparison_survival.png, comparison_privacy.png")

if __name__ == "__main__":
    generate_comparative_plots()