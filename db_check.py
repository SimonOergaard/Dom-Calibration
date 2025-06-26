import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import os

def plot_domz_vs_charge(file_path, output_dir, table_name="SRTInIcePulses", label="RD", target_x=106.94, target_y=27.09, tolerance=3.0):
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    con = sqlite3.connect(file_path)
    df = pd.read_sql_query(f"SELECT dom_x, dom_y, dom_z, charge FROM {table_name}", con)
    con.close()

    # Apply box filter
    df_filtered = df[
        (df["dom_x"] >= target_x - tolerance) & (df["dom_x"] <= target_x + tolerance) &
        (df["dom_y"] >= target_y - tolerance) & (df["dom_y"] <= target_y + tolerance)
    ]

    # Group by dom_z and sum charge
    df_grouped = df_filtered.groupby("dom_z", as_index=False)["charge"].sum()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df_grouped["dom_z"], df_grouped["charge"], alpha=0.7)
    plt.xlabel("DOM Z position [m]", fontsize=16)
    plt.ylabel("Total Charge", fontsize=16)
    plt.title(f"Charge vs DOM Z (x={target_x}±{tolerance}, y={target_y}±{tolerance})", fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    filename = f"domz_vs_charge_stopped_muon_sorted_{label}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

    print(f"Plot saved to {os.path.join(output_dir, filename)}")
if __name__ == "__main__":
    # Example usage
    file_path = "/groups/icecube/simon/GNN/workspace/storage/Training/stopped_through_classification/train_model_without_configs/muon_stopped_sorted.db"
    output_dir = "/groups/icecube/simon/GNN/workspace/plots"
    plot_domz_vs_charge(file_path, output_dir)