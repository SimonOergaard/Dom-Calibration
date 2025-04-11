import numpy as np
import pandas as pd
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor
import psutil
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
def print_memory_usage(message=""):
    """Prints the total memory usage across all processes related to the script."""
    current_process = psutil.Process(os.getpid())
    all_processes = [current_process] + current_process.children(recursive=True)
    total_memory = sum(proc.memory_info().rss for proc in all_processes if proc.is_running()) / (1024 * 1024)
    print(f"[Total Memory Usage] {message}: {total_memory:.2f} MB")

def process_events_with_charge_and_launches(pulses_df):
    """
    Computes the total charge per DOM.
    """
    print_memory_usage("Before processing events")

    # Compute total charge per DOM
    total_charge_per_dom = pulses_df.groupby(["dom_x", "dom_y", "dom_z"])["charge"].sum().reset_index()

    print_memory_usage("After processing events")
    return total_charge_per_dom


def classify_dom_strings(dom_x, dom_y):
    """
    Classifies DOMs into strings based on their (x, y) coordinates.

    Args:
        dom_x (float): x-coordinate of the DOM.
        dom_y (float): y-coordinate of the DOM.

    Returns:
        str: "string_79" if near (31.25, -72.93), "string_80" if near (72.37, -66.6), else None.
    """
    # Define known string positions with ±5m tolerance
    string_79_coords = (31.25, -72.93)
    string_81_coords = (41.6, 35.49)
    string_82_coords = (106.94,27.09)
    string_80_coords = (72.37, -66.6)
    string_83_coords = (113.19, -60.47)
    string_84_coords = (57.2,-105.52)
    string_85_coords = (-9.68 ,-79.5)
    string_86_coords = (-10.97, 6.72)
    tolerance = 1.0

    if np.isclose(dom_x, string_79_coords[0], atol=tolerance) and np.isclose(dom_y, string_79_coords[1], atol=tolerance):
        return "string_79"
    elif np.isclose(dom_x, string_80_coords[0], atol=tolerance) and np.isclose(dom_y, string_80_coords[1], atol=tolerance):
        return "string_80"
    elif np.isclose(dom_x, string_81_coords[0], atol=tolerance) and np.isclose(dom_y, string_81_coords[1], atol=tolerance):
        return "string_81"
    elif np.isclose(dom_x, string_83_coords[0], atol=tolerance) and np.isclose(dom_y, string_83_coords[1], atol=tolerance):
        return "string_83"
    elif np.isclose(dom_x, string_84_coords[0], atol=tolerance) and np.isclose(dom_y, string_84_coords[1], atol=tolerance):
        return "string_84"
    elif np.isclose(dom_x, string_85_coords[0], atol=tolerance) and np.isclose(dom_y, string_85_coords[1], atol=tolerance):
        return "string_85"
    elif np.isclose(dom_x, string_86_coords[0], atol=tolerance) and np.isclose(dom_y, string_86_coords[1], atol=tolerance):
        return "string_86"
    elif np.isclose(dom_x,string_82_coords[0],atol=tolerance) and np.isclose(dom_y,string_82_coords[1],atol=tolerance):
        return "string_82"
    else:
        return None  # Unmatched DOM

def process_chunk(pulses_chunk):
    """Process chunk of pulses and return total charge per DOM."""
    return process_events_with_charge_and_launches(pulses_chunk)

def plot_scatter_metrics(aggregated_metrics, output_dir):
    """
    Generates scatter plots for total charge vs. depth (dom_z), grouped by reconstructed strings and marked by RDE.

    Args:
        aggregated_metrics (dict): Dictionary containing:
            - "dom_x": List of DOM x-coordinates.
            - "dom_y": List of DOM y-coordinates.
            - "dom_z": List of DOM depths.
            - "total_charge": List of total charge values.
            - "rde": List of RDE values.
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    STRING_COLORS = {
        "string_79": "blue",
        "string_80": "green",
        "string_84": "orange",  # New String
        "string_85": "red"  # New String
    }
    # Convert dictionary to DataFrame for easy processing
    df = pd.DataFrame(aggregated_metrics)

    # Ensure each DOM position is unique and assigned a single RDE
    df_unique = df.drop_duplicates(subset=["dom_x", "dom_y", "dom_z"], keep="first")

    # Classify DOMs into strings
    df_unique["string_label"] = df_unique.apply(lambda row: classify_dom_strings(row["dom_x"], row["dom_y"]), axis=1)

    # Filter out DOMs that do not belong to string_79 or string_80
    df_filtered = df_unique[df_unique["string_label"].notna()].copy()

    # Assign markers based on RDE values
    df_filtered["marker"] = df_filtered["rde"].apply(lambda r: "^" if r > 1.0 else "o")

    # Assign colors dynamically based on string labels
    df_filtered["color"] = df_filtered["string_label"].map(STRING_COLORS).fillna("black")  # Default to black if unknown

    # Plot total charge per depth for all strings
    plt.figure(figsize=(12, 8))

    for _, row in df_filtered.iterrows():
        plt.scatter(row["dom_z"], row["total_charge"], color=row["color"], alpha=0.7, s=80, marker=row["marker"])

    plt.xlabel("Depth (m)")
    plt.ylabel("Total Charge")
    plt.title("Total Charge vs Depth (Reconstructed Strings, Marked by RDE)")
    plt.grid(axis="both", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Legend explanation for multiple strings and RDE markers
    legend_elements = []
    for string, color in STRING_COLORS.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f"{string.replace('_', ' ').title()}, RDE=1.0", markerfacecolor=color, markersize=10))
        legend_elements.append(Line2D([0], [0], marker='^', color='w', label=f"{string.replace('_', ' ').title()}, RDE > 1.0", markerfacecolor=color, markersize=10))

    plt.legend(handles=legend_elements, loc="upper right")

    # Save the plot
    plt.savefig(os.path.join(output_dir, "total_charge_vs_depth_reconstructed_rde_fixed_MC_NO_SRT.png"))
    plt.close()




def plot_predictions(predictions_df, output_dir):
    """
    Plots:
    1. Histogram of position_z_pred
    2. 2D Histogram (zenith_pred vs energy_pred)

    Args:
        predictions_df (pd.DataFrame): DataFrame containing predictions.
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- PLOT 1: Histogram of position_z_pred ---
    plt.figure(figsize=(8, 6))
    plt.hist(predictions_df["position_z"], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel("Position Z")
    plt.ylabel("Frequency")
    plt.title("Distribution of  Z Position")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/position_z_distribution_MC_NO_SRT.png")
    plt.close()
    print("✅ Saved position_z_distribution.png")

    # --- PLOT 2: 2D Histogram (zenith_pred vs energy_pred) ---
    plt.figure(figsize=(8, 6))
    plt.hist2d(
        predictions_df["zenith"],
        predictions_df["energy"],
        bins=(50, 50),
        cmap='viridis',
        cmin=1
    )
    plt.colorbar(label="Counts")
    plt.xlabel("Predicted Zenith (rad)")
    plt.ylabel("Predicted Energy (GeV)")
    plt.ylim(0, 600)
    plt.title("2D Histogram of Predicted Zenith vs. Energy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/zenith_vs_energy_2d_hist_MC_NO_SRT.png")
    plt.close()
    print("✅ Saved zenith_vs_energy_2d_hist.png")

def plot_charge_ratio(file_path, file_path_Raw, output_dir):
    """
    Processes two datasets in parallel (MC and RD), aggregates total charge per DOM,
    computes the charge ratio (MC charge / RD charge) for each unique DOM position,
    and then plots the ratio as a function of depth (dom_z). The DOMs are classified
    by string and marked according to the RDE value (from the MC dataset).
    
    Args:
        file_path (str): Path to the MC pulses database (using table SplitInIcePulsesSRT).
        file_path_Raw (str): Path to the RD pulses database (using table SplitInIcePulses).
        output_dir (str): Directory to save the generated plot.
    """


    os.makedirs(output_dir, exist_ok=True)

    # Define queries for each dataset
    pulse_query_MC = """
    SELECT dom_x, dom_y, dom_z, charge, dom_time, event_no, rde
    FROM SplitInIcePulses 
    """
    pulse_query_RD = """
    SELECT dom_x, dom_y, dom_z, charge, dom_time, event_no, rde
    FROM SplitInIcePulses
    """

    # Load data from each database
    con_MC = sqlite3.connect(file_path)
    con_RD = sqlite3.connect(file_path_Raw)
    pulses_df_MC = pd.read_sql_query(pulse_query_MC, con_MC)
    pulses_df_RD = pd.read_sql_query(pulse_query_RD, con_RD)
    con_MC.close()
    con_RD.close()

    # Process MC dataset in parallel
    mc_chunks = np.array_split(pulses_df_MC, 4)
    try:
        with ProcessPoolExecutor(max_workers=4) as executor:
            mc_results = list(executor.map(process_chunk, mc_chunks))
    except Exception as e:
        print(f"ERROR processing MC in parallel: {e}")
        return
    total_charge_MC_df = pd.concat(mc_results, ignore_index=True)
    agg_MC = total_charge_MC_df.groupby(["dom_x", "dom_y", "dom_z"])["charge"].sum().reset_index()
    agg_MC = agg_MC.merge(
        pulses_df_MC[['dom_x', 'dom_y', 'dom_z', 'rde']].drop_duplicates(),
        on=["dom_x", "dom_y", "dom_z"],
        how="left"
    )

    # Process RD dataset in parallel
    rd_chunks = np.array_split(pulses_df_RD, 4)
    try:
        with ProcessPoolExecutor(max_workers=4) as executor:
            rd_results = list(executor.map(process_chunk, rd_chunks))
    except Exception as e:
        print(f"ERROR processing RD in parallel: {e}")
        return
    total_charge_RD_df = pd.concat(rd_results, ignore_index=True)
    agg_RD = total_charge_RD_df.groupby(["dom_x", "dom_y", "dom_z"])["charge"].sum().reset_index()
    agg_RD = agg_RD.merge(
        pulses_df_RD[['dom_x', 'dom_y', 'dom_z', 'rde']].drop_duplicates(),
        on=["dom_x", "dom_y", "dom_z"],
        how="left"
    )

    # Round coordinates to avoid floating-point mismatches during merge
    for col in ["dom_x", "dom_y", "dom_z"]:
        agg_MC[col] = agg_MC[col].round(2)
        agg_RD[col] = agg_RD[col].round(2)

    # Merge aggregated MC and RD data on DOM coordinates
    merged = pd.merge(agg_MC, agg_RD, on=["dom_x", "dom_y", "dom_z"], suffixes=('_MC', '_RD'))
    # Compute charge ratio (MC/RD), with safeguard against division by zero
    merged["charge_ratio"] = merged.apply(
        lambda row: row["charge_MC"] / row["charge_RD"] if row["charge_RD"] != 0 else np.nan,
        axis=1
    )

    # Classify DOMs using your existing function (assumes function classify_dom_strings is defined)
    merged["string_label"] = merged.apply(lambda row: classify_dom_strings(row["dom_x"], row["dom_y"]), axis=1)
    merged_filtered = merged[merged["string_label"].notna()].copy()

    # Use the MC rde for marker styling: RDE > 1.0 uses "^", else "o"
    merged_filtered["marker"] = merged_filtered["rde_MC"].apply(lambda r: "^" if r > 1.0 else "o")

    # Define colors for the strings
    STRING_COLORS = {
        "string_79": "blue",
        "string_80": "green",
        "string_84": "orange",
        "string_85": "red"
    }
    merged_filtered["color"] = merged_filtered["string_label"].map(STRING_COLORS).fillna("black")

    # Create scatter plot: depth (dom_z) vs. charge ratio
    plt.figure(figsize=(12, 8))
    for _, row in merged_filtered.iterrows():
        plt.scatter(
            row["dom_z"],
            row["charge_ratio"],
            color=row["color"],
            alpha=0.7,
            s=80,
            marker=row["marker"]
        )
    plt.xlabel("Depth (m)")
    plt.ylabel("Charge Ratio (MC / RD)")
    plt.title("Charge Ratio vs. Depth (Classified by Reconstructed String)")
    plt.grid(axis="both", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Build legend elements
    legend_elements = []
    for string, color in STRING_COLORS.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                      label=f"{string.replace('_', ' ').title()}, RDE=1.0",
                                      markerfacecolor=color, markersize=10))
        legend_elements.append(Line2D([0], [0], marker='^', color='w',
                                      label=f"{string.replace('_', ' ').title()}, RDE > 1.0",
                                      markerfacecolor=color, markersize=10))
    plt.legend(handles=legend_elements, loc="upper right")

    # Save the plot
    plot_path = os.path.join(output_dir, "charge_ratio_vs_depth_NO_SRT.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved charge ratio plot to: {plot_path}")

def assign_dom_number(row, mc_df, tol=2):
    """
    For a given RD row, find the corresponding MC row based on dom_z.
    If the absolute difference in dom_z is within tol (meters),
    return the MC row's dom_number with the smallest difference.
    Otherwise, return -1.
    """
    # Filter MC rows to those with dom_z within tol
    candidates = mc_df[np.abs(mc_df['dom_z'] - row['dom_z']) <= tol]
    if not candidates.empty:
        # Pick the candidate with the smallest absolute difference in dom_z
        best_match = candidates.iloc[(np.abs(candidates['dom_z'] - row['dom_z'])).argmin()]
        return best_match['dom_number']
    else:
        return -1

def aggregate_charge_without_groupby(df, key_cols=["dom_x", "dom_y", "dom_z"], value_col="charge"):
    """
    Aggregates the values in value_col by the unique combinations in key_cols,
    using NumPy's vectorized operations (avoiding pandas groupby).

    Returns a DataFrame with one row per unique key and a summed value.
    Also merges the first occurrence of other columns in df.
    """
    # Get keys as a 2D numpy array.
    keys = df[key_cols].values
    # Find unique keys and an array of inverse indices.
    unique_keys, index, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    # Use np.bincount to sum the values for each unique key.
    sums = np.bincount(inverse, weights=df[value_col].values)
    # Build the result DataFrame.
    result = pd.DataFrame(unique_keys, columns=key_cols)
    result[value_col] = sums
    # Merge other columns from the first occurrence (if any)
    other_cols = list(set(df.columns) - set(key_cols) - {value_col})
    if other_cols:
        first_occurrence = df.drop_duplicates(subset=key_cols)[key_cols + other_cols]
        result = result.merge(first_occurrence, on=key_cols, how="left")
    return result


def plot_double_charge_ratio(file_path_MC, file_path_RD, output_dir, string_a, string_b):
    """
    Computes and plots the double ratio: (MC_string_a / MC_string_b) / (RD_string_a / RD_string_b),
    using different marker shapes based on RDE classification.
    
    Args:
        file_path_MC (str): Path to the MC pulses database.
        file_path_RD (str): Path to the RD pulses database.
        output_dir (str): Directory to save the generated plot.
        string_a (int or str): First string number to compare.
        string_b (int or str): Second string number to compare.
    """
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 🔹 **Step 1: Load MC & RD Data**
    con_MC = sqlite3.connect(file_path_MC)
    query_MC = f"SELECT dom_x, dom_y, dom_z, charge, event_no, rde, string, dom_number FROM SplitInIcePulses WHERE string IN ({string_a}, {string_b})"
    pulses_df_MC = pd.read_sql_query(query_MC, con_MC)
    con_MC.close()

    con_RD = sqlite3.connect(file_path_RD)
    pulses_df_RD = pd.read_sql_query("SELECT dom_x, dom_y, dom_z, rde, charge, event_no FROM SplitInIcePulses", con_RD)
    con_RD.close()

    # 🔹 **Step 2: Assign `string` to RD Using (x, y)**
    pulses_df_RD["string"] = pulses_df_RD.apply(lambda row: classify_dom_strings(row["dom_x"], row["dom_y"]), axis=1)

    # 🔹 **Step 3: Assign `dom_number` to RD from MC Based on dom_z within ±2 m**
    pulses_df_MC_sorted = pulses_df_MC.sort_values('dom_z').reset_index(drop=True)
    pulses_df_RD_sorted = pulses_df_RD.sort_values('dom_z').reset_index(drop=True)

    # Perform an asof merge on 'dom_z' using a tolerance of 2 meters and the nearest match
    merged_asof = pd.merge_asof(
        pulses_df_RD_sorted, 
        pulses_df_MC_sorted[['dom_z', 'dom_number']], 
        on='dom_z', 
        direction='nearest', 
        tolerance=2
    )

    # For RD rows without a match, fill with -1
    pulses_df_RD_sorted['dom_number'] = merged_asof['dom_number'].fillna(-1).astype(int)
    pulses_df_RD = pulses_df_RD_sorted.sort_index()

    # 🔹 **Step 4: Remove Unclassified RD DOMs**
    pulses_df_RD = pulses_df_RD[pulses_df_RD["string"].notna()]
    pulses_df_RD["dom_number"] = pulses_df_RD["dom_number"].fillna(-1).astype(int)


    def aggregate_charge_without_groupby(df, key_cols, value_col):
        # Placeholder for your aggregation function if needed
        return df.groupby(key_cols, as_index=False)[value_col].sum()

    # Process MC in parallel
    mc_chunks = np.array_split(pulses_df_MC, 4)
    try:
        with ProcessPoolExecutor(max_workers=4) as executor:
            mc_results = list(executor.map(process_chunk, mc_chunks))
    except Exception as e:
        print(f"ERROR processing MC in parallel: {e}")
        return
    total_charge_MC_df = pd.concat(mc_results, ignore_index=True)
    # Re-aggregate in case the same DOM appears in different chunks.
    agg_MC = aggregate_charge_without_groupby(total_charge_MC_df, key_cols=["dom_x", "dom_y", "dom_z"], value_col="charge")

    # Merge with unique MC DOM info (to get rde, string, dom_number)
    agg_MC = agg_MC.merge(
        pulses_df_MC[['dom_x', 'dom_y', 'dom_z', 'rde', 'string', 'dom_number']].drop_duplicates(),
        on=["dom_x", "dom_y", "dom_z"],
        how="left"
    )

    # Process RD in parallel
    rd_chunks = np.array_split(pulses_df_RD, 4)
    try:
        with ProcessPoolExecutor(max_workers=4) as executor:
            rd_results = list(executor.map(process_chunk, rd_chunks))
    except Exception as e:
        print(f"ERROR processing RD in parallel: {e}")
        return
    total_charge_RD_df = pd.concat(rd_results, ignore_index=True)
    agg_RD = aggregate_charge_without_groupby(total_charge_RD_df, key_cols=["dom_x", "dom_y", "dom_z"], value_col="charge")
    agg_RD = agg_RD.merge(
        pulses_df_RD[['dom_x', 'dom_y', 'dom_z', 'rde', 'string', 'dom_number']].drop_duplicates(),
        on=["dom_x", "dom_y", "dom_z"],
        how="left"
    )

    # --- Step 6: Round Coordinates to Avoid Floating-Point Mismatches ---
    for col in ["dom_x", "dom_y", "dom_z"]:
        agg_MC[col] = agg_MC[col].round(2)
        agg_RD[col] = agg_RD[col].round(2)

    # 🔹 **Step 7: Pivot Data for Separate Charge Columns**
    agg_MC_pivot = agg_MC.pivot_table(index="dom_number", columns="string", values="charge")
    agg_MC_pivot = agg_MC_pivot.rename(columns={int(string_a): f"MC_{string_a}", int(string_b): f"MC_{string_b}"})
    
    agg_RD_pivot = agg_RD.pivot_table(index="dom_number", columns="string", values="charge")
    agg_RD_pivot = agg_RD_pivot.rename(columns={int(string_a): f"RD_{string_a}", int(string_b): f"RD_{string_b}"})

    # 🔹 **Step 8: Merge MC and RD on `dom_number`**
    merged = pd.merge(agg_MC_pivot, agg_RD_pivot, on="dom_number", how="inner")

    # 🔹 **Step 9: Merge RDE Information**
    rde_MC = pulses_df_MC.pivot_table(index="dom_number", columns="string", values="rde", aggfunc="first")
    rde_MC = rde_MC.rename(columns={int(string_a): f"RDE_{string_a}", int(string_b): f"RDE_{string_b}"})
    merged = merged.merge(rde_MC, on="dom_number", how="left")

    # Ensure necessary columns exist to avoid division by zero
    if f"RD_{string_a}" not in merged.columns:
        merged[f"RD_{string_a}"] = 1
    if f"RD_{string_b}" not in merged.columns:
        merged[f"RD_{string_b}"] = 1
    if f"MC_{string_a}" not in merged.columns:
        merged[f"MC_{string_a}"] = 1
    if f"MC_{string_b}" not in merged.columns:
        merged[f"MC_{string_b}"] = 1

    # 🔹 **Step 10: Compute the Double Ratio**
    epsilon = 1e-8  # To prevent zero-division
    merged["double_ratio"] = (merged[f"MC_{string_a}"] / (merged[f"MC_{string_b}"] + epsilon)) / (merged[f"RD_{string_a}"] / (merged[f"RD_{string_b}"] + epsilon))

    # 🔹 **Step 11: Assign Marker Shapes Based on RDE Classification**
    def classify_marker(row):
        if row[f"RDE_{string_a}"] == 1.0 and row[f"RDE_{string_b}"] == 1.0:
            return "o"  # Both NQE (Circle)
        elif row[f"RDE_{string_a}"] == 1.0 and row[f"RDE_{string_b}"] == 1.35:
            return "s"  # NQE (string_a) & HQE (string_b) (Square)
        elif row[f"RDE_{string_a}"] == 1.35 and row[f"RDE_{string_b}"] == 1.0:
            return "^"  # HQE (string_a) & NQE (string_b) (Triangle)
        elif row[f"RDE_{string_a}"] == 1.35 and row[f"RDE_{string_b}"] == 1.35:
            return "D"  # Both HQE (Diamond)
        else:
            return "x"  # Unknown case

    merged["marker"] = merged.apply(classify_marker, axis=1)
    marker_desc = {
        "o": f"Both NQE ({string_a} & {string_b}) (Circle)",
        "s": f"NQE ({string_a}) & HQE ({string_b}) (Square)",
        "^": f"HQE ({string_a}) & NQE ({string_b}) (Triangle)",
        "D": f"Both HQE ({string_a} & {string_b}) (Diamond)",
        "x": "Unknown"
    }

    # 🔹 **Step 12: Plot**
    plt.figure(figsize=(10, 6))
    for marker in ["o", "s", "^", "D"]:
        subset = merged[merged["marker"] == marker]
        plt.scatter(subset.index, subset["double_ratio"], alpha=0.7, marker=marker, label=marker_desc.get(marker, marker))

    plt.axhline(1, linestyle="--", color="black", label="Ratio = 1")
    plt.xlabel("DOM Number")
    plt.ylabel("Double Charge Ratio")
    plt.title(f"Double Charge Ratio: (MC_{string_a} / MC_{string_b}) / (RD_{string_a} / RD_{string_b})")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"double_charge_ratio_ MC_{string_a}_MC_{string_b}_RD_{string_a}_RD_{string_b}_NO_SRT.png"))
    plt.close()

    print(f"Saved double charge ratio plot to: {output_dir}/double_charge_ratio.png")

    group_stats = merged.groupby("marker")["double_ratio"].agg(["mean","median","std"]).reset_index()
    group_stats["combination"] = f"{string_a}-{string_b}"
    print(f"Stats for combination {string_a}-{string_b}:")
    print(group_stats)
    return {"combination": f"{string_a}-{string_b}", "marker_stats": group_stats.to_dict(orient="records")}



    

def main():
    file_path_Raw = "/groups/icecube/ptzatzag/work/workspace/Merged_pulses/Unmergedpulsemap_RD_.db"
    #file_path = "/groups/icecube/ptzatzag/work/workspace/Merged_pulses/StoppedMuons/DB/60k_MergedPulsemap_RD_Final.db"
    #file_path_SRT = "/groups/icecube/simon/GNN/workspace/Scripts/filtered_all_big_data.db"
    file_path = '/groups/icecube/simon/GNN/workspace/data/Converted_I3_file/filtered_all_non_clean.db'
    #file_path = "/groups/icecube/petersen/GraphNetDatabaseRepository/dev_lvl3_genie_burnsample/dev_lvl3_genie_burnsample_v5.db"
    con = sqlite3.connect(file_path)
    con_Raw = sqlite3.connect(file_path_Raw)
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = con.execute(query).fetchall()
    print(f"Tables in the database: {tables}")
    query_Raw = "SELECT name FROM sqlite_master WHERE type='table';"
    tables_Raw = con_Raw.execute(query_Raw).fetchall()
    print(f"Tables in the database: {tables_Raw}")
    #Print the coulmns in the retro table
    #retro_query = "PRAGMA table_info(retro);"
    #retro_df = pd.read_sql_query(retro_query, con)
    #print(retro_df)
    #truth_query = "PRAGMA table_info(Truth);"
    #truth_df = pd.read_sql_query(truth_query, con)
    
    
    
    # Print first 5 rows of the Truth table
    #truth_query = "SELECT * FROM Truth LIMIT 5;"
    #truth_df = pd.read_sql_query(truth_query, con)
    #print(truth_df)
    print_memory_usage("Before loading pulses data")
    # Print the tables in the database
    cursor = con.cursor()
    # Step 1: Load all pulses from the database
    # From the SRTInIcePulses table, print the available columns
    pulse_query_raw = """
    SELECT dom_x, dom_y, dom_z, charge, dom_time, event_no, rde
    FROM SplitInIcePulses 
    """
    # For SRTInIcePulses 304650 unique events 
    # For SplitInIcePulses 140229 unique events    
    # pulse_query_raw = """
    # SELECT dom_x, dom_y, dom_z, charge, dom_time, event_no, rde
    # FROM SplitInIcePulses
    # """
    
    x = 1000000
    pulse_query = f"""
        SELECT dom_x, dom_y, dom_z, charge, dom_time, event_no, rde
        FROM SplitInIcePulses
        WHERE event_no IN (
            SELECT DISTINCT event_no
            FROM SplitInIcePulses
            ORDER BY event_no
            LIMIT {x}
        )
        """
    #pulses_df_chunks = pd.read_sql_query(pulse_query, con, chunksize=5000)
    #pulses_df = pd.concat(pulses_df_chunks, ignore_index=True)
    pulses_df = pd.read_sql_query(pulse_query, con)
    unique_events = pulses_df["event_no"].nunique()
    pulses_df_sorted = pulses_df.sort_values(by=["event_no"])
    
    pulses_df_raw = pd.read_sql_query(pulse_query_raw, con_Raw)
    unique_events_raw = pulses_df_raw["event_no"].nunique()
    pulses_df_raw_sorted = pulses_df_raw.sort_values(by=["event_no"])
    print(f"Loaded {len(pulses_df)} pulses from {unique_events} unique events.")
    # predictions_query = """
    # SELECT zenith_pred, energy_pred, azimuth_pred, 
    #        position_x_pred, position_y_pred, position_z_pred, event_no
    # FROM truth
    # """
    predictions_query = """
    SELECT zenith, energy, azimuth, 
           position_x, position_y, position_z, event_no
    FROM truth
    """
    predictions_df = pd.read_sql_query(predictions_query, con)
    
    con.close()
    con_Raw.close()
    print(f"Loaded {len(pulses_df)} pulses from {unique_events} unique events.")
    print_memory_usage("After loading pulses data")

    
    
   # Step 2: Process in parallel
    pulses_chunks = np.array_split(pulses_df, 4)

    try:
        with ProcessPoolExecutor(max_workers=4) as executor:
            combined_results = list(executor.map(process_chunk, pulses_chunks))
    except Exception as e:
        print(f"ERROR in multiprocessing: {e}")
        import traceback
        traceback.print_exc()
        return

    print_memory_usage("After processing events in parallel")

    # Step 3: Aggregate results across chunks
    total_charge_df = pd.concat(combined_results, ignore_index=True)
    grouped = total_charge_df.groupby(["dom_x", "dom_y", "dom_z"])["charge"].sum().reset_index()

    # Step 4: Match RDE values
    grouped = grouped.merge(pulses_df[['dom_x', 'dom_y', 'dom_z', 'rde']].drop_duplicates(), 
                            on=["dom_x", "dom_y", "dom_z"], how="left")

    # Store final results
    aggregated_metrics = {
        "dom_x": grouped["dom_x"].tolist(),
        "dom_y": grouped["dom_y"].tolist(),
        "dom_z": grouped["dom_z"].tolist(),
        "total_charge": grouped["charge"].tolist(),
        "rde": grouped["rde"].tolist()
    }

    print_memory_usage("After aggregating metrics")

    # Step 5: Plot results
    output_dir = "/groups/icecube/simon/GNN/workspace/Plots"
    # Define the string combinations to analyze
    combinations = [(79, 80), (81, 86), (79, 85), (80, 83), (83, 85), (82, 86), (84, 85)]
    
    # Collect marker-level stats for each combination
    all_marker_stats = []
    for s_a, s_b in combinations:
        stats = plot_double_charge_ratio(file_path, file_path_Raw, output_dir, s_a, s_b)
        # stats is a dict with "combination" and "marker_stats" keys
        for marker_stat in stats["marker_stats"]:
            marker_stat["combination"] = stats["combination"]
            all_marker_stats.append(marker_stat)
    
    # Create a DataFrame from the collected marker stats
    df_stats = pd.DataFrame(all_marker_stats)
    print("Combined Marker Stats:")
    print(df_stats)
    
    # Create combined plots for mean, median, and std.
    # We'll create one plot per statistic with combination on x-axis and different markers as hues.
    stats_types = ["mean", "median", "std"]
    
    marker_desc_map = {
    "o": "Both NQE",
    "s": "NQE & HQE",
    "^": "HQE & NQE",
    "D": "Both HQE",
    "x": "Unknown"
    }
    
    for stat in stats_types:
        plt.figure(figsize=(12, 6))
        # Pivot data: rows = combination, columns = marker
        pivot_df = df_stats.pivot(index="combination", columns="marker", values=stat)
        pivot_df = pivot_df.rename(columns=marker_desc_map)
        pivot_df.plot(kind="bar")
        plt.ylabel(stat.capitalize())
        plt.title(f"{stat.capitalize()} of Double Charge Ratio per Marker Group")
        plt.xlabel("String Combination")
        plt.grid(True)
        combined_stat_plot_path = os.path.join(output_dir, f"combined_{stat}_per_marker_NO_SRT.png")
        plt.savefig(combined_stat_plot_path)
        plt.close()
        print(f"Saved combined {stat} plot to: {combined_stat_plot_path}")
    
    
    print_memory_usage("After plotting scatter metrics")

if __name__ == "__main__":
    main()