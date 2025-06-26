import numpy as np
import pandas as pd
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor
import psutil
from collections import defaultdict
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
    known_strings = {
        79: (31.25, -72.93),
        80: (72.37, -66.6),
        81: (41.6, 35.49),
        82: (106.94, 27.09),
        83: (113.19, -60.47),
        84: (57.2, -105.52),
        85: (-9.68, -79.5),
        86: (-10.97, 6.72),
    }
    tolerance = 3.0  # meters

    for string_num, (x_ref, y_ref) in known_strings.items():
        if np.isclose(dom_x, x_ref, atol=tolerance) and np.isclose(dom_y, y_ref, atol=tolerance):
            return string_num
    return None  # If no match

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

    plt.xlabel("Depth (m)", fontsize=20)
    plt.ylabel("Total Charge", fontsize=20)
    plt.title("Total Charge vs Depth (Reconstructed Strings, Marked by RDE)")
    plt.grid(axis="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # Legend explanation for multiple strings and RDE markers
    legend_elements = []
    for string, color in STRING_COLORS.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f"{string.replace('_', ' ').title()}, RDE=1.0", markerfacecolor=color, markersize=10))
        legend_elements.append(Line2D([0], [0], marker='^', color='w', label=f"{string.replace('_', ' ').title()}, RDE > 1.0", markerfacecolor=color, markersize=10))

    plt.legend(handles=legend_elements, loc="upper right")

    # Save the plot
    plt.savefig(os.path.join(output_dir, "total_charge_vs_depth_reconstructed_rde_fixed_MC_SRT.png"))
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
    plt.savefig(f"{output_dir}/position_z_distribution_MC_SRT_stopped.png")
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
    FROM SplitInIcePulsesSRT
    WHERE event_no IN (
        SELECT event_no FROM truth WHERE stopped_muon == 1
    )
    """

    pulse_query_RD = """
    SELECT dom_x, dom_y, dom_z, charge, dom_time, event_no, rde
    FROM SRTInIcePulses
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
            mc_results = list(executor.map(process_chunk_1, mc_chunks))
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
    print("Check RD1 DOM count:", agg_rd1.groupby(["string", "dom_number"]).size().value_counts())

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

def aggregate_charge(df):
    return df.groupby(["dom_x", "dom_y", "dom_z"], as_index=False)["charge"].sum()

# Parallel processing
def process_chunk(chunk):
    return chunk.groupby(["dom_x", "dom_y", "dom_z"], as_index=False)["charge"].sum()

def plot_raw_data_ratio(file_path_MC, file_path_RD_1, file_path_RD_2, output_dir, string_a, string_b):
    os.makedirs(output_dir, exist_ok=True)

    # Load MC for DOM classification
    con_mc = sqlite3.connect(file_path_MC)
    query_MC = f"""
        SELECT dom_x, dom_y, dom_z, rde, string, dom_number 
        FROM SplitInIcePulsesSRT 
        WHERE string IN ({string_a}, {string_b})
    """
    pulses_mc = pd.read_sql_query(query_MC, con_mc)
    con_mc.close()

    # Load RD dataset 1
    con_rd1 = sqlite3.connect(file_path_RD_1)
    pulses_rd1 = pd.read_sql_query(
        "SELECT dom_x, dom_y, dom_z, charge, event_no, rde FROM SRTInIcePulses", con_rd1
    )
    con_rd1.close()

    # Load RD dataset 2
    con_rd2 = sqlite3.connect(file_path_RD_2)
    pulses_rd2 = pd.read_sql_query(
        "SELECT dom_x, dom_y, dom_z, charge, event_no, rde FROM SplitInIcePulses", con_rd2
    )
    con_rd2.close()

    # Assign strings
    pulses_rd1["string"] = pulses_rd1.apply(lambda row: classify_dom_strings(row["dom_x"], row["dom_y"]), axis=1)
    pulses_rd2["string"] = pulses_rd2.apply(lambda row: classify_dom_strings(row["dom_x"], row["dom_y"]), axis=1)

    # Sort & clean
    for df in [pulses_mc, pulses_rd1, pulses_rd2]:
        df.sort_values("dom_z", inplace=True)
        df.dropna(subset=["dom_z"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Attach `dom_number` to RD1 and RD2 using MC DOM positions
    for pulses_rd in [pulses_rd1, pulses_rd2]:
        merged_asof = pd.merge_asof(
            pulses_rd,
            pulses_mc[["dom_z", "dom_number"]],
            on="dom_z",
            direction="nearest",
            tolerance=2
        )
        pulses_rd["dom_number"] = merged_asof["dom_number"].fillna(-1).astype(int)

    # Only keep classified DOMs
    pulses_rd1 = pulses_rd1[pulses_rd1["string"].notna()]
    pulses_rd2 = pulses_rd2[pulses_rd2["string"].notna()]

    try:
        with ProcessPoolExecutor(max_workers=12) as executor:
            chunks1 = np.array_split(pulses_rd1, 12)
            chunks2 = np.array_split(pulses_rd2, 12)
            result1 = list(executor.map(process_chunk, chunks1))
            result2 = list(executor.map(process_chunk, chunks2))
    except Exception as e:
        print(f"Parallel processing error: {e}")
        return

    agg_rd1 = pd.concat(result1, ignore_index=True)
    agg_rd2 = pd.concat(result2, ignore_index=True)

    # Round to prevent merge issues
    for col in ["dom_x", "dom_y", "dom_z"]:
        agg_rd1[col] = agg_rd1[col].round(2)
        agg_rd2[col] = agg_rd2[col].round(2)

    # Merge to get DOM info
    dom_info = pulses_mc[["dom_x", "dom_y", "dom_z", "string", "dom_number", "rde"]].drop_duplicates()
    agg_rd1 = agg_rd1.merge(dom_info, on=["dom_x", "dom_y", "dom_z"], how="left")
    agg_rd2 = agg_rd2.merge(dom_info, on=["dom_x", "dom_y", "dom_z"], how="left")

    # Pivot by string
    pivot_rd1 = agg_rd1.pivot_table(index="dom_number", columns="string", values="charge")
    pivot_rd1 = pivot_rd1.rename(columns={int(string_a): f"RD1_{string_a}", int(string_b): f"RD1_{string_b}"})

    pivot_rd2 = agg_rd2.pivot_table(index="dom_number", columns="string", values="charge")
    pivot_rd2 = pivot_rd2.rename(columns={int(string_a): f"RD2_{string_a}", int(string_b): f"RD2_{string_b}"})

    merged = pd.merge(pivot_rd1, pivot_rd2, on="dom_number", how="inner")

    # Add RDE info
    rde_info = dom_info.pivot_table(index="dom_number", columns="string", values="rde", aggfunc="first")
    rde_info = rde_info.rename(columns={int(string_a): f"RDE_{string_a}", int(string_b): f"RDE_{string_b}"})
    merged = merged.merge(rde_info, on="dom_number", how="left")

    # Compute double ratio: RD1 / RD2
    epsilon = 1e-8
    merged["double_ratio"] = (
        (merged[f"RD1_{string_a}"] / (merged[f"RD1_{string_b}"] + epsilon)) /
        (merged[f"RD2_{string_a}"] / (merged[f"RD2_{string_b}"] + epsilon))
    )

    def classify_marker(row):
        if row[f"RDE_{string_a}"] == 1.0 and row[f"RDE_{string_b}"] == 1.0:
            return "o"
        elif row[f"RDE_{string_a}"] == 1.0 and row[f"RDE_{string_b}"] == 1.35:
            return "s"
        elif row[f"RDE_{string_a}"] == 1.35 and row[f"RDE_{string_b}"] == 1.0:
            return "^"
        elif row[f"RDE_{string_a}"] == 1.35 and row[f"RDE_{string_b}"] == 1.35:
            return "D"
        else:
            return "x"

    merged["marker"] = merged.apply(classify_marker, axis=1)

    marker_desc = {
        "o": f"Both NQE ({string_a} & {string_b})",
        "s": f"NQE ({string_a}) & HQE ({string_b})",
        "^": f"HQE ({string_a}) & NQE ({string_b})",
        "D": f"Both HQE ({string_a} & {string_b})",
        "x": "Unknown"
    }

    # Plot
    plt.figure(figsize=(10, 6))
    for marker in ["o", "s", "^", "D"]:
        subset = merged[merged["marker"] == marker]
        plt.scatter(subset.index, subset["double_ratio"], alpha=0.7, marker=marker, label=marker_desc.get(marker))

    plt.axhline(1, linestyle="--", color="black", label="Ratio = 1")
    plt.xlabel("DOM Number")
    plt.ylabel("Charge Ratio (RD1 / RD2)")
    plt.title(f"Charge Ratio: (RD1_{string_a} / RD1_{string_b}) / (RD2_{string_a} / RD2_{string_b})")
    plt.legend()
    plt.grid()
    plot_path = os.path.join(output_dir, f"charge_ratio_RD1_RD2_{string_a}_{string_b}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved charge ratio plot to: {plot_path}")

    group_stats = merged.groupby("marker")["double_ratio"].agg(["mean", "median", "std"]).reset_index()
    group_stats["combination"] = f"{string_a}-{string_b}"
    print(group_stats)

    return {"combination": f"{string_a}-{string_b}", "marker_stats": group_stats.to_dict(orient="records")}

def process_chunk_1(df):
    return df.groupby(["string", "dom_number","rde"], as_index=False)["charge"].sum()


def classify_marker(row, string_a, string_b):
    rde_a = row.get(f"RDE_{string_a}", np.nan)
    rde_b = row.get(f"RDE_{string_b}", np.nan)
    if rde_a == 1.0 and rde_b == 1.0:
        return "o"
    elif rde_a == 1.0 and rde_b == 1.35:
        return "s"
    elif rde_a == 1.35 and rde_b == 1.0:
        return "^"
    elif rde_a == 1.35 and rde_b == 1.35:
        return "D"
    else:
        return "x"



def plot_internal_split_charge_ratio(file_path_MC, file_path_RD, output_dir, string_a, string_b, split_fraction=0.5):
    os.makedirs(output_dir, exist_ok=True)

    con_mc = sqlite3.connect(file_path_MC)
    pulses_mc = pd.read_sql_query(
        f"SELECT dom_x, dom_y, dom_z, rde, string, charge, dom_number FROM SplitInIcePulsesSRT WHERE string IN ({string_a}, {string_b})",
        con_mc
    )
    con_mc.close()
    #SRTInIcePulses, SplitInIcePulses
    con_rd = sqlite3.connect(file_path_RD)
    pulses_rd = pd.read_sql_query(
        "SELECT dom_x, dom_y, dom_z, charge, event_no FROM SplitInIcePulses",
        con_rd
    )
    con_rd.close()

    pulses_rd["string"] = pulses_rd.apply(lambda row: classify_dom_strings(row["dom_x"], row["dom_y"]), axis=1)
    pulses_rd = pulses_rd[pulses_rd["string"].notna()].copy()
    pulses_rd["string"] = pulses_rd["string"].astype(int)
    pulses_mc["string"] = pulses_mc["string"].astype(int)

    dom_lookup = defaultdict(lambda: (-1, np.nan))
    for _, row in pulses_mc.iterrows():
        key = (row["string"], round(row["dom_z"], 2))
        dom_lookup[key] = (int(row["dom_number"]), row["rde"])

    pulses_rd[["dom_number", "rde"]] = pulses_rd.apply(
        lambda row: pd.Series(dom_lookup[(row["string"], round(row["dom_z"], 2))]), axis=1
    )
    pulses_rd = pulses_rd[pulses_rd["dom_number"] >= 0].copy()

    unique_events = pulses_rd["event_no"].unique()
    np.random.shuffle(unique_events)
    split_idx = int(len(unique_events) * split_fraction)
    events_rd1 = unique_events[:split_idx]
    events_rd2 = unique_events[split_idx:]

    pulses_rd1 = pulses_rd[pulses_rd["event_no"].isin(events_rd1)].copy()
    pulses_rd2 = pulses_rd[pulses_rd["event_no"].isin(events_rd2)].copy()

    n_workers = min(40, os.cpu_count())
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        chunks1 = np.array_split(pulses_rd1, n_workers)
        chunks2 = np.array_split(pulses_rd2, n_workers)
        result1 = list(executor.map(process_chunk_1, chunks1))
        result2 = list(executor.map(process_chunk_1, chunks2))

    # Correctly aggregate charge per (string, dom_number)
    agg_rd1 = pd.concat(result1, ignore_index=True).groupby(["string", "dom_number"], as_index=False)["charge"].sum()
    agg_rd2 = pd.concat(result2, ignore_index=True).groupby(["string", "dom_number"], as_index=False)["charge"].sum()

    rde_map = pulses_rd.drop_duplicates(["string", "dom_number"])[["string", "dom_number", "rde"]]
    agg_rd1 = agg_rd1.merge(rde_map, on=["string", "dom_number"], how="left")
    agg_rd2 = agg_rd2.merge(rde_map, on=["string", "dom_number"], how="left")

    pivot_rd1 = agg_rd1.pivot(index="dom_number", columns="string", values="charge")
    pivot_rd2 = agg_rd2.pivot(index="dom_number", columns="string", values="charge")

    pivot_rd1 = pivot_rd1.rename(columns={string_a: f"{string_a}_RD1", string_b: f"{string_b}_RD1"})
    pivot_rd2 = pivot_rd2.rename(columns={string_a: f"{string_a}_RD2", string_b: f"{string_b}_RD2"})


    merged = pivot_rd1.join(pivot_rd2, lsuffix="_RD1", rsuffix="_RD2").reset_index()


    rde_map_a = pulses_mc[pulses_mc["string"] == string_a].drop_duplicates("dom_number").set_index("dom_number")["rde"]
    rde_map_b = pulses_mc[pulses_mc["string"] == string_b].drop_duplicates("dom_number").set_index("dom_number")["rde"]

    merged[f"RDE_{string_a}"] = merged.index.map(rde_map_a)
    merged[f"RDE_{string_b}"] = merged.index.map(rde_map_b)

    charge_threshold = 0.1
    
    required_cols = [f"{string_a}_RD1", f"{string_b}_RD1", f"{string_a}_RD2", f"{string_b}_RD2"]
    missing = [col for col in required_cols if col not in merged.columns]
    if missing:
        raise KeyError(f"Missing expected columns in merged DataFrame: {missing}")

    mask = (
        (merged[f"{string_a}_RD1"] > charge_threshold) &
        (merged[f"{string_b}_RD1"] > charge_threshold) &
        (merged[f"{string_a}_RD2"] > charge_threshold) &
        (merged[f"{string_b}_RD2"] > charge_threshold)
    )
    merged = merged[mask].copy()
    print("Unique event_no in RD:", len(pulses_rd["event_no"].unique()))
    epsilon = 1e-8
    merged["double_ratio"] = (
        (merged[f"{string_a}_RD1"] / (merged[f"{string_b}_RD1"] + epsilon)) /
        (merged[f"{string_a}_RD2"] / (merged[f"{string_b}_RD2"] + epsilon))
    )
    merged["marker"] = merged.apply(lambda row: classify_marker(row, string_a, string_b), axis=1)


    plt.figure(figsize=(10, 6))
    marker_desc = {
        "o": f"Both NQE ({string_a} & {string_b})",
        "s": f"NQE ({string_a}) & HQE ({string_b})",
        "^": f"HQE ({string_a}) & NQE ({string_b})",
        "D": f"Both HQE ({string_a} & {string_b})",
        "x": "Unknown"
    }
    markers = {"o": "o", "s": "s", "^": "^", "D": "D", "x": "x"}
    
    for marker, mstyle in markers.items():
        sub = merged[merged["marker"] == marker]
        plt.scatter(sub["dom_number"], sub["double_ratio"], alpha=0.7, marker=mstyle, label=marker_desc.get(marker, marker))


    plt.axhline(1, linestyle="--", color="black", label="Ratio = 1")
    plt.xlabel("DOM Number")
    plt.ylabel("Charge Double Ratio (RD1 / RD2)")
    plt.title(f"Internal Split Charge Ratio: (RD1_{string_a} / RD1_{string_b}) / (RD2_{string_a} / RD2_{string_b})")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"internal_split_charge_ratio_big_{string_a}_{string_b}.png"))
    plt.close()

    # --- Total Charge Plot ---
    plt.figure(figsize=(12, 6))
    for df, label in zip([agg_rd1, agg_rd2], ["RD1", "RD2"]):
        df = df.rename(columns={"charge": f"{label}_charge"})
        for (s, rde_val), sub in df.groupby(["string", "rde"]):
            marker = "o" if label == "RD1" else "s"
            plt.scatter(sub["dom_number"], sub[f"{label}_charge"], label=f"{label} - String {s} - RDE {rde_val}", marker=marker, alpha=0.6)

    plt.xlabel("DOM Number")
    plt.ylabel("Total Charge")
    plt.title("Total Charge per DOM")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "total_charge_per_dom_big.png"))
    plt.close()

    group_stats = merged.groupby("marker")["double_ratio"].agg(["mean", "median", "std"]).reset_index()
    group_stats["combination"] = f"{string_a}-{string_b}"
    print(group_stats)

    return {"combination": f"{string_a}-{string_b}", "marker_stats": group_stats.to_dict(orient="records")}


def plot_double_charge_ratio(file_path_MC, file_path_RD, output_dir, string_a, string_b):
    os.makedirs(output_dir, exist_ok=True)

    con_mc = sqlite3.connect(file_path_MC)
    pulses_mc = pd.read_sql_query(
        f"SELECT dom_x, dom_y, dom_z, rde, string, charge, dom_number FROM SplitInIcePulsesSRT WHERE string IN ({string_a}, {string_b})",
        con_mc
    )
    con_mc.close()

    con_rd = sqlite3.connect(file_path_RD)
    pulses_rd = pd.read_sql_query(
        "SELECT dom_x, dom_y, dom_z, charge, event_no FROM SplitInIcePulses",
        con_rd
    )
    con_rd.close()

    pulses_rd["string"] = pulses_rd.apply(lambda row: classify_dom_strings(row["dom_x"], row["dom_y"]), axis=1)
    pulses_rd = pulses_rd[pulses_rd["string"].notna()].copy()
    pulses_rd["string"] = pulses_rd["string"].astype(int)
    pulses_mc["string"] = pulses_mc["string"].astype(int)

    dom_lookup = defaultdict(lambda: (-1, np.nan))
    for _, row in pulses_mc.iterrows():
        key = (row["string"], round(row["dom_z"], 2))
        dom_lookup[key] = (int(row["dom_number"]), row["rde"])

    pulses_rd[["dom_number", "rde"]] = pulses_rd.apply(
        lambda row: pd.Series(dom_lookup[(row["string"], round(row["dom_z"], 2))]), axis=1
    )
    pulses_rd = pulses_rd[pulses_rd["dom_number"] >= 0].copy()
    #print the len of unique event_no
    print("Unique event_no in RD:", len(pulses_rd["event_no"].unique()))
    n_workers = min(40, os.cpu_count())
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        mc_chunks = np.array_split(pulses_mc, n_workers)
        rd_chunks = np.array_split(pulses_rd, n_workers)
        result_mc = list(executor.map(process_chunk_1, mc_chunks))
        result_rd = list(executor.map(process_chunk_1, rd_chunks))


    agg_rd = pd.concat(result_rd, ignore_index=True).groupby(["string", "dom_number"], as_index=False)["charge"].sum()
    agg_mc = pd.concat(result_mc, ignore_index=True).groupby(["string", "dom_number", "rde"], as_index=False)["charge"].sum()

    rde_map = pulses_rd.drop_duplicates(["string", "dom_number"])[["string", "dom_number", "rde"]]
    agg_rd = agg_rd.merge(rde_map, on=["string", "dom_number"], how="left")
    
    # Filter and rename charge for each string
    agg_mc_a = agg_mc[agg_mc["string"] == string_a][["dom_number", "charge"]].rename(columns={"charge": f"{string_a}_MC"})
    agg_mc_b = agg_mc[agg_mc["string"] == string_b][["dom_number", "charge"]].rename(columns={"charge": f"{string_b}_MC"})
    agg_rd_a = agg_rd[agg_rd["string"] == string_a][["dom_number", "charge"]].rename(columns={"charge": f"{string_a}_RD"})
    agg_rd_b = agg_rd[agg_rd["string"] == string_b][["dom_number", "charge"]].rename(columns={"charge": f"{string_b}_RD"})

    # Merge them on dom_number
    merged = agg_mc_a.merge(agg_mc_b, on="dom_number", how="inner") \
                    .merge(agg_rd_a, on="dom_number", how="inner") \
                    .merge(agg_rd_b, on="dom_number", how="inner")

    # Merge in RDE info if needed
    rde_map_a = pulses_mc[pulses_mc["string"] == string_a].drop_duplicates("dom_number").set_index("dom_number")["rde"]
    rde_map_b = pulses_mc[pulses_mc["string"] == string_b].drop_duplicates("dom_number").set_index("dom_number")["rde"]
    merged[f"RDE_{string_a}"] = merged["dom_number"].map(rde_map_a)
    merged[f"RDE_{string_b}"] = merged["dom_number"].map(rde_map_b)

    # Filter: ensure non-zero charges (optional: raise threshold)
    charge_threshold = 0.1
    mask = (
        (merged[f"{string_a}_MC"] > charge_threshold) &
        (merged[f"{string_b}_MC"] > charge_threshold) &
        (merged[f"{string_a}_RD"] > charge_threshold) &
        (merged[f"{string_b}_RD"] > charge_threshold)
    )
    merged = merged[mask].copy()

    # Compute double ratio
    epsilon = 1e-8
    merged["double_ratio"] = (
        (merged[f"{string_a}_MC"] / (merged[f"{string_b}_MC"] + epsilon)) /
        (merged[f"{string_a}_RD"] / (merged[f"{string_b}_RD"] + epsilon))
    )

    # Classify marker if needed
    merged["marker"] = merged.apply(lambda row: classify_marker(row, string_a, string_b), axis=1)


    marker_desc = {
        "o": f"Both NQE ({string_a} & {string_b})",
        "s": f"NQE ({string_a}) & HQE ({string_b})",
        "^": f"HQE ({string_a}) & NQE ({string_b})",
        "D": f"Both HQE ({string_a} & {string_b})",
        "x": "Unknown"
    }
    markers = {"o": "o", "s": "s", "^": "^", "D": "D", "x": "x"}

    plt.figure(figsize=(10, 6), constrained_layout=True)

    for marker, mstyle in markers.items():
        sub = merged[merged["marker"] == marker]
        plt.scatter(sub["dom_number"], sub["double_ratio"], alpha=0.7, marker=mstyle, label=marker_desc.get(marker, marker))

    plt.axhline(1, linestyle="--", color="black", label="Ratio = 1")
    plt.xlabel("DOM Number", fontsize=22)
    plt.ylabel("Double Charge Ratio", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f"Double Charge Ratio: (MC_{string_a} / MC_{string_b}) / (RD_{string_a} / RD_{string_b})", fontsize=24)
    plt.legend(fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"double_charge_panos_ratio_MC_{string_a}_{string_b}_RD_{string_a}_{string_b}.png"))
    plt.close()

    # Total charge plot
    plt.figure(figsize=(12, 6))
    for df, label in zip([agg_mc, agg_rd], ["MC", "RD"]):
        for (s, rde_val), sub in df.groupby(["string", "rde"]):
            marker = "o" if label == "MC" else "s"
            plt.scatter(sub["dom_number"], sub["charge"], label=f"{label} - String {s} - RDE {rde_val}", marker=marker, alpha=0.6)

    plt.xlabel("DOM Number", fontsize=20)
    plt.ylabel("Total Charge", fontsize=20)
    plt.title("Total Charge per DOM", fontsize=22)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"total_charge_per_dom_{string_a}_{string_b}.png"))
    plt.close()

    group_stats = merged.groupby("marker")["double_ratio"].agg(["mean", "median", "std"]).reset_index()
    group_stats["combination"] = f"{string_a}-{string_b}"
    print(group_stats)
    return {"combination": f"{string_a}-{string_b}", "marker_stats": group_stats.to_dict(orient="records")}



def main():
    file_path_Raw = "/groups/icecube/simon/GNN/workspace/storage/Training/stopped_through_classification/train_model_without_configs/muon_stopped_sorted.db"#
    file_path_panos = "/groups/icecube/ptzatzag/work/workspace/Merged_pulses/Unmergedpulsemap_RD_.db"
    #file_path = "/groups/icecube/ptzatzag/work/workspace/Merged_pulses/StoppedMuons/DB/60k_MergedPulsemap_RD_Final.db"
    #file_path = "/groups/icecube/simon/GNN/workspace/Scripts/filtered_all.db"
    file_path = '/groups/icecube/simon/GNN/workspace/Scripts/filtered_all_big_data.db'
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
    # cursor = con.cursor()
    # # Step 1: Load all pulses from the database
    # # From the SRTInIcePulses table, print the available columns
    # pulse_query_raw = """
    # SELECT dom_x, dom_y, dom_z, charge, dom_time, event_no, rde
    # FROM SRTInIcePulses 
    # """
    # # For SRTInIcePulses 304650 unique events 
    # # For SplitInIcePulses 140229 unique events    
    # # pulse_query_raw = """
    # # SELECT dom_x, dom_y, dom_z, charge, dom_time, event_no, rde
    # # FROM SplitInIcePulses
    # # """
    
    # x = 1600000
    # pulse_query = f"""
    # SELECT dom_x, dom_y, dom_z, charge, dom_time, event_no, rde
    # FROM SplitInIcePulsesSRT
    # WHERE event_no IN (
    #     SELECT event_no
    #     FROM (
    #         SELECT DISTINCT event_no
    #         FROM SplitInIcePulsesSRT
    #         WHERE event_no IN (
    #             SELECT event_no FROM truth WHERE stopped_muon == 1
    #         )
    #         ORDER BY event_no
    #         LIMIT {x}
    #     )
    # )
    # """
    # #pulses_df_chunks = pd.read_sql_query(pulse_query, con, chunksize=5000)
    # #pulses_df = pd.concat(pulses_df_chunks, ignore_index=True)
    # pulses_df = pd.read_sql_query(pulse_query, con)
    # unique_events = pulses_df["event_no"].nunique()
    # pulses_df_sorted = pulses_df.sort_values(by=["event_no"])
    
    # pulses_df_raw = pd.read_sql_query(pulse_query_raw, con_Raw)
    # unique_events_raw = pulses_df_raw["event_no"].nunique()
    # pulses_df_raw_sorted = pulses_df_raw.sort_values(by=["event_no"])
    # print(f"Loaded {len(pulses_df)} pulses from {unique_events} unique events.")
    # # predictions_query = """
    # # SELECT zenith_pred, energy_pred, azimuth_pred, 
    # #        position_x_pred, position_y_pred, position_z_pred, event_no
    # # FROM truth
    # # """
    # predictions_query = """
    # SELECT zenith, energy, azimuth, 
    #        position_x, position_y, position_z, event_no
    # FROM truth
    # """
    # predictions_df = pd.read_sql_query(predictions_query, con)
    
    # con.close()
    # con_Raw.close()
    # print(f"Loaded {len(pulses_df)} pulses from {unique_events} unique events.")
    # print_memory_usage("After loading pulses data")

    
    
#    # Step 2: Process in parallel
#     pulses_chunks = np.array_split(pulses_df, 4)

#     try:
#         with ProcessPoolExecutor(max_workers=4) as executor:
#             combined_results = list(executor.map(process_chunk, pulses_chunks))
#     except Exception as e:
#         print(f"ERROR in multiprocessing: {e}")
#         import traceback
#         traceback.print_exc()
#         return

#     print_memory_usage("After processing events in parallel")
#     print(f"📊 Processing {len(pulses_df)} pulses and {len(predictions_df)} predictions.")

#     # Step 3: Aggregate results across chunks
#     total_charge_df = pd.concat(combined_results, ignore_index=True)
#     print(f"Total charge DataFrame shape: {total_charge_df.shape}")
#     grouped = total_charge_df.groupby(["dom_x", "dom_y", "dom_z"])["charge"].sum().reset_index()

#     # Step 4: Match RDE values
#     grouped = grouped.merge(pulses_df[['dom_x', 'dom_y', 'dom_z', 'rde']].drop_duplicates(), 
#                             on=["dom_x", "dom_y", "dom_z"], how="left")
    
#     # Store final results
#     aggregated_metrics = {
#         "dom_x": grouped["dom_x"].tolist(),
#         "dom_y": grouped["dom_y"].tolist(),
#         "dom_z": grouped["dom_z"].tolist(),
#         "total_charge": grouped["charge"].tolist(),
#         "rde": grouped["rde"].tolist()
#     }

#     print_memory_usage("After aggregating metrics")

    # Step 5: Plot results
    output_dir = "/groups/icecube/simon/GNN/workspace/Plots"
    # Define the string combinations to analyze
    combinations = [(79, 80), (81, 86), (79, 85), (80, 83), (83, 85), (82, 86), (84, 85)]
    
    plot_internal_split_charge_ratio(file_path_MC=file_path, file_path_RD=file_path_panos, output_dir=output_dir, string_a=79, string_b=80)
    
    # Collect marker-level stats for each combination
    all_marker_stats = []
    for s_a, s_b in combinations:
        stats = plot_double_charge_ratio(file_path, file_path_panos, output_dir, s_a, s_b)
        # stats is a dict with "combination" and "marker_stats" keys
        for marker_stat in stats["marker_stats"]:
            marker_stat["combination"] = stats["combination"]
            all_marker_stats.append(marker_stat)
    
    # # Create a DataFrame from the collected marker stats
    # df_stats = pd.DataFrame(all_marker_stats)
    # print("Combined Marker Stats:")
    # print(df_stats)
    
    # # Create combined plots for mean, median, and std.
    # # We'll create one plot per statistic with combination on x-axis and different markers as hues.
    # stats_types = ["mean", "median", "std"]
    
    # marker_desc_map = {
    # "o": "Both NQE",
    # "s": "NQE & HQE",
    # "^": "HQE & NQE",
    # "D": "Both HQE",
    # "x": "Unknown"
    # }
    
    # for stat in stats_types:
    #     plt.figure(figsize=(12, 6))
    #     # Pivot data: rows = combination, columns = marker
    #     pivot_df = df_stats.pivot(index="combination", columns="marker", values=stat)
    #     pivot_df = pivot_df.rename(columns=marker_desc_map)
    #     pivot_df.plot(kind="bar")
    #     plt.ylabel(stat.capitalize())
    #     plt.title(f"{stat.capitalize()} of Double Charge Ratio per Marker Group")
    #     plt.xlabel("String Combination")
    #     plt.grid(True)
    #     combined_stat_plot_path = os.path.join(output_dir, f"combined_{stat}_per_marker_srt.png")
    #     plt.savefig(combined_stat_plot_path)
    #     plt.close()
    #     print(f"Saved combined {stat} plot to: {combined_stat_plot_path}")
    
    
    #plot_raw_data_ratio(file_path_MC=file_path, file_path_RD_1=file_path_Raw, file_path_RD_2=file_path_panos, output_dir=output_dir, string_a=79, string_b=80)
    print_memory_usage("After plotting scatter metrics")

if __name__ == "__main__":
    main()