import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import os
import sys
import matplotlib.colors as colors
from concurrent.futures import ProcessPoolExecutor
from line_profiler import LineProfiler
import yappi
def profile_code():
    """Runs the script with profiling enabled."""

    # 🔹 Start Yappi profiling (global)
    yappi.set_clock_type("cpu")
    yappi.start()

    # 🔹 Run main execution
    main()

    # 🔹 Stop profiling
    yappi.stop()

    # 🔹 Save stats for later
    yappi.get_func_stats().save("main_profile.yappi", type="pstat")

    # 🔹 Print general function performance
    print("\n🔹 Overall Function Performance (Yappi):")
    stats = yappi.get_func_stats()
    stats.sort("ttot").print_all()

    # 🔹 Show thread performance in multiprocessing
    print("\n🔹 Worker Thread Performance:")
    yappi.get_thread_stats().print_all()


def classify_dom_strings(x, y):
    # Dummy classifier: return 79 if x < 0, else 80.
    return 79 if x < 0 else 80

def aggregate_charge_dict(df, key_cols, value_col):
    """
    Vectorized aggregation of df[value_col] grouped by unique rows defined by key_cols using NumPy.
    Returns a dictionary mapping tuple(key) -> summed value.
    """
    keys = df[key_cols].to_numpy()
    unique_keys, inv = np.unique(keys, axis=0, return_inverse=True)
    sums = np.bincount(inv, weights=df[value_col].to_numpy())
    return {tuple(unique_keys[i]): sums[i] for i in range(len(unique_keys))}

def preprocess_pulses(pulses_np, pulse_event_col):
    """
    Group pulses (in a numpy array) by event_no.
    Returns a dictionary mapping event_no -> pulses (numpy array rows).
    """
    unique_events, inverse = np.unique(pulses_np[:, pulse_event_col], return_inverse=True)
    sorted_indices = np.argsort(inverse)
    sorted_pulses = pulses_np[sorted_indices]
    sorted_inverse = inverse[sorted_indices]
    event_start = np.searchsorted(sorted_inverse, np.arange(len(unique_events)))
    event_end = np.append(event_start[1:], len(sorted_inverse))
    event_map = {unique_events[i]: sorted_pulses[event_start[i]:event_end[i]] 
                 for i in range(len(unique_events))}
    return event_map

def calculate_minimum_distance_to_track(pos_array, true_x, true_y, true_z,
                                        true_zenith, true_azimuth, track_length):
    """
    Compute the Euclidean distance from each DOM (in pos_array) to the muon track.
    """
    stop_point = np.array([true_x, true_y, true_z])
    direction_vector = np.array([
        np.sin(true_zenith)*np.cos(true_azimuth),
        np.sin(true_zenith)*np.sin(true_azimuth),
        np.cos(true_zenith)
    ])
    dom_vectors = pos_array[:,:3] - stop_point
    projection_lengths = np.dot(dom_vectors, direction_vector)
    clamped_projections = np.clip(projection_lengths, 0, track_length)
    closest_points = stop_point + clamped_projections[:, np.newaxis] * direction_vector
    return np.linalg.norm(pos_array[:, :3]  - closest_points, axis=1)

def compute_dom_ratio_all_ranges(distance_lower, upper_bounds, event_chunks, pulses_df, pos_array):
    """
    For a fixed lower bound (e.g. 10) and a set of upper bounds (e.g. 20,30,...,160),
    process events only once and update, for each upper bound, per dom_number:
      - expected_hits: count of events where the DOM qualifies (its minimum distance is in [lower, upper))
      - total_charge: summed pulse charge (using a preprocessed lookup from pulses_df)
    For each upper bound, compute RIDE = total_charge / expected_hits for each dom_number on string 79 and 80,
    and then compute the ratio:
         ratio = (RIDE on string 79) / (RIDE on string 80)
         (for each dom_number that exists on both strings).
         
    Returns:
      A dictionary mapping each upper bound to a ratio dictionary:
         {upper_bound: {dom_number: ratio, ...}, ...}
    """
    # Preprocess pulses into a dictionary keyed by event_no.
    pulses_np = pulses_df.to_numpy()
    pulse_event_col = pulses_df.columns.get_loc("event_no")
    event_map = preprocess_pulses(pulses_np, pulse_event_col)
    
    # Column indices for pulse DataFrame lookups.
    col_string = pulses_df.columns.get_loc("string")
    col_dom_number = pulses_df.columns.get_loc("dom_number")
    col_charge = pulses_df.columns.get_loc("charge")
    
    # For pos_array, extract the string IDs and dom_numbers.
    pos_strings = pos_array[:, 3].astype(int)  # e.g. 79 or 80
    pos_dom_numbers = pos_array[:, 4].astype(int)
    
    # Initialize accumulators for each upper bound.
    # Each entry will have separate dictionaries for string 79 and 80:
    #   "hits_79", "chg_79", "hits_80", "chg_80"
    accum = {}
    for ub in upper_bounds:
        accum[ub] = {"hits_79": {}, "chg_79": {}, "hits_80": {}, "chg_80": {}}
    
    # Process events from each chunk.
    for chunk in event_chunks:
        for event in chunk.itertuples(index=False):
            event_no = event.event_no
            # Compute min distances for all DOMs for this event.
            min_dists = calculate_minimum_distance_to_track(
                pos_array,
                event.position_x, event.position_y, event.position_z,
                event.zenith, event.azimuth, event.track_length
            )
            # Build a lookup for pulses in this event keyed by (string, dom_number).
            pulse_dict = {}
            if event_no in event_map:
                for row in event_map[event_no]:
                    key = (int(row[col_string]), int(row[col_dom_number]))
                    pulse_dict[key] = pulse_dict.get(key, 0) + row[col_charge]
            
            # For each upper bound, update accumulators.
            for ub in upper_bounds:
                # Compute mask for DOMs with min distance in [distance_lower, ub)
                mask = (min_dists >= distance_lower) & (min_dists < ub)
                for i, valid in enumerate(mask):
                    if valid:
                        dom_num = pos_dom_numbers[i]
                        s_id = pos_strings[i]
                        if s_id == 79:
                            accum[ub]["hits_79"][dom_num] = 1 #accum[ub]["hits_79"].get(dom_num, 0) + 1
                            accum[ub]["chg_79"][dom_num] = accum[ub]["chg_79"].get(dom_num, 0) + pulse_dict.get((79, dom_num), 0)
                        elif s_id == 80:
                            accum[ub]["hits_80"][dom_num] = 1 #accum[ub]["hits_80"].get(dom_num, 0) + 1
                            accum[ub]["chg_80"][dom_num] = accum[ub]["chg_80"].get(dom_num, 0) + pulse_dict.get((80, dom_num), 0)
    
    # For each upper bound, compute the ratio for each dom_number.
    ratio_all = {}
    for ub in upper_bounds:
        ratio_dict = {}
        hits79 = accum[ub]["hits_79"]
        chg79 = accum[ub]["chg_79"]
        hits80 = accum[ub]["hits_80"]
        chg80 = accum[ub]["chg_80"]
        # Compute ratios only for dom_numbers that appear in both strings.
        common_doms = set(hits79.keys()) & set(hits80.keys())
        for dom in common_doms:
            ride79 = chg79.get(dom, 0) / hits79[dom] if hits79[dom] > 0 else 0
            ride80 = chg80.get(dom, 0) / hits80[dom] if hits80[dom] > 0 else 0
            ratio_dict[dom] = ride79 / ride80 if ride80 > 0 else 0
        ratio_all[ub] = ratio_dict
    return ratio_all

def plot_dom_ratio_heatmap(ratio_results, output_dir):
    """
    Plot a heatmap of per-DOM ratios (RIDE string 79 / RIDE string 80) for varying upper bounds.
    X-axis: upper bound values.
    Y-axis: DOM numbers from string 79.
    
    Parameters:
      ratio_results: dict, mapping each upper bound to a dict {dom_number: ratio}
      output_dir: string, directory where the plot will be saved.
    """
    # Sorted list of upper bounds for the x-axis.
    upper_bounds = sorted(ratio_results.keys())
    
    # For the y-axis, get the union of all dom_numbers (assuming these come from string 79).
    all_doms = sorted(set().union(*(ratio_results[ub].keys() for ub in upper_bounds)))
    
    # Build a 2D matrix with rows = DOM numbers and columns = upper bounds.
    data_matrix = np.full((len(all_doms), len(upper_bounds)), np.nan)
    for j, ub in enumerate(upper_bounds):
        for i, dom in enumerate(all_doms):
            if dom in ratio_results[ub]:
                data_matrix[i, j] = ratio_results[ub][dom]
    norm = colors.TwoSlopeNorm(vmin=0.8, vcenter=1, vmax=1.2)
    # Create the heatmap.
    plt.figure(figsize=(14, 10))
    im = plt.imshow(data_matrix, aspect='auto', origin='lower', cmap='bwr', norm=norm)
    plt.colorbar(im, label="Ratio (RIDE79 / RIDE80)")
    plt.xlabel("Upper Bound")
    plt.ylabel("DOM Number (String 79)")
    plt.title("Heatmap of DOM Ratios (RIDE String 79 / RIDE String 80)")
    
    # Set ticks.
    plt.xticks(np.arange(len(upper_bounds)), upper_bounds)
    plt.yticks(np.arange(len(all_doms)), all_doms)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "dom_ratio_heatmap.png"), dpi=300)
    plt.close()
    
def main():
    # Load data from the filtered DB.
    file_path = '/groups/icecube/simon/GNN/workspace/Scripts/filtered_all_big_data.db'
    con = sqlite3.connect(file_path)
    df_truth = pd.read_sql_query("SELECT * FROM truth", con)
    pulses_df = pd.read_sql_query(
        "SELECT dom_x, dom_y, dom_z, charge, event_no, string, rde, dom_number FROM SplitInIcePulsesSRT",
        con
    )
    con.close()
    
    # Restrict pulses to strings 79 and 80.
    pulses_df = pulses_df[pulses_df['string'].isin([79,80])]
    # Use all unique DOMs (so monitor from string 80 is available).
    pos_array = pulses_df[['dom_x','dom_y','dom_z','string','dom_number']].drop_duplicates().to_numpy()
    
    # Split truth events into chunks.
    event_chunks = np.array_split(df_truth, 10)
    
    distance_lower = 10
    upper_bounds = np.arange(15, 161, 5)  # 20, 30, 40, ... 160
    output_dir = '/groups/icecube/simon/GNN/workspace/Plots/'
    # Compute the per-DOM ratios for all distance ranges.
    ratio_results = compute_dom_ratio_all_ranges(distance_lower, upper_bounds, event_chunks, pulses_df, pos_array)
    plot_dom_ratio_heatmap(ratio_results, output_dir)
    
    # Print results for each upper bound.
    for ub in sorted(ratio_results.keys()):
        print(f"Distance range: {distance_lower} to {ub}")
        ratio_dict = ratio_results[ub]
        for dom in sorted(ratio_dict.keys()):
            print(f"  DOM {dom}: Ratio = {ratio_dict[dom]}")
        print("-" * 40)


if __name__ == "__main__":
    profile_code()