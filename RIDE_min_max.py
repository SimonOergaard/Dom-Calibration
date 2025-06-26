import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import os
import sys
import matplotlib.colors as colors
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.colors as mcolors


def define_non_overlapping_bins(start=0, stop=160, step=10):
    """Creates non-overlapping bin edges."""
    edges = np.arange(start, stop + step, step)  # e.g., [0, 10, 20, ..., 160]
    lowers = edges[:-1]  # [0, 10, ..., 150]
    uppers = edges[1:]   # [10, 20, ..., 160]
    return lowers, uppers


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


def process_event_chunk(chunk, pulses_df_np, pos_array, pulse_event_col, col_string, col_dom_number, col_charge,
                        lower_bound, upper_bound):
    event_map = preprocess_pulses(pulses_df_np, pulse_event_col)
    pos_strings = pos_array[:, 3].astype(int)
    pos_dom_numbers = pos_array[:, 4].astype(int)

    accum = {"chg": {}, "hits": {}}

    for event in chunk.itertuples(index=False):
        event_no = event.event_no
        min_dists = calculate_minimum_distance_to_track(
            pos_array,
            event.position_x, event.position_y, event.position_z,
            event.zenith, event.azimuth, event.track_length
        )

        pulse_dict = {}
        hit_dict = {}
        if event_no in event_map:
            for row in event_map[event_no]:
                key = (int(row[col_string]), int(row[col_dom_number]))
                pulse_dict[key] = pulse_dict.get(key, 0) + row[col_charge]
                hit_dict[key] = hit_dict.get(key, 0) + 1

        mask = (min_dists >= lower_bound) & (min_dists < upper_bound)
        for i, valid in enumerate(mask):
            if valid:
                dom_num = int(pos_dom_numbers[i])
                s_id = int(pos_strings[i])
                dom_key = f"{s_id}-{dom_num}"
                accum["chg"][dom_key] = accum["chg"].get(dom_key, 0) + pulse_dict.get((s_id, dom_num), 0)
                accum["hits"][dom_key] = accum["hits"].get(dom_key, 0) + 1

    return accum



def process_event_chunk_wrapper(args):
    return process_event_chunk(*args)

def compute_dom_charge_non_cumulative(distance_lowers, distance_uppers, event_chunks, pulses_df, pos_array):
    pulses_df_np = pulses_df.to_numpy()
    pulse_event_col = pulses_df.columns.get_loc("event_no")
    col_string = pulses_df.columns.get_loc("string")
    col_dom_number = pulses_df.columns.get_loc("dom_number")
    col_charge = pulses_df.columns.get_loc("charge")

    merged_accum = {f"{lb}-{ub}": {"chg": {}, "hits": {}} for lb, ub in zip(distance_lowers, distance_uppers)}

    args_list = []
    for lb, ub in zip(distance_lowers, distance_uppers):
        for chunk in event_chunks:
            args_list.append((chunk, pulses_df_np, pos_array, pulse_event_col,
                              col_string, col_dom_number, col_charge, lb, ub))

    with ProcessPoolExecutor(max_workers=12) as executor:
        results = list(tqdm(executor.map(process_event_chunk_wrapper, args_list), total=len(args_list)))

    # Combine results from chunks
    idx = 0
    for lb, ub in zip(distance_lowers, distance_uppers):
        bin_key = f"{lb}-{ub}"
        for _ in range(len(event_chunks)):
            result = results[idx]
            idx += 1
            for dom_key, val in result["chg"].items():
                merged_accum[bin_key]["chg"][dom_key] = merged_accum[bin_key]["chg"].get(dom_key, 0) + val
            for dom_key, val in result["hits"].items():
                merged_accum[bin_key]["hits"][dom_key] = merged_accum[bin_key]["hits"].get(dom_key, 0) + val

    return merged_accum



def plot_dom_RIDE_non_cumulative(merged_accum, output_dir, filename="dom_ride_non_cumulative_deep.png"):
    bins = sorted(merged_accum.keys(), key=lambda x: float(x.split('-')[0]))
    
    # Collect all DOM keys
    all_dom_keys = sorted(set().union(*(merged_accum[bin_key]['hits'].keys() for bin_key in bins)))

    sorted_doms = sorted(
        all_dom_keys,
        key=lambda x: (int(x.split('-')[1]), int(x.split('-')[0]))
    )

    data_matrix = np.full((len(sorted_doms), len(bins)), np.nan)

    for j, bin_key in enumerate(bins):
        chg = merged_accum[bin_key]['chg']
        hits = merged_accum[bin_key]['hits']

        for i, dom_key in enumerate(sorted_doms):
            string, dom_number = dom_key.split('-')
            dom_number = int(dom_number)

            test_hits = hits.get(dom_key, 0)
            test_charge = chg.get(dom_key, 0)
            # 🔥 Important: we look up monitor even if not plotting it
            monitor_hits = hits.get(f"80-{dom_number}", 0)
            monitor_charge = chg.get(f"80-{dom_number}", 0)

            if test_hits > 0 and monitor_hits > 0 and monitor_charge > 0:
                ride_test = test_charge / test_hits
                ride_monitor = monitor_charge / monitor_hits
                ride_ratio = ride_test / ride_monitor
                data_matrix[i, j] = ride_ratio

    # Plotting
    plt.figure(figsize=(14, 10))
    norm = colors.TwoSlopeNorm(vmin=0.9, vcenter=1.35, vmax=1.5)  # Center around 1.35
    im = plt.imshow(data_matrix, aspect='auto', origin='lower', cmap='bwr', norm=norm)
    plt.colorbar(im, label="RIDE Ratio (DOM / Monitor DOM)")
    plt.xlabel("Distance Bin")
    plt.ylabel("DOM (string-dom)")
    plt.title("RIDE per DOM (relative to string 80)")

    plt.xticks(np.arange(len(bins)), bins, rotation=45)
    plt.yticks(np.arange(len(sorted_doms)), sorted_doms)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def plot_dom_hit_heatmap_non_cumulative(charge_by_bin, output_dir, filename="dom_hits_non_cumulative_deep.png"):
    bins = sorted(charge_by_bin.keys(), key=lambda x: float(x.split('-')[0]))
    all_dom_keys = sorted(set().union(*(charge_by_bin[bin_key]['hits'].keys() for bin_key in bins)))
    sorted_doms = sorted(
        all_dom_keys,
        key=lambda x: (int(x.split('-')[1]), int(x.split('-')[0]))
    )
    data_matrix = np.full((len(sorted_doms), len(bins)), 0.0)
    for j, bin_key in enumerate(bins):
        for i, dom_key in enumerate(sorted_doms):
            data_matrix[i, j] = charge_by_bin[bin_key]['hits'].get(dom_key, 0)
    # Create a masked array where 0.0 will be masked (treated specially)
    masked_data = np.ma.masked_where(data_matrix == 0, data_matrix)
    # Create a new colormap: white for 0, then viridis
    cmap = plt.cm.viridis
    new_cmap = cmap.copy()
    new_cmap.set_bad(color='white')  # masked values → white
    plt.figure(figsize=(14, 10))
    im = plt.imshow(masked_data, aspect='auto', origin='lower', cmap=new_cmap)
    plt.colorbar(im, label="Total Hits per Bin")
    plt.xticks(np.arange(len(bins)), bins, rotation=45)
    plt.yticks(np.arange(len(sorted_doms)), sorted_doms)
    plt.xlabel("Distance Bin (m)")
    plt.ylabel("DOM (string-dom_number)")
    plt.title("Hits Collected per DOM per Distance Bin (Non-cumulative)")
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
       

def plot_dom_charge_heatmap_non_cumulative(charge_by_bin, output_dir, filename="dom_charge_non_cumulative.png"):
    bins = sorted(charge_by_bin.keys(), key=lambda x: float(x.split('-')[0]))
    all_dom_keys = sorted(set().union(*(charge_by_bin[bin_key]['chg'].keys() for bin_key in bins)))

    sorted_doms = sorted(
        all_dom_keys,
        key=lambda x: (int(x.split('-')[1]), int(x.split('-')[0]))
    )


    data_matrix = np.full((len(sorted_doms), len(bins)), 0.0)

    for j, bin_key in enumerate(bins):
        for i, dom_key in enumerate(sorted_doms):
            data_matrix[i, j] = charge_by_bin[bin_key]['chg'].get(dom_key, 0)

    # Create a masked array where 0.0 will be masked (treated specially)
    masked_data = np.ma.masked_where(data_matrix == 0, data_matrix)

    # Create a new colormap: white for 0, then viridis
    cmap = plt.cm.viridis
    new_cmap = cmap.copy()
    new_cmap.set_bad(color='white')  # masked values → white

    plt.figure(figsize=(16, 12))
    im = plt.imshow(masked_data, aspect='auto', origin='lower', cmap=new_cmap)
    cbar = plt.colorbar(im)
    cbar.set_label("Total Charge Collected", fontsize=20)

    plt.xlabel("Distance Bin (m)", fontsize=20)
    plt.ylabel("DOM (string-dom_number)", fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Charge Collected per DOM per Distance Bin (Non-cumulative)", fontsize=22)
    plt.xticks(np.arange(len(bins)), bins, rotation=45)
    plt.yticks(np.arange(len(sorted_doms)), sorted_doms)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def plot_dom_ride_ratio_heatmap(merged_accum, output_dir, filename="ride_ratio_per_dom_heatmap_deep.png"):

    bins = sorted(merged_accum.keys(), key=lambda x: float(x.split('-')[0]))
    
    # Step 1: collect all DOM keys (full data)
    all_dom_keys = sorted(set().union(*(merged_accum[bin_key]['hits'].keys() for bin_key in bins)))

    # Step 2: for plotting only exclude string 80
    plot_dom_keys = [k for k in all_dom_keys if not k.startswith("80-")]

    sorted_doms = sorted(plot_dom_keys, key=lambda x: (int(x.split('-')[1]), int(x.split('-')[0])))

    data_matrix = np.full((len(sorted_doms), len(bins)), np.nan)

    for j, bin_key in enumerate(bins):
        chg = merged_accum[bin_key]['chg']
        hits = merged_accum[bin_key]['hits']

        for i, dom_key in enumerate(sorted_doms):
            string, dom_number = dom_key.split('-')
            dom_number = int(dom_number)

            test_hits = hits.get(dom_key, 0)
            test_charge = chg.get(dom_key, 0)
            # 🔥 Important: we look up monitor even if not plotting it
            monitor_hits = hits.get(f"80-{dom_number}", 0)
            monitor_charge = chg.get(f"80-{dom_number}", 0)

            if test_hits > 0 and monitor_hits > 0 and monitor_charge > 0:
                ride_test = test_charge / test_hits
                ride_monitor = monitor_charge / monitor_hits
                ride_ratio = ride_test / ride_monitor
                data_matrix[i, j] = ride_ratio

    # Plotting
    plt.figure(figsize=(14, 10))
    norm = colors.TwoSlopeNorm(vmin=0.9, vcenter=1.35, vmax=1.5)  # Center around 1.35
    im = plt.imshow(data_matrix, aspect='auto', origin='lower', cmap='bwr', norm=norm)
    plt.colorbar(im, label="RIDE Ratio (DOM / Monitor DOM)")
    plt.xlabel("Distance Bin")
    plt.ylabel("DOM (string-dom)")
    plt.title("RIDE per DOM (relative to string 80)")

    plt.xticks(np.arange(len(bins)), bins, rotation=45)
    plt.yticks(np.arange(len(sorted_doms)), sorted_doms)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()



  
def main():
    file_path = '/groups/icecube/simon/GNN/workspace/Scripts/filtered_all_big_data.db'
    con = sqlite3.connect(file_path)
    df_truth = pd.read_sql_query("SELECT * FROM truth", con)
    pulses_df = pd.read_sql_query(
        "SELECT dom_x, dom_y, dom_z, charge, event_no, string, dom_time, rde, dom_number FROM SplitInIcePulsesSRT",
        con
    )
    con.close()
    
    # Restrict pulses to strings 79 and 80.
    pulses_df = pulses_df[
    pulses_df['string'].isin([80, 81, 82, 83, 84, 85]) & 
        (pulses_df['dom_number'] > 10)
    ]
    pos_array = pulses_df[['dom_x','dom_y','dom_z','string','dom_number']].drop_duplicates().to_numpy()
    
    # Split truth events into chunks.
    event_chunks = np.array_split(df_truth, 12)
    
    distance_lower, upper_bounds = define_non_overlapping_bins(0, 160, 10)
    output_dir = '/groups/icecube/simon/GNN/workspace/Plots/'
    
    merged_accum = compute_dom_charge_non_cumulative(
        distance_lower, upper_bounds, event_chunks, pulses_df, pos_array
    )
    
    plot_dom_charge_heatmap_non_cumulative(merged_accum, output_dir)
    plot_dom_ride_ratio_heatmap(merged_accum, output_dir)
    plot_dom_hit_heatmap_non_cumulative(merged_accum, output_dir)
    plot_dom_RIDE_non_cumulative(merged_accum, output_dir)
    print("Plot saved to:", os.path.join(output_dir, "dom_charge_non_cumulative.png"))
    print("Processing complete.")
    from collections import defaultdict

    # Accumulate total charge per DOM across all bins
    dom_charge_total = defaultdict(float)

    for bin_data in merged_accum.values():
        for dom_key, charge in bin_data["chg"].items():
            dom_charge_total[dom_key] += charge

    # Print results sorted by DOM number then string
    for dom_key in sorted(dom_charge_total, key=lambda x: (int(x.split("-")[1]), int(x.split("-")[0]))):
        print(f"DOM {dom_key}: total charge = {dom_charge_total[dom_key]:.2f} PE")

        
if __name__ == "__main__":
    # Uncomment the following line to run the script with profiling
    # profile_code()
    main()
    
    
    
    