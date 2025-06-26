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

def calculate_expected_arrival_time(x0, y0, z0, zenith, azimuth, pos_array, dom_string, dom_number, 
                                    track_start_time=0, c=0.2998, n=1.33):
    """
    Calculates the expected arrival time of a Cherenkov photon from a muon track to a DOM,
    accounting for muon propagation and Cherenkov emission angle.
    
    track_start_time: in ns, optional offset if DOM times are relative to event start.
    """
    # Get DOM position
    dom_mask = (pos_array[:, 3] == dom_string) & (pos_array[:, 4] == dom_number)
    if not np.any(dom_mask):
        return np.inf
    dom_pos = pos_array[dom_mask][0, :3]

    # Muon direction vector (unit)
    v = np.array([
        np.sin(zenith) * np.cos(azimuth),
        np.sin(zenith) * np.sin(azimuth),
        np.cos(zenith)
    ])
    
    r0 = np.array([x0, y0, z0])  # muon start position
    r_dom = dom_pos
    r_rel = r_dom - r0

    proj_length = np.dot(r_rel, v)  # projection of DOM onto track direction
    r_para = proj_length * v
    r_perp = r_rel - r_para
    d_perp = np.linalg.norm(r_perp)

    theta_c = np.arccos(1 / n)
    d_track = d_perp / np.tan(theta_c)  # how far along the track the emission occurs

    emission_point = r0 + v * d_track
    d_mu = d_track
    d_gamma = np.linalg.norm(r_dom - emission_point)

    t_mu = d_mu / c
    t_gamma = n * d_gamma / c
    expected_time = track_start_time + t_mu + t_gamma

    # Debugging output
    print(f"[DEBUG] DOM ({dom_string}, {dom_number}) | d_mu={d_mu:.2f} m | d_gamma={d_gamma:.2f} m")
    print(f"[DEBUG] t_mu={t_mu:.2f} ns | t_gamma={t_gamma:.2f} ns | expected_time={expected_time:.2f} ns")

    return expected_time

def compute_t0_from_earliest_photon(pulse_rows, pos_array, event, col_string, col_dom_number, col_dom_time,
                                    c=0.2998, n=1.33):
    earliest_row = min(pulse_rows, key=lambda row: row[col_dom_time])
    dom_string = int(earliest_row[col_string])
    dom_number = int(earliest_row[col_dom_number])
    dom_time = earliest_row[col_dom_time]

    dom_mask = (pos_array[:, 3] == dom_string) & (pos_array[:, 4] == dom_number)
    if not np.any(dom_mask):
        return None
    dom_pos = pos_array[dom_mask][0, :3]

    v = np.array([
        np.sin(event.zenith) * np.cos(event.azimuth),
        np.sin(event.zenith) * np.sin(event.azimuth),
        np.cos(event.zenith)
    ])
    r0 = np.array([event.position_x, event.position_y, event.position_z])
    r_rel = dom_pos - r0
    proj_length = np.dot(r_rel, v)
    r_perp = r_rel - proj_length * v
    d_perp = np.linalg.norm(r_perp)
    theta_c = np.arccos(1 / n)
    d_track = d_perp / np.tan(theta_c)
    emission_point = r0 + v * d_track
    d_gamma = np.linalg.norm(dom_pos - emission_point)
    t_gamma = n * d_gamma / c
    t_mu = d_track / c
    return dom_time - (t_gamma + t_mu)



def process_event_chunk(chunk, pulses_df_np, pos_array, pulse_event_col, col_string, col_dom_number, col_charge, col_dom_time,
                        distance_lower, upper_bounds, timing_cut_ns=380):
    """
    Process a single chunk of events, now with a physics-based timing cut 
    to reduce scattered light based on Cherenkov emission geometry.
    """
    # Preprocess pulse data by event
    event_map = preprocess_pulses(pulses_df_np, pulse_event_col)
    pos_strings = pos_array[:, 3].astype(int)
    pos_dom_numbers = pos_array[:, 4].astype(int)

    accum = {ub: {"hits_79": {}, "chg_79": {}, "hits_80": {}, "chg_80": {}} for ub in upper_bounds}

    for event in chunk.itertuples(index=False):
        event_no = event.event_no
        min_dists = calculate_minimum_distance_to_track(
            pos_array,
            event.position_x, event.position_y, event.position_z,
            event.zenith, event.azimuth, event.track_length
        )

        pulse_dict = {}
        if event_no in event_map:
            
            t0 = compute_t0_from_earliest_photon(
                pulse_rows=event_map[event_no],
                pos_array=pos_array,
                event=event,
                col_string=col_string,
                col_dom_number=col_dom_number,
                col_dom_time=col_dom_time
            )
            for row in event_map[event_no]:
                dom_string = int(row[col_string])
                dom_number = int(row[col_dom_number])
                dom_time = row[col_dom_time]

                expected_time = calculate_expected_arrival_time(
                    event.position_x, event.position_y, event.position_z,
                    event.zenith, event.azimuth,
                    pos_array,
                    dom_string, dom_number,
                    track_start_time=0
                )
                dom_shifted_time = dom_time - t0
                expected_time_shifted = expected_time 
                delta_t = abs(dom_shifted_time - expected_time_shifted)
                print(f"DOM time: {dom_shifted_time:.2f} ns | Expected time: {expected_time_shifted:.2f} ns |Earliest time : {t0:.2f} | Δt: {delta_t:.2f} ns")

                # Apply physics-based timing cut
                if delta_t <= timing_cut_ns:
                    
                    key = (dom_string, dom_number)
                    
                    pulse_dict[key] = pulse_dict.get(key, 0) + row[col_charge]

        for ub in upper_bounds:
            mask = (min_dists >= distance_lower) & (min_dists < ub)
            for i, valid in enumerate(mask):
                if valid:
                    dom_num = int(pos_dom_numbers[i])
                    s_id = int(pos_strings[i])
                    if s_id == 80:
                        accum[ub]["hits_80"][dom_num] = accum[ub]["hits_80"].get(dom_num, 0) + 1
                        accum[ub]["chg_80"][dom_num] = accum[ub]["chg_80"].get(dom_num, 0) + pulse_dict.get((80, dom_num), 0)
                    else:
                        key = f"{s_id}-{dom_num}"
                        accum[ub]["hits_79"][key] = accum[ub]["hits_79"].get(key, 0) + 1
                        accum[ub]["chg_79"][key] = accum[ub]["chg_79"].get(key, 0) + pulse_dict.get((s_id, dom_num), 0)

    return accum



def process_event_chunk_wrapper(args):
    return process_event_chunk(*args)


def compute_dom_ratio_all_ranges(distance_lower, upper_bounds, event_chunks, pulses_df, pos_array):
    """
    Parallelized RIDE calculation.
    
    Returns:
      - ratio_all: per-DOM RIDE ratios (string X / string 80)
      - hit_counts_all: DOM-level hit counts per distance bin (only string X DOMs)
    """
    pulses_df_np = pulses_df.to_numpy()
    pulse_event_col = pulses_df.columns.get_loc("event_no")
    col_string = pulses_df.columns.get_loc("string")
    col_dom_number = pulses_df.columns.get_loc("dom_number")
    col_dom_time = pulses_df.columns.get_loc("dom_time")
    col_charge = pulses_df.columns.get_loc("charge")

    merged_accum = {ub: {"hits_79": {}, "chg_79": {}, "hits_80": {}, "chg_80": {}} for ub in upper_bounds}

    with ProcessPoolExecutor(max_workers=12) as executor:
        args_list = [
            (chunk, pulses_df_np, pos_array, pulse_event_col,
             col_string, col_dom_number, col_charge, col_dom_time, distance_lower, upper_bounds)
            for chunk in event_chunks
        ]
        results = list(tqdm(executor.map(process_event_chunk_wrapper, args_list), total=len(args_list)))

    # Merge results
    for result in results:
        for ub in upper_bounds:
            for key in ["hits_79", "chg_79", "hits_80", "chg_80"]:
                for dom, val in result[ub][key].items():
                    merged_accum[ub][key][dom] = merged_accum[ub][key].get(dom, 0) + val

    # Compute final ratio and hit count maps
    ratio_all = {}
    hit_counts_all = {}

    for ub in upper_bounds:
        hits_79 = merged_accum[ub]["hits_79"]
        chg_79 = merged_accum[ub]["chg_79"]
        hits_80 = merged_accum[ub]["hits_80"]
        chg_80 = merged_accum[ub]["chg_80"]

        ratio_dict = {}
        hit_count_bin = {}

        for dom_key in hits_79:
            try:
                string_id, dom_number = dom_key.split("-")
                dom_number = int(dom_number)

                if dom_number not in hits_80:
                    continue  # no reference DOM hit for comparison

                hits_x = hits_79[dom_key]
                hits_y = hits_80.get(dom_number, 0)

                if hits_x == 0 or hits_y == 0:
                    continue

                ride_x = chg_79.get(dom_key, 0) / hits_x
                ride_y = chg_80.get(dom_number, 0) / hits_y

                if ride_y > 0:
                    ratio_dict[dom_key] = ride_x / ride_y
                    hit_count_bin[dom_key] = hits_x  # Only from string X
            except Exception as e:
                print(f"[Warning] Could not compute ratio for {dom_key}: {e}")

        ratio_all[ub] = ratio_dict
        hit_counts_all[ub] = hit_count_bin  # Excludes string 80 by design

    return ratio_all, hit_counts_all, {
        ub: merged_accum[ub]["chg_79"] for ub in upper_bounds
    } , {
    ub: merged_accum[ub]["chg_80"] for ub in upper_bounds
    }


def plot_dom_ratio_heatmap_time(ratio_results, output_dir, filename="dom_ratio_heatmap_timing_filtered.png"):
    """
    Plot a heatmap of DOM RIDE ratios (string X / string 80) using timing-filtered pulses.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    import os

    upper_bounds = sorted(ratio_results.keys())
    all_dom_keys = sorted(set().union(*(ratio_results[ub].keys() for ub in upper_bounds)))

    # Sort DOMs by dom_number then string for grouped display
    sorted_doms = sorted(all_dom_keys, key=lambda x: (int(x.split('-')[1]), int(x.split('-')[0])))

    # Build 2D matrix: rows = DOMs, cols = upper_bounds
    data_matrix = np.full((len(sorted_doms), len(upper_bounds)), np.nan)
    for j, ub in enumerate(upper_bounds):
        for i, dom_key in enumerate(sorted_doms):
            if dom_key in ratio_results[ub]:
                data_matrix[i, j] = ratio_results[ub][dom_key]

    # Plot
    plt.figure(figsize=(14, 10))
    norm = colors.TwoSlopeNorm(vmin=0.9, vcenter=1.35, vmax=1.5)
    im = plt.imshow(data_matrix, aspect='auto', origin='lower', cmap='bwr', norm=norm)

    plt.colorbar(im, label="RIDE Ratio (String X / String 80)")
    plt.xlabel("Distance Upper Bound (m)")
    plt.ylabel("DOM (string-dom)")
    plt.title("RIDE Ratios (Timing-Filtered Pulses)")

    plt.xticks(np.arange(len(upper_bounds)), upper_bounds)
    plt.yticks(np.arange(len(sorted_doms)), sorted_doms)

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
def plot_combined_dom_and_string_heatmaps(hit_counts_all, output_dir, filename="combined_participation_heatmap.png"):
    """
    Plots two heatmaps side-by-side:
    - Left: DOM-level participation (string-dom on Y-axis, distance bins on X-axis)
    - Right: String-level participation (count of DOMs from each string per bin)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.colors as colors
    import os

    upper_bounds = sorted(hit_counts_all.keys())
    all_dom_keys = sorted(set().union(*(hit_counts_all[ub].keys() for ub in upper_bounds)))
    sorted_doms = sorted(all_dom_keys, key=lambda x: (int(x.split('-')[1]), int(x.split('-')[0])))

    # DOM-level matrix
    dom_matrix = np.zeros((len(sorted_doms), len(upper_bounds)))
    for j, ub in enumerate(upper_bounds):
        for i, dom_key in enumerate(sorted_doms):
            if dom_key in hit_counts_all[ub]:
                dom_matrix[i, j] = hit_counts_all[ub][dom_key]

    # String-level matrix
    all_strings = sorted(set(int(dom.split('-')[0]) for dom in sorted_doms))
    string_index = {s: i for i, s in enumerate(all_strings)}
    string_matrix = np.zeros((len(all_strings), len(upper_bounds)))
    for j, ub in enumerate(upper_bounds):
        for dom_key in hit_counts_all[ub]:
            s_id = int(dom_key.split('-')[0])
            i = string_index[s_id]
            string_matrix[i, j] += 1

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(18, 10), sharex=True)

    # DOM heatmap
    dom_norm = colors.LogNorm(vmin=1, vmax=np.nanmax(dom_matrix))
    im1 = axs[0].imshow(dom_matrix, aspect='auto', origin='lower', cmap='plasma', norm=dom_norm)
    axs[0].set_title("DOM Participation (string-dom)")
    axs[0].set_yticks(np.arange(len(sorted_doms)))
    axs[0].set_yticklabels(sorted_doms)
    axs[0].set_ylabel("DOM (string-dom)")
    axs[0].set_xlabel("Distance Upper Bound (m)")
    axs[0].set_xticks(np.arange(len(upper_bounds)))
    axs[0].set_xticklabels(upper_bounds)
    plt.colorbar(im1, ax=axs[0], label="DOM Hit Count")

    # String heatmap
    im2 = axs[1].imshow(string_matrix, aspect='auto', origin='lower', cmap='plasma')
    axs[1].set_title("String Participation (Number of DOMs per Bin)")
    axs[1].set_yticks(np.arange(len(all_strings)))
    axs[1].set_yticklabels(all_strings)
    axs[1].set_xlabel("Distance Upper Bound (m)")
    axs[1].set_xticks(np.arange(len(upper_bounds)))
    axs[1].set_xticklabels(upper_bounds)
    plt.colorbar(im2, ax=axs[1], label="DOM Count")

    plt.suptitle("DOM and String Participation Heatmaps by Distance Bin", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    

def plot_dom_charge_heatmap(charge_79_all, charge_80_all, output_dir, filename="dom_charge_heatmap_time.png", log_scale=True):
    """
    Plots a heatmap of total DOM charge per distance bin for both string X and string 80 DOMs.

    Parameters:
        charge_79_all: dict {ub: {string-dom: charge}}
        charge_80_all: dict {ub: {dom_number: charge}} → will be reformatted as '80-dom'
        output_dir: str
        filename: str
        log_scale: bool
    """
    # Combine both charge dicts using "string-dom" key format
    charge_all = {}
    for ub in sorted(charge_79_all.keys()):
        combined = charge_79_all[ub].copy()
        for dom, val in charge_80_all.get(ub, {}).items():
            combined[f"80-{dom}"] = val
        charge_all[ub] = combined

    upper_bounds = sorted(charge_all.keys())
    all_dom_keys = sorted(set().union(*(charge_all[ub].keys() for ub in upper_bounds)))
    sorted_doms = sorted(all_dom_keys, key=lambda x: (int(x.split('-')[1]), int(x.split('-')[0])))

    data_matrix = np.full((len(sorted_doms), len(upper_bounds)), 0.0)
    for j, ub in enumerate(upper_bounds):
        for i, dom_key in enumerate(sorted_doms):
            if dom_key in charge_all[ub]:
                data_matrix[i, j] = charge_all[ub][dom_key]

    # Plot
    plt.figure(figsize=(14, 10))
    if log_scale:
        norm = colors.LogNorm(vmin=1e-2, vmax=np.max(data_matrix))
    else:
        norm = None
    im = plt.imshow(data_matrix, aspect='auto', origin='lower', cmap='inferno', norm=norm)

    plt.colorbar(im, label="Total Charge (PE)")
    plt.xlabel("Distance Upper Bound (m)")
    plt.ylabel("DOM (string-dom_number)")
    plt.title("Total Charge per DOM per Distance Bin (including String 80)")

    plt.xticks(np.arange(len(upper_bounds)), upper_bounds)
    plt.yticks(np.arange(len(sorted_doms)), sorted_doms)

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def main():
    # Load data from the filtered DB.
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
    (pulses_df['dom_number'] < 11)
    ]
    # pulses_df = pulses_df[
    # ((pulses_df['string'] == 84)) |
    # ((pulses_df['string'] == 80) & (pulses_df['rde'] == 1.0))
    # ]

    # # Find dom_numbers present in filtered string 80
    # doms_in_80 = set(pulses_df[pulses_df['string'] == 80]['dom_number'].unique())

    # # Keep only DOMs from string 84 that also exist in string 80
    # pulses_df = pulses_df[~(
    #     (pulses_df['string'] == 84) & 
    #     (~pulses_df['dom_number'].isin(doms_in_80))
    # )]
    # Use all unique DOMs (so monitor from string 80 is available).
    pos_array = pulses_df[['dom_x','dom_y','dom_z','string','dom_number']].drop_duplicates().to_numpy()
    
    # Split truth events into chunks.
    event_chunks = np.array_split(df_truth, 12)
    
    distance_lower = 10
    upper_bounds = np.arange(15, 161, 5)  # 20, 30, 40, ... 160
    output_dir = '/groups/icecube/simon/GNN/workspace/Plots/'
    # Compute the per-DOM ratios for all distance ranges.
    ratio_results,hit_counts_all,chg_79_all, chg_80_all  = compute_dom_ratio_all_ranges(distance_lower, upper_bounds, event_chunks, pulses_df, pos_array)

    plot_dom_ratio_heatmap_time(ratio_results, output_dir)
    plot_combined_dom_and_string_heatmaps(hit_counts_all, output_dir)
    #plot_dom_hit_heatmap(hit_counts_all, output_dir,log_scale=False)
    plot_dom_charge_heatmap(chg_79_all,chg_80_all, output_dir, log_scale=False)
    # Print results for each upper bound.
    for ub in sorted(ratio_results.keys()):
        print(f"Distance range: {distance_lower} to {ub}")
        ratio_dict = ratio_results[ub]
        for dom in sorted(ratio_dict.keys()):
            print(f"  DOM {dom}: Ratio = {ratio_dict[dom]}")
        print("-" * 40)


if __name__ == "__main__":
    main()