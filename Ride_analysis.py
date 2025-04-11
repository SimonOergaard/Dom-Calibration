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

def compute_weighted_expectation(pos_array, true_x, true_y, true_z,
                              true_zenith, true_azimuth, track_length,
                              photon_speed=0.299792458/1.32, distance_lower=10,
                              cherenkov_angle_deg=41, muon_speed=0.299792458):
    """
    Compute the expected photon arrival time for each DOM based on the Cherenkov emission point.
    The emission point is determined by projecting backwards from the muon stop point such that
    the angle from the emission point to the DOM satisfies the Cherenkov condition.

    Returns:
        expected_times_ns: array of photon arrival times including muon and photon travel time.
            (np.inf for DOMs with distance < distance_lower).
    """
    expected_times_ns = np.full(pos_array.shape[0], np.inf)

    stop_point = np.array([true_x, true_y, true_z])  # endpoint of the muon track
    direction_vector = np.array([
        np.sin(true_zenith)*np.cos(true_azimuth),
        np.sin(true_zenith)*np.sin(true_azimuth),
        np.cos(true_zenith)
    ])

    cos_theta_c = np.cos(np.deg2rad(cherenkov_angle_deg))

    dom_vectors = pos_array[:, :3] - stop_point
    dom_distances = np.linalg.norm(dom_vectors, axis=1)
    dom_unit_vectors = dom_vectors / (dom_distances[:, np.newaxis] + 1e-10)

    # Compute projection distance backwards from stop point
    backward_proj = -dom_distances * cos_theta_c
    clamped_proj = np.clip(backward_proj, -track_length, 0)

    emission_points = stop_point + clamped_proj[:, np.newaxis] * direction_vector
    photon_travel_distances = np.linalg.norm(pos_array[:, :3] - emission_points, axis=1)
    muon_travel_distances = -clamped_proj  # muon travels backward

    muon_travel_times = muon_travel_distances / muon_speed
    photon_travel_times = photon_travel_distances / photon_speed

    total_travel_times_ns = (muon_travel_times + photon_travel_times)

    valid_doms = photon_travel_distances >= distance_lower
    expected_times_ns[valid_doms] = total_travel_times_ns[valid_doms]

    return expected_times_ns



def process_event_chunk(chunk, pulses_df_np, pos_array,
                        pulse_event_col, col_string, col_dom_number, col_charge, col_dom_time,
                        distance_lower, upper_bounds):
    """
    Process a single chunk of events. Only adds charge if DOM is within distance range and pulse is within timing window.
    """
    event_map = preprocess_pulses(pulses_df_np, pulse_event_col)
    pos_strings = pos_array[:, 3].astype(int)
    pos_dom_numbers = pos_array[:, 4].astype(int)

    accum = {ub: {"hits_79": {}, "chg_79": {}, "hits_80": {}, "chg_80": {}} for ub in upper_bounds}
    timing_window_ns = 40000  # allowable time deviation from expected direct light
    filtered_out = 0
    for event in chunk.itertuples(index=False):
        event_no = event.event_no

        # Compute min distances for all DOMs
        min_dists = calculate_minimum_distance_to_track(
            pos_array,
            event.position_x, event.position_y, event.position_z,
            event.zenith, event.azimuth, event.track_length
        )

        # Get expected arrival times (or inf if too close)
        expected_times_ns = compute_weighted_expectation(
            pos_array,
            event.position_x, event.position_y, event.position_z,
            event.zenith, event.azimuth, event.track_length
        )
        # Time-zero is the earliest expected photon arrival time across all DOMs
        #reference_time = np.min(expected_times_ns[np.isfinite(expected_times_ns)])


        if event_no not in event_map:
            continue

        event_pulses = event_map[event_no]
        if len(event_pulses) == 0:
            continue

        #t0 = np.min(event_pulses[:, col_dom_time])
        #centered_times = event_pulses[:, col_dom_time] - reference_time
        finite_mask = np.isfinite(expected_times_ns)
        earliest_dom_idx = np.argmin(expected_times_ns[finite_mask])
        expected_dom_indices = np.where(finite_mask)[0]
        earliest_dom_global_idx = expected_dom_indices[earliest_dom_idx]

        # Compute the emission time of this earliest DOM:
        muon_speed = 0.299792458  # m/ns
        photon_speed = 0.299792458 / 1.32

        muon_travel_time = (
            expected_times_ns[earliest_dom_global_idx] - 
            np.linalg.norm(pos_array[earliest_dom_global_idx, :3] - 
                        (np.array([event.position_x, event.position_y, event.position_z])) 
                        ) / photon_speed
        )
        emission_t0 = muon_travel_time

        # Shift DOM times by t0 = earliest emission time
        centered_times = event_pulses[:, col_dom_time] - emission_t0
        # Build a charge lookup dictionary, filtered by timing cut
        pulse_dict = {}
        expected_times_ns -= emission_t0
        for row, arrival_time in zip(event_pulses, centered_times):
            s_id = int(row[col_string])
            dom = int(row[col_dom_number])
            chg = row[col_charge]

            dom_idx = np.where((pos_strings == s_id) & (pos_dom_numbers == dom))[0]
            if len(dom_idx) == 0:
                print(f"DOM {s_id}-{dom} not found in pos_array.")
                continue
            i = dom_idx[0]

            expected_time = expected_times_ns[i]
            if np.isinf(expected_time):
                print(f"DOM {s_id}-{dom} has infinite expected time.")
                continue
            print(f"DOM: {s_id}-{dom} | Expected: {expected_time:.2f} ns | Arrival: {arrival_time:.2f} ns | Δt = {abs(arrival_time - expected_time):.2f} ns")

            if abs(arrival_time - expected_time) > timing_window_ns:
                filtered_out +=1
                continue

            key = (s_id, dom)
            pulse_dict[key] = pulse_dict.get(key, 0) + chg
            print("Expected times (ns):", np.min(expected_times_ns), np.max(expected_times_ns))
            print("DOM times     (ns):", np.min(centered_times), np.max(centered_times))

        # Accumulate charge for DOMs passing distance + timing cut
        for ub in upper_bounds:
            mask = (min_dists >= distance_lower) & (min_dists < ub)
            for i, valid in enumerate(mask):
                if not valid:
                    continue
                dom_num = pos_dom_numbers[i]
                s_id = pos_strings[i]
                key = (s_id, dom_num)
                chg = pulse_dict.get(key, 0)

                if s_id == 80:
                    accum[ub]["hits_80"][dom_num] = accum[ub]["hits_80"].get(dom_num, 0) + 1
                    accum[ub]["chg_80"][dom_num] = accum[ub]["chg_80"].get(dom_num, 0) + chg
                else:
                    dom_key = f"{s_id}-{dom_num}"
                    accum[ub]["hits_79"][dom_key] = accum[ub]["hits_79"].get(dom_key, 0) + 1
                    accum[ub]["chg_79"][dom_key] = accum[ub]["chg_79"].get(dom_key, 0) + chg
    print(f"DOMs filtered by timing: {filtered_out}")
    return accum

def process_event_chunk_wrapper(args):
    return process_event_chunk(*args)


def compute_dom_ratio_all_ranges(distance_lower, upper_bounds, event_chunks, pulses_df, pos_array):
    """
    Now parallelized across event chunks using ProcessPoolExecutor.
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
        
    for result in results:
        for ub in upper_bounds:
            for key in ["hits_79", "chg_79", "hits_80", "chg_80"]:
                for dom, val in result[ub][key].items():
                    merged_accum[ub][key][dom] = merged_accum[ub][key].get(dom, 0) + val
                    
    # Final ratio calculation
    ratio_all = {}
    for ub in upper_bounds:
        ratio_dict = {}
        hits_other = merged_accum[ub]["hits_79"]
        chg_other = merged_accum[ub]["chg_79"]
        hits80 = merged_accum[ub]["hits_80"]
        chg80 = merged_accum[ub]["chg_80"]
      #  common_doms = set(hits79.keys()) & set(hits80.keys())
        for dom_key in hits_other:
            try:
                string_id, dom_number = dom_key.split('-')
                dom_number = int(dom_number)

                ride_other = chg_other[dom_key] / hits_other[dom_key] if hits_other[dom_key] > 0 else 0
                ride_80 = chg80.get(dom_number, 0) / hits80.get(dom_number, 1) if hits80.get(dom_number, 0) > 0 else 0

                ratio_dict[dom_key] = ride_other / ride_80 if ride_80 > 0 else 0
            except Exception as e:
                print(f"[Warning] Could not compute ratio for {dom_key}: {e}")

        ratio_all[ub] = ratio_dict
    return ratio_all


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
    ratio_results = compute_dom_ratio_all_ranges(distance_lower, upper_bounds, event_chunks, pulses_df, pos_array)

    plot_dom_ratio_heatmap_time(ratio_results, output_dir)
    # Print results for each upper bound.
    for ub in sorted(ratio_results.keys()):
        print(f"Distance range: {distance_lower} to {ub}")
        ratio_dict = ratio_results[ub]
        for dom in sorted(ratio_dict.keys()):
            print(f"  DOM {dom}: Ratio = {ratio_dict[dom]}")
        print("-" * 40)


if __name__ == "__main__":
    main()