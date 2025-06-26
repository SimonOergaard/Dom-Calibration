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


def process_event_chunk(chunk, pulses_df_np, pos_array, pulse_event_col, col_string, col_dom_number, col_charge,
                        distance_lower, upper_bounds):
    """
    Process a single chunk of events. Used for parallel execution.
    """
    # Preprocess pulse data by event.
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
            for row in event_map[event_no]:
                key = (int(row[col_string]), int(row[col_dom_number]))
                pulse_dict[key] = pulse_dict.get(key, 0) + row[col_charge]

        for ub in upper_bounds:
            mask = (min_dists >= distance_lower) & (min_dists < ub)
            for i, valid in enumerate(mask):
                if valid:
                    dom_num = int(pos_dom_numbers[i])
                    s_id = int(pos_strings[i])
                    monitor_id = 79
                    if s_id == monitor_id:
                        accum[ub]["hits_80"][dom_num] = accum[ub]["hits_80"].get(dom_num, 0) + 1
                        accum[ub]["chg_80"][dom_num] = accum[ub]["chg_80"].get(dom_num, 0) + pulse_dict.get((monitor_id, dom_num), 0)
                    else:
                        key = f"{s_id}-{dom_num}"
                        accum[ub]["hits_79"][key] = accum[ub]["hits_79"].get(key, 0) + 1
                        accum[ub]["chg_79"][key] = accum[ub]["chg_79"].get(key, 0) + pulse_dict.get((s_id, dom_num), 0)

    return accum




def process_event_chunk_wrapper(args):
    return process_event_chunk(*args)

def compute_dom_ratio_all_ranges(distance_lower, upper_bounds, event_chunks, pulses_df, pos_array):
    """
    Parallelized RIDE calculation. Computes both:
      - ratio_all: per-DOM RIDE ratios (string X / string 80)
      - hit_counts_all: total hit counts per DOM per bin
    """
    pulses_df_np = pulses_df.to_numpy()
    pulse_event_col = pulses_df.columns.get_loc("event_no")
    col_string = pulses_df.columns.get_loc("string")
    col_dom_number = pulses_df.columns.get_loc("dom_number")
    col_charge = pulses_df.columns.get_loc("charge")

    merged_accum = {ub: {"hits_79": {}, "chg_79": {}, "hits_80": {}, "chg_80": {}} for ub in upper_bounds}

    with ProcessPoolExecutor(max_workers=30) as executor:
        args_list = [
            (chunk, pulses_df_np, pos_array, pulse_event_col,
             col_string, col_dom_number, col_charge, distance_lower, upper_bounds)
            for chunk in event_chunks
        ]
        results = list(tqdm(executor.map(process_event_chunk_wrapper, args_list), total=len(args_list)))

    # Merge results from all chunks
    for result in results:
        for ub in upper_bounds:
            for key in ["hits_79", "chg_79", "hits_80", "chg_80"]:
                for dom, val in result[ub][key].items():
                    if isinstance(val, dict):
                        print(f"[Warning] Unexpected dict in merged_accum[{ub}]['{key}'][{dom}]")
                        continue
                    merged_accum[ub][key][dom] = merged_accum[ub][key].get(dom, 0) + val

    # Final ratio and hit count calculation
    ratio_all = {}
    hit_counts_all = {}

    for ub in upper_bounds:
        ratio_dict = {}
        hit_counts_all[ub] = {}

        hits_other = merged_accum[ub]["hits_79"]
        chg_other = merged_accum[ub]["chg_79"]
        hits80 = merged_accum[ub]["hits_80"]
        chg80 = merged_accum[ub]["chg_80"]

        for dom_key in hits_other:
            try:
                string_id, dom_number = dom_key.split('-')
                dom_number = int(dom_number)

                if dom_number not in hits80:
                    continue

                hits = hits_other[dom_key]
                hits_80 = hits80.get(dom_number, 0)

                if hits == 0 or hits_80 == 0:
                    continue  # skip if no hits to avoid div-by-zero

                # Safely compute total hits
                if isinstance(hits_80, dict):
                    print(f"[Warning] hits80[{dom_number}] is a dict! Skipping.")
                    continue

                hit_counts_all[ub][dom_key] = hits + hits_80

                ride_other = chg_other.get(dom_key, 0) / hits
                ride_80 = chg80.get(dom_number, 0) / hits_80

                if ride_80 > 0:
                    ratio_dict[dom_key] = ride_other / ride_80

            except Exception as e:
                print(f"[Warning] Could not compute ratio for {dom_key}: {e}")

        ratio_all[ub] = ratio_dict
    print(f"Computed ratios for upper bounds: {upper_bounds}")
    return ratio_all, hit_counts_all, {ub: merged_accum[ub]["hits_80"] for ub in upper_bounds}, {ub: merged_accum[ub]["chg_79"] for ub in upper_bounds}, {ub: merged_accum[ub]["chg_80"] for ub in upper_bounds}, {ub: merged_accum[ub]["hits_79"] for ub in upper_bounds}


def compute_grouped_statistics_with_sample_variance(ratio_all, chg_79_all, chg_80_all, hits_79_all, hits_80_all, upper_bounds):
    """
    Compute grouped medians and propagated errors by DOM_number,
    including both propagated measurement errors and sample variance errors.
    """
    all_results = []

    for ub in upper_bounds:
        ratio_dict = ratio_all[ub]
        chg79 = chg_79_all[ub]
        chg80 = chg_80_all[ub]
        hits79 = hits_79_all[ub]
        hits80 = hits_80_all[ub]

        grouped = {}

        for dom_key, ratio in ratio_dict.items():
            try:
                string_id, dom_number = dom_key.split("-")
                dom_number = int(dom_number)

                hits_other = hits79.get(dom_key, 0)
                chg_other = chg79.get(dom_key, 0)
                
                hits_string80 = hits80.get(dom_number, 0)
                chg_string80 = chg80.get(dom_number, 0)

                if hits_other == 0 or hits_string80 == 0:
                    continue

                ride_other = chg_other / hits_other
                ride_80 = chg_string80 / hits_string80

                sigma_other = ride_other / np.sqrt(hits_other)
                sigma_80 = ride_80 / np.sqrt(hits_string80)

                ratio_value = ride_other / ride_80

                # Propagated error
                rel_error = np.sqrt((sigma_other/ride_other)**2 + (sigma_80/ride_80)**2)
                absolute_error = ratio_value * rel_error

                if dom_number not in grouped:
                    grouped[dom_number] = {"ratios": [], "propagated_errors": []}
                
                grouped[dom_number]["ratios"].append(ratio_value)
                grouped[dom_number]["propagated_errors"].append(absolute_error)

            except Exception as e:
                print(f"[Warning] Problem processing {dom_key}: {e}")

        # After collecting all DOMs for this upper bound
        for dom_number, data in grouped.items():
            ratios = np.array(data["ratios"])
            propagated_errors = np.array(data["propagated_errors"])
            N = len(ratios)

            if N == 0:
                continue

            # Median ratio
            median_ratio = np.median(ratios)

            # Measurement error: combine individual propagated errors
            median_idx = np.argsort(ratios)[len(ratios) // 2]
            propagated_error_group = propagated_errors[median_idx]

            # Sample variance error
            if N > 1:
                sample_std = np.std(ratios, ddof=1)  # unbiased sample std (N-1)
                sample_variance_error = sample_std / np.sqrt(N)
            else:
                sample_variance_error = np.nan  # undefined for N=1

            # Final conservative choice: take maximum
            if np.isnan(sample_variance_error):
                final_error = propagated_error_group
            else:
                final_error = max(propagated_error_group, sample_variance_error)

            all_results.append({
                "upper_bound": ub,
                "dom_number": dom_number,
                "median_ratio": median_ratio,
                "propagated_error": propagated_error_group,
                "sample_variance_error": sample_variance_error,
                "final_error": final_error,
                "n_DOMs_in_group": N
            })

    return pd.DataFrame(all_results)



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
    #all_doms = sorted(set().union(*(ratio_results[ub].keys() for ub in upper_bounds)))
    all_combined_doms = set().union(*(ratio_results[ub].keys() for ub in upper_bounds))

    # Sort them numerically by dom_number
    def get_dom_number(dom_key):
        return int(dom_key.split("-")[1])

    sorted_doms = sorted(all_combined_doms, key=lambda x: (int(x.split('-')[1]), int(x.split('-')[0])))


    # Build a 2D matrix with rows = DOM numbers and columns = upper bounds.
    data_matrix = np.full((len(all_combined_doms), len(upper_bounds)), np.nan)
    for j, ub in enumerate(upper_bounds):
        for i, key in enumerate(sorted_doms):
            if key in ratio_results[ub]:
                data_matrix[i, j] = ratio_results[ub][key]

    norm = colors.TwoSlopeNorm(vmin=0.9, vcenter=1.35, vmax=1.5)
    # Create the heatmap.
    plt.figure(figsize=(16, 12))
    im = plt.imshow(data_matrix, aspect='auto', origin='lower', cmap='bwr', norm=norm)
    cbar = plt.colorbar(im)
    cbar.set_label("RIDE Value", fontsize=18)
    plt.xlabel("Distance Upper Bound (m)", fontsize=20)
    plt.ylabel("DOM (string-dom_number)", fontsize=20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("RIDE of (String X) to (String 80) — DOMs < 11", fontsize=22)

        
    plt.xticks(np.arange(len(upper_bounds)), upper_bounds)
    plt.yticks(np.arange(len(sorted_doms)), sorted_doms)

        
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "dom_ratio_heatmap_deepcore.png"), dpi=300)
    plt.close()
    
def plot_dom_hit_heatmap(hit_counts, output_dir, filename="dom_hit_counts_heatmap.png"):
    upper_bounds = sorted(hit_counts.keys())
    all_dom_keys = sorted(set().union(*(hit_counts[ub].keys() for ub in upper_bounds)))
    sorted_doms = sorted(all_dom_keys, key=lambda x: (int(x.split('-')[1]), int(x.split('-')[0])))

    data_matrix = np.full((len(sorted_doms), len(upper_bounds)), 0)
    for j, ub in enumerate(upper_bounds):
        for i, dom_key in enumerate(sorted_doms):
            if dom_key in hit_counts[ub]:
                data_matrix[i, j] = hit_counts[ub][dom_key]

    plt.figure(figsize=(14, 10))
    im = plt.imshow(data_matrix, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(im, label="Total Event Count (String X + String 80)")
    plt.xlabel("Distance Upper Bound (m)")
    plt.ylabel("DOM (string-dom_number)")
    plt.title("Number of Events Contributing per DOM per Distance Bin")

    plt.xticks(np.arange(len(upper_bounds)), upper_bounds)
    plt.yticks(np.arange(len(sorted_doms)), sorted_doms)

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def plot_dom_hit_heatmap_string80_only(hits_80_by_bin, output_dir, filename="dom_hit_counts_string80.png"):
    upper_bounds = sorted(hits_80_by_bin.keys())

    all_dom_keys = sorted(set().union(*(hits_80_by_bin[ub].keys() for ub in upper_bounds)))
    sorted_doms = sorted(all_dom_keys, key=lambda x: int(x))  # Only dom_number as key

    data_matrix = np.full((len(sorted_doms), len(upper_bounds)), 0)
    for j, ub in enumerate(upper_bounds):
        for i, dom in enumerate(sorted_doms):
            if dom in hits_80_by_bin[ub]:
                data_matrix[i, j] = hits_80_by_bin[ub][dom]

    plt.figure(figsize=(14, 10))
    im = plt.imshow(data_matrix, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(im, label="Event Count (String 80)")
    plt.xlabel("Distance Upper Bound (m)")
    plt.ylabel("DOM Number (String 80)")
    plt.title("DOM Participation Heatmap — String 80 Only")
    plt.xticks(np.arange(len(upper_bounds)), upper_bounds)
    plt.yticks(np.arange(len(sorted_doms)), sorted_doms)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()



def plot_dom_charge_heatmap_all_strings(chg_79_by_bin, chg_80_by_bin, output_dir, filename="dom_charge_all_strings.png"):
    upper_bounds = sorted(chg_79_by_bin.keys())

    all_dom_keys = set()

    for ub in upper_bounds:
        all_dom_keys.update(chg_79_by_bin[ub].keys())
        all_dom_keys.update([f"80-{dom}" for dom in chg_80_by_bin[ub].keys()])

    if not all_dom_keys:
        raise ValueError("No valid DOM keys found for plotting. Input data might be malformed.")

    sorted_doms = sorted(
        all_dom_keys,
        key=lambda x: (int(x.split('-')[1]), int(x.split('-')[0]))
    )

    data_matrix = np.full((len(sorted_doms), len(upper_bounds)), 0.0)

    for j, ub in enumerate(upper_bounds):
        for i, dom_key in enumerate(sorted_doms):
            if dom_key in chg_79_by_bin[ub]:
                data_matrix[i, j] = chg_79_by_bin[ub][dom_key]
            else:
                try:
                    string, dom = dom_key.split('-')
                    if string == "80":
                        dom = int(dom)
                        data_matrix[i, j] = chg_80_by_bin[ub].get(dom, 0)
                except Exception as e:
                    print(f"[Warning] Failed to parse dom_key '{dom_key}': {e}")

    # Plotting
    plt.figure(figsize=(14, 10))
    im = plt.imshow(data_matrix, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(im, label="Total Charge (All Strings)")
    plt.xlabel("Distance Upper Bound (m)")
    plt.ylabel("DOM (string-dom_number)")
    plt.title("Charge Collected per DOM (All Strings)")
    plt.xticks(np.arange(len(upper_bounds)), upper_bounds)
    plt.yticks(np.arange(len(sorted_doms)), sorted_doms)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def plot_recomputed_ride_ratio(hits_79_by_bin, chg_79_by_bin, hits_80_by_bin, chg_80_by_bin, output_dir, filename="dom_ratio_recomputed_79_80.png"):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    import os

    upper_bounds = sorted(hits_79_by_bin.keys())

    all_dom_keys = sorted(set().union(*(hits_79_by_bin[ub].keys() for ub in upper_bounds)))

    # Sort DOMs by DOM number then string for grouped display
    sorted_doms = sorted(all_dom_keys, key=lambda x: (int(x.split('-')[1]), int(x.split('-')[0])))

    data_matrix = np.full((len(sorted_doms), len(upper_bounds)), np.nan)

    for j, ub in enumerate(upper_bounds):
        for i, dom_key in enumerate(sorted_doms):
            try:
                hits_79 = hits_79_by_bin[ub].get(dom_key, 0)
                chg_79 = chg_79_by_bin[ub].get(dom_key, 0)

                string, dom = dom_key.split('-')
                dom = int(dom)

                hits_80 = hits_80_by_bin[ub].get(dom, 0)
                chg_80 = chg_80_by_bin[ub].get(dom, 0)

                if hits_79 > 0 and hits_80 > 0:
                    ride_79 = chg_79 / hits_79
                    ride_80 = chg_80 / hits_80
                    if ride_80 > 0:
                        data_matrix[i, j] = ride_79 / ride_80
            except Exception as e:
                print(f"[Warning] Failed to compute RIDE ratio for {dom_key} in bin {ub}: {e}")

    # Plotting
    plt.figure(figsize=(16, 18))
    norm = colors.TwoSlopeNorm(vmin=0.9, vcenter=1.35, vmax=1.5)
    im = plt.imshow(data_matrix, aspect='auto', origin='lower', cmap='bwr', norm=norm)
    cbar = plt.colorbar(im)
    cbar.set_label("RIDE Ratio (String X / String 80)", fontsize=18)
    
    plt.xlabel("Distance Upper Bound (m)", fontsize=20)
    plt.ylabel("DOM (string-dom)", fontsize=20)
    plt.title("RIDE of (String X) to (String 80) - DOMS < 11", fontsize=22)

    plt.xticks(np.arange(len(upper_bounds)), upper_bounds)
    plt.yticks(np.arange(len(sorted_doms)), sorted_doms)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    

def plot_hit_fractions(hits_79_by_bin, hits_80_by_bin, output_dir, filename="dom_hit_ratio.png"):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    import os

    upper_bounds = sorted(hits_79_by_bin.keys())

    all_dom_keys = sorted(set().union(*(hits_79_by_bin[ub].keys() for ub in upper_bounds)))

    # Sort DOMs by DOM number then string for grouped display
    sorted_doms = sorted(all_dom_keys, key=lambda x: (int(x.split('-')[1]), int(x.split('-')[0])))

    data_matrix = np.full((len(sorted_doms), len(upper_bounds)), np.nan)

    for j, ub in enumerate(upper_bounds):
        for i, dom_key in enumerate(sorted_doms):
            try:
                hits_79 = hits_79_by_bin[ub].get(dom_key, 0)
                

                string, dom = dom_key.split('-')
                dom = int(dom)

                hits_80 = hits_80_by_bin[ub].get(dom, 0)

                if hits_79 > 0 and hits_80 > 0:
                    ride_79 =  hits_79
                    ride_80 =  hits_80
                    if ride_80 > 0:
                        data_matrix[i, j] =ride_80 /ride_79
            except Exception as e:
                print(f"[Warning] Failed to compute RIDE ratio for {dom_key} in bin {ub}: {e}")

    # Plotting
    plt.figure(figsize=(14, 10))
    norm = colors.TwoSlopeNorm(vmin=0.8, vcenter=1, vmax=1.2)
    im = plt.imshow(data_matrix, aspect='auto', origin='lower', cmap='bwr', norm=norm)

    plt.colorbar(im, label="HIT Ratio (String 80 / String 79)")
    plt.xlabel("Distance Upper Bound (m)")
    plt.ylabel("DOM (string-dom)")
    plt.title("Hit Ratios from Accumulated Data")

    plt.xticks(np.arange(len(upper_bounds)), upper_bounds)
    plt.yticks(np.arange(len(sorted_doms)), sorted_doms)

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def plot_ride_vs_distance_for_selected_doms(grouped_df, output_dir, selected_doms=[2, 4, 6, 8, 10]):
    """
    Plot Median RIDE vs Distance Upper Bound for selected DOM_numbers.
    Each line = one DOM number.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_doms)))  # distinct colors for DOMs

    for idx, dom_number in enumerate(selected_doms):
        subset = grouped_df[grouped_df["dom_number"] == dom_number]
        if subset.empty:
            continue
        distance_bins = subset["upper_bound"]
        medians = subset["median_ratio"]
        errors = subset["final_error"]  # use conservative final_error

        plt.errorbar(distance_bins, medians, yerr=errors, fmt='o-', color=colors[idx],
                     label=f"DOM {dom_number}", capsize=4, markersize=5, linewidth=2)

    plt.axhline(1, linestyle='--', color='black', label='Ratio = 1')
    plt.axhline(1.35, linestyle=':', color='gray', label='RIDE Reference 1.35')

    plt.xlabel("Distance Upper Bound (m)", fontsize=22)
    plt.ylabel("Median RIDE (String X / String 80)", fontsize=22)
    plt.title("Median RIDE vs Distance for Selected DOM Numbers", fontsize=24)
    plt.legend(fontsize=16)
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, "ride_vs_distance_selected_doms.png"), dpi=300)
    plt.close()

    print(f"Saved RIDE vs distance plot for selected DOMs to {output_dir}/ride_vs_distance_selected_doms.png")




def plot_ride_vs_distance_for_selected_doms_separate_errors(grouped_df, output_dir, selected_doms=[2, 4, 6, 8, 10]):
    """
    Make two separate plots:
    1. RIDE vs Distance with propagated errors only
    2. RIDE vs Distance with sample variance errors only
    3. Takes a DataFrame grouped by 'dom_number' and 'upper_bound'
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_doms)))  # distinct colors for DOMs

    # --- Plot 1: Propagated errors only ---
    plt.figure(figsize=(14, 8))

    for idx, dom_number in enumerate(selected_doms):
        subset = grouped_df[grouped_df["dom_number"] == dom_number]
        if subset.empty:
            continue
        distance_bins = subset["upper_bound"]
        medians = subset["median_ratio"]
        propagated_errors = subset["propagated_error"]

        plt.errorbar(distance_bins, medians, yerr=propagated_errors, fmt='o-', color=colors[idx],
                     label=f"DOM {dom_number}", capsize=4, markersize=5, linewidth=2)

    plt.axhline(1, linestyle='--', color='black', label='Ratio = 1')
    plt.axhline(1.35, linestyle=':', color='gray', label='RIDE Reference 1.35')

    plt.xlabel("Distance Upper Bound (m)", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel("Median RIDE (String X / String 79)", fontsize=22)
    plt.title("Median RIDE vs Distance — Propagated Errors", fontsize=24)
    plt.legend(fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, "ride_vs_distance_propagated_errors_79.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot with propagated errors to {output_dir}/ride_vs_distance_propagated_errors.png")

    # --- Plot 2: Sample variance errors only ---
    plt.figure(figsize=(14, 8))

    for idx, dom_number in enumerate(selected_doms):
        subset = grouped_df[grouped_df["dom_number"] == dom_number]
        if subset.empty:
            continue
        distance_bins = subset["upper_bound"]
        medians = subset["median_ratio"]
        sample_variance_errors = subset["sample_variance_error"]

        plt.errorbar(distance_bins, medians, yerr=sample_variance_errors, fmt='o-', color=colors[idx],
                     label=f"DOM {dom_number}", capsize=4, markersize=5, linewidth=2)

    plt.axhline(1, linestyle='--', color='black', label='Ratio = 1')
    plt.axhline(1.35, linestyle=':', color='gray', label='RIDE Reference 1.35')

    plt.xlabel("Distance Upper Bound (m)")
    plt.ylabel("Median RIDE (String X / String 80)")
    plt.title("Median RIDE vs Distance — Sample Variance Errors")
    plt.legend(fontsize='small')
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, "ride_vs_distance_sample_variance_errors.png"), dpi=300)
    plt.close()

    print(f"Saved plot with sample variance errors to {output_dir}/ride_vs_distance_sample_variance_errors.png")



def plot_dom_hits_heatmap_other_strings(hits_79_by_bin, output_dir, filename="dom_hits_other_strings.png"):

    upper_bounds = sorted(hits_79_by_bin.keys())

    all_dom_keys = sorted(set().union(*(hits_79_by_bin[ub].keys() for ub in upper_bounds)))
    sorted_doms = sorted(
        [k for k in all_dom_keys if isinstance(k, str) and '-' in k],
        key=lambda x: (int(x.split('-')[1]), int(x.split('-')[0]))
    )

    data_matrix = np.full((len(sorted_doms), len(upper_bounds)), 0.0)
    for j, ub in enumerate(upper_bounds):
        for i, dom_key in enumerate(sorted_doms):
            if dom_key in hits_79_by_bin[ub]:
                data_matrix[i, j] = hits_79_by_bin[ub][dom_key]

    # Plot
    plt.figure(figsize=(14, 10))
    im = plt.imshow(data_matrix, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(im, label="Hit Count (Other Strings)")
    plt.xlabel("Distance Upper Bound (m)")
    plt.ylabel("DOM (string-dom)")
    plt.title("Hit Count — Strings ≠ 80")

    plt.xticks(np.arange(len(upper_bounds)), upper_bounds)
    plt.yticks(np.arange(len(sorted_doms)), sorted_doms)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def plot_event_energy(df_truth, output_path=None, bins=50, log_y=True):
    """
    Plot the energy distribution of events in df_truth.

    Parameters:
    - df_truth: DataFrame with an 'Energy' or 'energy' column (in GeV).
    - output_path: If provided, saves the plot to this path instead of showing it.
    - bins: Number of bins in the histogram.
    - log_y: Whether to use a logarithmic scale on the y-axis.
    """
    energy_col = 'Energy' if 'Energy' in df_truth.columns else 'energy'
    
    energies = df_truth[energy_col]
    
    plt.figure(figsize=(8, 6))
    plt.hist(energies, bins=bins, color='skyblue', edgecolor='black')
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Count')
    plt.title('Event Energy Distribution')
    print(f"length of energy array: {len(energies)}")
    if log_y:
        plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
np.random.seed(42)

def main():
    # Load data from the filtered DB.
    
    #file_path = '/groups/icecube/simon/GNN/workspace/Scripts/filtered_all_big_data.db'
    file_path = '/groups/icecube/simon/GNN/workspace/data/Converted_I3_file/2mill_stopping_muons.db'
  #  file_path_2 = '/groups/icecube/simon/GNN/workspace/data/Stopped_muons/filtered_all_big_data_with_predictions_and_flag.db'
    con = sqlite3.connect(file_path)
    df_truth = pd.read_sql_query("SELECT * FROM truth", con)
    pulses_df = pd.read_sql_query(
        "SELECT dom_x, dom_y, dom_z, charge, event_no, string, dom_time, rde, dom_number FROM SplitInIcePulsesSRT",
        con
    )
    con.close()
    
    #Open DB connection
    # con = sqlite3.connect(file_path_2)

    # # Load prediction table with only needed columns
    # df_truth = pd.read_sql_query(
    #     """
    #     SELECT 
    #         event_no, 
    #         zenith_pred AS zenith, 
    #         azimuth_pred AS azimuth,
    #         position_x_true_scaled AS position_x,
    #         position_y_true_scaled AS position_y,
    #         position_z_true_scaled AS position_z
    #     FROM prediction
    #     """, 
    #     con
    # )

    # # Add track_length = 10000
    # df_truth["track_length"] = 10000

    # # Convert event_no to int to ensure proper matching
    # event_nos = tuple(df_truth["event_no"].astype(int).tolist())

    # # Load pulses only for those event_nos
    # query = f"""
    # SELECT 
    #     dom_x, dom_y, dom_z, charge, event_no, string, dom_time, rde, dom_number 
    # FROM SplitInIcePulsesSRT
    # WHERE event_no IN {event_nos}
    # """
    # pulses_df = pd.read_sql_query(query, con)

    # # Close DB connection
    # con.close()
        
    
    # Restrict pulses to strings 79 and 80.
    # pulses_df = pulses_df[
    # pulses_df['string'].isin([79, 81, 82, 83, 84, 85, 86]) & 
    # (pulses_df['dom_number'] > 10)
    # ]
    
    pulses_df = pulses_df[
    pulses_df['string'].isin([79, 80])
    ]
    
    
    # pulses_df = pulses_df[
    # ((pulses_df['string'] == 84)) |
    # ((pulses_df['string'] == 80) & (pulses_df['rde'] == 1.0))
    # ]

    # # Find dom_numbers present in filtered string 80
    #doms_in_80 = set(pulses_df[pulses_df['string'] == 80]['dom_number'].unique())
    doms_in_80 = set(pulses_df[pulses_df['string'] == 79]['dom_number'].unique())
    # # Keep only DOMs from string 84 that also exist in string 80
    # pulses_df = pulses_df[~(
    #     (pulses_df['string'] == 84) & 
    #     (~pulses_df['dom_number'].isin(doms_in_80))
    # )]

    print(f"Number of unique event_nos in df_truth: {len(df_truth['event_no'].unique())}")
    
    # unique_event_nos = df_truth['event_no'].unique()
    # np.random.shuffle(unique_event_nos)  # Shuffle in-place
    # event_no_split_1, event_no_split_2 = np.array_split(unique_event_nos, 2)

    # #Create matching splits of truth and pulses
    # split_truths = [
    #     df_truth[df_truth['event_no'].isin(event_no_split_1)],
    #     df_truth[df_truth['event_no'].isin(event_no_split_2)]
    # ]

    # split_pulses = [
    #     pulses_df[pulses_df['event_no'].isin(event_no_split_1)],
    #     pulses_df[pulses_df['event_no'].isin(event_no_split_2)]
    # ]

    # Use all unique DOMs for position array (same for both splits
    #pulses_df = pulses_df[pulses_df['string'].isin([84,80])]
    
    # Use all unique DOMs (so monitor from string 80 is available).
    pos_array = pulses_df[['dom_x','dom_y','dom_z','string','dom_number']].drop_duplicates().to_numpy()
    
    # Split truth events into chunks.
    event_chunks = np.array_split(df_truth, 30)
    
    distance_lower = 10
    upper_bounds = np.arange(15, 161, 5)  # 20, 30, 40, ... 160
    output_dir = '/groups/icecube/simon/GNN/workspace/Plots/'
    # Loop through both splits
    # for i, (df_truth_split, pulses_split) in enumerate(zip(split_truths, split_pulses)):
        
    #     # Re-chunk events in each split for parallel processing
    #     event_chunks = np.array_split(df_truth_split, 30)
        
    #     # Compute ratios
    #     ratio_results, hit_counts, hits_80_by_bin, chg_79_by_bin, chg_80_by_bin, hits_79_by_bin = compute_dom_ratio_all_ranges(
    #         distance_lower, upper_bounds, event_chunks, pulses_split, pos_array
    #     )

    #     # Compute stats
    #     grouped_df = compute_grouped_statistics_with_sample_variance(
    #         ratio_results, chg_79_by_bin, chg_80_by_bin, hits_79_by_bin, hits_80_by_bin, upper_bounds
    #     )

    #     # Plot, using split index for output directory
    #     split_output_dir = f"{output_dir.rstrip('/')}_split_deep_{i+1}/"
    #     plot_ride_vs_distance_for_selected_doms_separate_errors(
    #         grouped_df, split_output_dir, selected_doms=[16, 29, 30, 37, 43 ,44 ,54]
    #     )
    #     plot_event_energy(
    #         df_truth_split, output_path=os.path.join(split_output_dir, "event_energy_distribution.png")
    #     )
    # Compute the per-DOM ratios for all distance ranges.
    print("Computing DOM ratios...")
    ratio_results, hit_counts, hits_80_by_bin, chg_79_by_bin, chg_80_by_bin, hits_79_by_bin = compute_dom_ratio_all_ranges(
        distance_lower, upper_bounds, event_chunks, pulses_df, pos_array
    )
    
    grouped_df = compute_grouped_statistics_with_sample_variance(
        ratio_results, chg_79_by_bin, chg_80_by_bin, hits_79_by_bin, hits_80_by_bin, upper_bounds
    )
    # 2. Plot
    #plot_ride_vs_distance_for_selected_doms(grouped_df, output_dir, selected_doms=[2, 4, 6, 8, 10])# [33,32,31,30,29,43,44,42,45,46]
    #plot_ride_vs_distance_for_selected_doms_separate_errors(grouped_df, output_dir, selected_doms= [33,32,31,30,29,43,44,42,45,46])#[16, 29, 30, 37, 43 ,44 ,54])
    #plot_dom_ratio_heatmap(ratio_results, output_dir)
    #plot_dom_hit_heatmap(hit_counts, output_dir)
    #plot_dom_hit_heatmap_string80_only(hits_80_by_bin, output_dir)
    #plot_dom_charge_heatmap_all_strings(chg_79_by_bin, chg_80_by_bin, output_dir)
    plot_recomputed_ride_ratio(hits_79_by_bin, chg_79_by_bin, hits_80_by_bin, chg_80_by_bin, output_dir)
    #plot_dom_hits_heatmap_other_strings(hits_79_by_bin, output_dir)
    #plot_hit_fractions(hits_79_by_bin, hits_80_by_bin, output_dir)
    # Print results for each upper bound.
    # for ub in sorted(ratio_results.keys()):
    #     print(f"Distance range: {distance_lower} to {ub}")
    #     ratio_dict = ratio_results[ub]
    #     for dom in sorted(ratio_dict.keys()):
    #         print(f"  DOM {dom}: Ratio = {ratio_dict[dom]}")
    #     print("-" * 40)


if __name__ == "__main__":
    main()