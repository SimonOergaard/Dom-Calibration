import os
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool

def calculate_minimum_distances(pos_array, true_x, true_y, true_z, true_zenith, true_azimuth, track_length, steps=1000):
    steps = np.linspace(0, track_length, steps)
    line_array = np.array([true_x, true_y, true_z]) + np.array(
        [np.sin(true_zenith) * np.cos(true_azimuth),
         np.sin(true_zenith) * np.sin(true_azimuth),
         np.cos(true_zenith)]
    ) * steps[:, np.newaxis]

    expanded_pos_array = np.expand_dims(pos_array, axis=2)
    line_array = np.swapaxes(line_array, 0, -1)
    dist_array = expanded_pos_array - line_array

    dist_array = np.sqrt(np.sum(np.square(dist_array), axis=1))
    min_distances = np.min(dist_array, axis=1)
    return min_distances

def process_events(event_chunk, pos_array):
    all_min_distances = []
    event_numbers = []

    for _, event in tqdm(event_chunk.iterrows(), total=len(event_chunk), desc="Processing Events"):
        true_x, true_y, true_z = event[['position_x', 'position_y', 'position_z']]
        true_zenith, true_azimuth = event[['zenith', 'azimuth']]
        track_length = event['track_length']

        min_distances = calculate_minimum_distances(
            pos_array, true_x, true_y, true_z, true_zenith, true_azimuth, track_length
        )
        all_min_distances.append(min_distances)
        event_numbers.append(event['event_no'])

    return np.array(all_min_distances), np.array(event_numbers)

def compute_ride_for_distance_range_parallel(args):
    lower_bound, upper_bound, min_distances, event_numbers, pulses_df, pos_array = args

    expected_hits = np.zeros(pos_array.shape[0])
    total_charge = np.zeros(pos_array.shape[0])
    ride_values = []

    for event_idx, event_no in enumerate(event_numbers):
        doms_in_range_mask = (lower_bound <= min_distances[event_idx]) & (min_distances[event_idx] < upper_bound)
        expected_hits[doms_in_range_mask] += 1

        event_pulses = pulses_df[pulses_df['event_no'] == event_no]

        #unique_doms = event_pulses.drop_duplicates(subset=['dom_x', 'dom_y', 'dom_z'])
        #unique_doms_array = unique_doms[['dom_x', 'dom_y', 'dom_z']].to_numpy()
        #unique_charges = unique_doms['charge'].to_numpy()
        # Group by unique DOMs and sum their charges
        grouped_doms = event_pulses.groupby(['dom_x', 'dom_y', 'dom_z']).agg({'charge': 'sum'}).reset_index()
        unique_doms_array = grouped_doms[['dom_x', 'dom_y', 'dom_z']].to_numpy()
        unique_charges = grouped_doms['charge'].to_numpy()
        for dom_id, charge in zip(unique_doms_array, unique_charges):
            idx = np.where((pos_array == dom_id).all(axis=1))[0]
            if len(idx) > 0 and doms_in_range_mask[idx[0]]:
                total_charge[idx[0]] += charge

    valid_indices = expected_hits > 0
    ride_values_array = np.zeros_like(expected_hits, dtype=float)
    ride_values_array[valid_indices] = total_charge[valid_indices] / expected_hits[valid_indices]

    ride_values.extend(ride_values_array[valid_indices])

    mean_ride = np.mean(ride_values) if ride_values else 0
    median_ride = np.median(ride_values) if ride_values else 0

    # Debugging for specific bins
    if lower_bound == 5 and upper_bound == 40:
        print(f"Bin ({lower_bound}, {upper_bound}):")
        print(f"Expected Hits: {expected_hits}")
        print(f"Total Charge: {total_charge}")
        print(f"Mean RIDE: {mean_ride}")
        print(f"Median RIDE: {median_ride}")

    return mean_ride, median_ride

def compute_ride_for_distance_ranges(min_distances, event_numbers, pulses_df, pos_array, bounds, num_workers):
    mean_ride_matrix = np.zeros((len(bounds), len(bounds)))
    median_ride_matrix = np.zeros((len(bounds), len(bounds)))

    tasks = [
        (bounds[i], bounds[j], min_distances, event_numbers, pulses_df, pos_array)
        for i in range(len(bounds))
        for j in range(i + 1, len(bounds))
    ]

    with Pool(num_workers) as pool:
        results = pool.map(compute_ride_for_distance_range_parallel, tasks)

    idx = 0
    for i in range(len(bounds)):
        for j in range(i + 1, len(bounds)):
            mean_ride_matrix[i, j], median_ride_matrix[i, j] = results[idx]
            idx += 1
    
    return mean_ride_matrix, median_ride_matrix

def plot_combined_heatmaps(bounds, mean_ride_nqe, mean_ride_hqe, mean_ride_upgrade, focus_depth_bin):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True)

    extent = [bounds[0], bounds[-1], bounds[0], bounds[-1]]

    # Mask the arrays where values are 0 or non-existing
    masked_nqe = np.ma.masked_where(mean_ride_nqe == 0, mean_ride_nqe)
    masked_hqe = np.ma.masked_where(mean_ride_hqe == 0, mean_ride_hqe)
    masked_upgrade = np.ma.masked_where(mean_ride_upgrade == 0, mean_ride_upgrade)

    cmap = plt.cm.coolwarm
    cmap.set_bad(color='black')

    ax = axes[0]
    heatmap_nqe = ax.imshow(
        masked_nqe,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=extent,
        vmin=0.8,
        vmax=1.2,
    )
    fig.colorbar(heatmap_nqe, ax=ax, label="Mean RIDE")
    ax.set_title(f"NQE at depths [{focus_depth_bin}]")
    ax.set_xlabel("Max DOM Distance")
    ax.set_ylabel("Min DOM Distance")

    ax = axes[1]
    heatmap_hqe = ax.imshow(
        masked_hqe,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=extent,
        vmin=1.1,
        vmax=1.6,
    )
    fig.colorbar(heatmap_hqe, ax=ax, label="Mean RIDE")
    ax.set_title(f"HQE at depths [{focus_depth_bin}]")
    ax.set_xlabel("Max DOM Distance")
    ax.set_ylabel("Min DOM Distance")

    ax = axes[2]
    heatmap_upgrade = ax.imshow(
        masked_upgrade,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=extent,
        vmin=0.6,
        vmax=1.2,
    )
    fig.colorbar(heatmap_upgrade, ax=ax, label="Mean RIDE")
    ax.set_title(f"Upgrade at depths [{focus_depth_bin}]")
    ax.set_xlabel("Max DOM Distance")
    ax.set_ylabel("Min DOM Distance")

    pdf_filename = f"combined_charge_fraction_depth_{focus_depth_bin}.pdf"
    plt.savefig(pdf_filename, format="pdf")
    print(f"Saved combined plot: {pdf_filename}")

def main():
    file_path = '/groups/icecube/simon/GNN/workspace/filtered.db'
    con = sqlite3.connect(file_path)
    df_truth = pd.read_sql_query("SELECT * FROM truth", con)
    pulses_df = pd.read_sql_query("SELECT * FROM SplitInIcePulsesSRT", con)
    con.close()

    df_truth = df_truth.iloc[:10000]
    filtered_events = df_truth['event_no'].unique()
    pulses_df = pulses_df[pulses_df['event_no'].isin(filtered_events)]

    all_depth_bins = np.arange(-500, 0, 10)
    pulses_df['depth_bin'] = pd.cut(pulses_df['dom_z'], bins=all_depth_bins, labels=all_depth_bins[:-1])
    upgrade_df = pulses_df[pulses_df['string'] >= 87.0]
    pulses_df = pulses_df[pulses_df['string'] < 87.0]

    focus_depth_bin = -320
    filtered_doms = pulses_df[pulses_df['depth_bin'] == focus_depth_bin]
    filtered_upgrade_doms = upgrade_df[upgrade_df['depth_bin'] == focus_depth_bin]
    filtered_HQE = filtered_doms[filtered_doms['rde'] == 1.35]
    filtered_NQE = filtered_doms[filtered_doms['rde'] == 1.0]

    pos_array_HQE = filtered_HQE[['dom_x', 'dom_y', 'dom_z']].drop_duplicates().to_numpy()
    pos_array_NQE = filtered_NQE[['dom_x', 'dom_y', 'dom_z']].drop_duplicates().to_numpy()
    pos_array_upgrade = filtered_upgrade_doms[['dom_x', 'dom_y', 'dom_z']].drop_duplicates().to_numpy()
    print(f"Number of DOMs in upgrade: {len(pos_array_upgrade)}")
    num_workers = 8
    event_chunks = np.array_split(df_truth, num_workers)

    with Pool(num_workers) as pool:
        hqe_results = pool.starmap(process_events, [(chunk, pos_array_HQE) for chunk in event_chunks])
        min_distances_HQE, event_numbers_HQE = zip(*hqe_results)
        min_distances_HQE = np.vstack(min_distances_HQE)
        event_numbers_HQE = np.concatenate(event_numbers_HQE)

        nqe_results = pool.starmap(process_events, [(chunk, pos_array_NQE) for chunk in event_chunks])
        min_distances_NQE, event_numbers_NQE = zip(*nqe_results)
        min_distances_NQE = np.vstack(min_distances_NQE)
        event_numbers_NQE = np.concatenate(event_numbers_NQE)

        upgrade_results = pool.starmap(process_events, [(chunk, pos_array_upgrade) for chunk in event_chunks])
        min_distances_upgrade, event_numbers_upgrade = zip(*upgrade_results)
        min_distances_upgrade = np.vstack(min_distances_upgrade)
        event_numbers_upgrade = np.concatenate(event_numbers_upgrade)

    bounds = np.arange(0, 205, 5)

    mean_ride_nqe, median_ride_nqe = compute_ride_for_distance_ranges(
        min_distances_NQE, event_numbers_NQE, pulses_df, pos_array_NQE, bounds, num_workers
    )
    mean_ride_hqe, median_ride_hqe = compute_ride_for_distance_ranges(
        min_distances_HQE, event_numbers_HQE, pulses_df, pos_array_HQE, bounds, num_workers
    )
    mean_ride_upgrade, median_ride_upgrade = compute_ride_for_distance_ranges(
        min_distances_upgrade, event_numbers_upgrade, upgrade_df, pos_array_upgrade, bounds, num_workers
    )

    normalized_mean_ride_nqe = np.divide(mean_ride_nqe, median_ride_nqe, out=np.zeros_like(mean_ride_nqe), where=median_ride_nqe > 0)
    normalized_mean_ride_hqe = np.divide(mean_ride_hqe, median_ride_nqe, out=np.zeros_like(mean_ride_hqe), where=median_ride_nqe > 0)
    normalized_mean_ride_upgrade = np.divide(mean_ride_upgrade, median_ride_nqe, out=np.zeros_like(mean_ride_upgrade), where=median_ride_nqe > 0)

    plot_combined_heatmaps(bounds, normalized_mean_ride_nqe, normalized_mean_ride_hqe, normalized_mean_ride_upgrade, focus_depth_bin)

if __name__ == "__main__":
    main()
