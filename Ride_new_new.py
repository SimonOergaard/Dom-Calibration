import os
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool


def calculate_minimum_distances(pos_array, true_x, true_y, true_z, true_zenith, true_azimuth, track_length, steps=1000):
    """
    Calculate the minimum distance between DOMs and a muon track.
    """
    steps = np.linspace(0, track_length, steps)
    line_array = np.array([true_x, true_y, true_z]) + np.array(
        [np.sin(true_zenith) * np.cos(true_azimuth),
         np.sin(true_zenith) * np.sin(true_azimuth),
         np.cos(true_zenith)]
    ) * steps[:, np.newaxis]

    # Broadcast DOM positions for distance calculations
    expanded_pos_array = np.expand_dims(pos_array, axis=2)  # Shape: (N_DOMS, 3, 1)
    line_array = np.swapaxes(line_array, 0, -1)  # Shape: (3, 1000)
    dist_array = expanded_pos_array - line_array  # Shape: (N_DOMS, 3, 1000)

    # Calculate distances
    dist_array = np.sqrt(np.sum(np.square(dist_array), axis=1))  # Shape: (N_DOMS, 1000)
    min_distances = np.min(dist_array, axis=1)  # Shape: (N_DOMS)
    return min_distances


def process_events(event_chunk, pos_array):
    """
    Process events in chunks, calculating distances and collecting event-level data.
    """
    all_min_distances = []  # Collect distances for all events
    event_numbers = []  # Collect corresponding event numbers

    for _, event in tqdm(event_chunk.iterrows(), total=len(event_chunk), desc="Processing Events"):
        true_x, true_y, true_z = event[['position_x', 'position_y', 'position_z']]
        true_zenith, true_azimuth = event[['zenith', 'azimuth']]
        track_length = event['track_length']

        # Calculate minimum distances
        min_distances = calculate_minimum_distances(
            pos_array, true_x, true_y, true_z, true_zenith, true_azimuth, track_length
        )
        all_min_distances.append(min_distances)
        event_numbers.append(event['event_no'])

    return np.array(all_min_distances), np.array(event_numbers)



def compute_ride_for_distance_ranges(min_distances, event_numbers, pulses_df, pos_array, bounds, filtered_doms):
    """
    Compute expected hits, charge, mean RIDE, and median RIDE for different distance ranges.
    """
    mean_ride_matrix = np.zeros((len(bounds), len(bounds)))
    median_ride_matrix = np.zeros((len(bounds), len(bounds)))

    for i, lower_bound in enumerate(bounds):
        for j, upper_bound in enumerate(bounds):
            if lower_bound >= upper_bound:
                continue

            expected_hits = np.zeros(pos_array.shape[0])
            total_charge = np.zeros(pos_array.shape[0])
            ride_values = []  # Collect all valid RIDE values

            # Process each event individually
            for event_idx, event_no in enumerate(event_numbers):
                # Identify DOMs within the current distance range
                doms_in_range_mask = (lower_bound <= min_distances[event_idx]) & (min_distances[event_idx] < upper_bound)
                expected_hits[doms_in_range_mask] += 1

                # Filter pulses for the current event
                event_pulses = pulses_df[pulses_df['event_no'] == event_no]

                # Group by unique DOMs
                unique_doms = event_pulses.drop_duplicates(subset=['dom_x', 'dom_y', 'dom_z'])
                unique_doms_array = unique_doms[['dom_x', 'dom_y', 'dom_z']].to_numpy()
                unique_charges = unique_doms['charge'].to_numpy()

                # Accumulate charge for DOMs within the distance range
                for dom_id, charge in zip(unique_doms_array, unique_charges):
                    idx = np.where((pos_array == dom_id).all(axis=1))[0]
                    if len(idx) > 0 and doms_in_range_mask[idx[0]]:
                        total_charge[idx[0]] += charge

            # Calculate RIDE for valid DOMs
            valid_indices = expected_hits > 0
            ride_values_array = np.zeros_like(expected_hits, dtype=float)
            ride_values_array[valid_indices] = total_charge[valid_indices] / expected_hits[valid_indices]

            # Collect valid RIDE values
            ride_values.extend(ride_values_array[valid_indices])

            # Calculate the mean RIDE for this bin
            mean_ride = np.mean(ride_values) if ride_values else 0
            median_ride = np.median(ride_values) if ride_values else 0

            mean_ride_matrix[i, j] = mean_ride
            median_ride_matrix[i, j] = median_ride

            # Debugging for specific bins
            if lower_bound == 5 and upper_bound == 40:
                print(f"Bin ({lower_bound}, {upper_bound}):")
                print(f"Expected Hits: {expected_hits}")
                print(f"Total Charge: {total_charge}")
                print(f"Mean RIDE: {mean_ride}")
                print(f"Median RIDE: {median_ride}")

    # Debug the final matrices
    print("Mean RIDE Matrix:")
    print(mean_ride_matrix)
    print("Median RIDE Matrix:")
    print(median_ride_matrix)
    return mean_ride_matrix, median_ride_matrix



def plot_combined_heatmaps(bounds, mean_ride_nqe, mean_ride_hqe, focus_depth_bin):
    """
    Plot combined heatmaps for NQE and HQE showing the mean RIDE across distance ranges.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), constrained_layout=True)

    extent = [bounds[0], bounds[-1], bounds[0], bounds[-1]]

    # NQE Plot
    ax = axes[0]
    heatmap_nqe = ax.imshow(
        mean_ride_nqe,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        extent=extent,
        vmin=0.8,
        vmax=1.6,
    )
    fig.colorbar(heatmap_nqe, ax=ax, label="Mean RIDE")
    ax.set_title(f"NQE at depths [{focus_depth_bin}]")
    ax.set_xlabel("Max DOM Distance")
    ax.set_ylabel("Min DOM Distance")

    # HQE Plot
    ax = axes[1]
    heatmap_hqe = ax.imshow(
        mean_ride_hqe,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        extent=extent,
        vmin=0.8,
        vmax=1.6,
    )
    fig.colorbar(heatmap_hqe, ax=ax, label="Mean RIDE")
    ax.set_title(f"HQE at depths [{focus_depth_bin}]")
    ax.set_xlabel("Max DOM Distance")
    ax.set_ylabel("Min DOM Distance")

    ax = axes[2]
    heatmap_upgrade = ax.imshow(
        mean_ride_upgrade,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        extent=extent,
        vmin=0.8,
        vmax=1.6,
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

    # Filter DOMs by depth and strings
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
    # Split events into chunks
    num_workers = 8
    event_chunks = np.array_split(df_truth, num_workers)

    # Calculate minimum distances once for NQE and HQE
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
        

    # Define distance ranges
    bounds = np.arange(0, 205, 5)

    # Compute RIDE matrices and median values
    mean_ride_nqe, median_ride_nqe = compute_ride_for_distance_ranges(
        min_distances_NQE, event_numbers_NQE, pulses_df, pos_array_NQE, bounds, filtered_NQE
    )
    mean_ride_hqe, median_ride_hqe = compute_ride_for_distance_ranges(
        min_distances_HQE, event_numbers_HQE, pulses_df, pos_array_HQE, bounds, filtered_HQE
    )
    
    mean_ride_upgrade, median_ride_upgrade = compute_ride_for_distance_ranges(
        min_distances_upgrade, event_numbers_upgrade, pulses_df, pos_array_upgrade, bounds, filtered_upgrade_doms
    )

    # Normalize matrices using their respective median values
    normalized_mean_ride_nqe = np.divide(mean_ride_nqe, median_ride_nqe, out=np.zeros_like(mean_ride_nqe), where=median_ride_nqe > 0)
    normalized_mean_ride_hqe = np.divide(mean_ride_hqe, median_ride_nqe, out=np.zeros_like(mean_ride_hqe), where=median_ride_nqe > 0)
    normalized_mean_ride_upgrade = np.divide(mean_ride_upgrade, median_ride_nqe, out=np.zeros_like(mean_ride_upgrade), where=median_ride_nqe > 0)

    # Plot combined heatmaps
    plot_combined_heatmaps(bounds, normalized_mean_ride_nqe, normalized_mean_ride_hqe, normalized_mean_ride_upgrade, focus_depth_bin)



if __name__ == "__main__":
    main()
