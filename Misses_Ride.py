import os
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from tqdm import tqdm


def calculate_minimum_distance_to_track(pos_array, true_x, true_y, true_z, true_zenith, true_azimuth, track_length, steps=1000):
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


def process_events(event_chunk, pulses_df, pos_array, distance_range):
    """
    Process events in chunks, counting expected and actual hits for each DOM.
    """
    expected_hits = np.zeros(pos_array.shape[0])
    actual_hits = np.zeros(pos_array.shape[0])

    for _, event in tqdm(event_chunk.iterrows(), total=len(event_chunk), desc="Processing Events"):
        true_x, true_y, true_z = event[['position_x', 'position_y', 'position_z']]
        true_zenith, true_azimuth = event[['zenith', 'azimuth']]
        track_length = event['track_length']

        # Calculate minimum distances
        min_distances = calculate_minimum_distance_to_track(
            pos_array, true_x, true_y, true_z, true_zenith, true_azimuth, track_length
        )

        # Count expected hits
        expected_hits[(distance_range[0] < min_distances) & (min_distances <= distance_range[1])] += 1
        #expected_hits[(min_distances < distance_range[1])] += 1
        # Filter pulses for this event
        event_pulses = pulses_df[pulses_df['event_no'] == event['event_no']]

        # Group by unique DOMs
        unique_doms = event_pulses[['dom_x', 'dom_y', 'dom_z']].drop_duplicates().to_numpy()
        doms_in_range_mask = (distance_range[0] < min_distances) & (min_distances <= distance_range[1])
        doms_in_range_indices = np.where(doms_in_range_mask)[0]

        # Use these indices to filter the DOMs in pos_array
        doms_in_range = pos_array[doms_in_range_indices]

        # Count actual hits only for DOMs within the distance range
        for dom in unique_doms:
            idx = np.where((doms_in_range == dom).all(axis=1))[0]
            if len(idx) > 0:
                actual_hits[doms_in_range_indices[idx[0]]] += 1

    return expected_hits, actual_hits


def plot_heatmap(dom_positions, expected_hits, actual_hits, focus_depth_bin, eff_label):
    """
    Plot the heatmap of actual/expected hits for DOMs.
    """
    actual_hits = np.nan_to_num(actual_hits, nan=0.0)
    expected_hits = np.nan_to_num(expected_hits, nan=0.0)

    # Calculate miss ratios
    with np.errstate(divide='ignore', invalid='ignore'):
        miss_ratios = np.where(expected_hits > 0, actual_hits / expected_hits, 0)

    # Ensure DOM positions are unique
    unique_doms = dom_positions.drop_duplicates(subset=['dom_x', 'dom_y', 'dom_z']).reset_index()
    unique_x = unique_doms['dom_x'].to_numpy()
    unique_y = unique_doms['dom_y'].to_numpy()

    # Match miss_ratios length to unique DOM positions
    if len(miss_ratios) > len(unique_doms):
        miss_ratios = miss_ratios[: len(unique_doms)]

    # Add miss_ratios to unique DOM DataFrame
    unique_doms['miss_ratio'] = miss_ratios

    # Create a pivot table for the heatmap
    pivot_table = unique_doms.pivot_table(
        values='miss_ratio',
        index='dom_y',
        columns='dom_x',
        aggfunc='mean'
    )

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(
        pivot_table,
        origin='lower',
        aspect='auto',
        cmap='coolwarm',
        extent=[
            unique_doms['dom_x'].min(),
            unique_doms['dom_x'].max(),
            unique_doms['dom_y'].min(),
            unique_doms['dom_y'].max()
        ]
    )
    plt.colorbar(label='Miss Ratio (Σ(misses) / Σ(expected_hits))')
    plt.title(f"Miss Ratio Heatmap for {eff_label} DOMs at Depth [{focus_depth_bin}]")
    plt.xlabel('DOM x-coordinate')
    plt.ylabel('DOM y-coordinate')
    plt.savefig(f"miss_heatmap_{eff_label}_depth_{focus_depth_bin}.pdf", format='pdf')
    print(f"Saved heatmap: miss_heatmap_{eff_label}_depth_{focus_depth_bin}.pdf")





def plot_hit_fraction(
    nqe_positions, hqe_positions, nqe_expected_hits, nqe_actual_hits, hqe_expected_hits, hqe_actual_hits
):
    """
    Plot hit fraction scatter plot for both NQE and HQE DOMs with a shared colorbar.
    """
    # Calculate hit fractions
    nqe_actual_hits = np.nan_to_num(nqe_actual_hits, nan=0.0)
    nqe_expected_hits = np.nan_to_num(nqe_expected_hits, nan=0.0)
    hqe_actual_hits = np.nan_to_num(hqe_actual_hits, nan=0.0)
    hqe_expected_hits = np.nan_to_num(hqe_expected_hits, nan=0.0)

    # Calculate hit fractions
    with np.errstate(divide='ignore', invalid='ignore'):
        nqe_hit_fraction = np.where(nqe_expected_hits > 0, nqe_actual_hits / nqe_expected_hits, 0)
        hqe_hit_fraction = np.where(hqe_expected_hits > 0, hqe_actual_hits / hqe_expected_hits, 0)

    # Extract unique DOM positions for NQE
    nqe_unique_positions = nqe_positions.drop_duplicates(subset=['dom_x', 'dom_y']).reset_index()
    nqe_x = nqe_unique_positions['dom_x'].to_numpy()
    nqe_y = nqe_unique_positions['dom_y'].to_numpy()

    # Extract unique DOM positions for HQE
    hqe_unique_positions = hqe_positions.drop_duplicates(subset=['dom_x', 'dom_y']).reset_index()
    hqe_x = hqe_unique_positions['dom_x'].to_numpy()
    hqe_y = hqe_unique_positions['dom_y'].to_numpy()

    # Ensure hit fractions match the unique positions
    nqe_hit_fraction = nqe_hit_fraction[: len(nqe_x)]  # Match length of unique NQE DOMs
    hqe_hit_fraction = hqe_hit_fraction[: len(hqe_x)]  # Match length of unique HQE DOMs

    # Combine hit fractions for consistent color scaling
    combined_hit_fractions = np.concatenate((nqe_hit_fraction, hqe_hit_fraction))

    # Calculate mean hit fractions
    mean_fraction_nqe = np.mean(nqe_hit_fraction[nqe_expected_hits[: len(nqe_x)] > 0])
    mean_fraction_hqe = np.mean(hqe_hit_fraction[hqe_expected_hits[: len(hqe_x)] > 0])

    print(f"Mean hit-fraction NQE: {mean_fraction_nqe:.4f}")
    print(f"Mean hit-fraction HQE: {mean_fraction_hqe:.4f}")

    # Combine positions for scatter plot
    combined_x = np.concatenate((nqe_x, hqe_x))
    combined_y = np.concatenate((nqe_y, hqe_y))

    # Plot scatter plot for NQE and HQE hit fractions
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        combined_x,
        combined_y,
        c=combined_hit_fractions,
        cmap="viridis",
        s=40,
    )
    plt.colorbar(scatter, label="Hit Fraction (Actual / Expected)")

    plt.xlabel("DOM x-coordinate")
    plt.ylabel("DOM y-coordinate")
    plt.title("DOM Hit Fraction Scatter Plot (NQE & HQE)")
    plt.savefig("hit_fraction_scatter_plot_nqe_hqe_shared_colorbar.pdf", format="pdf")
    print("Saved scatter plot: hit_fraction_scatter_plot_nqe_hqe_shared_colorbar.pdf")






def main():
    file_path = '/groups/icecube/simon/GNN/workspace/filtered.db'
    con = sqlite3.connect(file_path)
    df_truth = pd.read_sql_query("SELECT * FROM truth", con)
    pulses_df = pd.read_sql_query("SELECT * FROM SplitInIcePulsesSRT", con)
    con.close()
    
    # Limit the number of events for testing
    df_truth = df_truth.iloc[:10000]
    filtered_events = df_truth['event_no'].unique()
    pulses_df = pulses_df[pulses_df['event_no'].isin(filtered_events)]


    # Filter DOMs by depth and strings
    all_depth_bins = np.arange(-500, 0, 10)
    pulses_df['depth_bin'] = pd.cut(pulses_df['dom_z'], bins=all_depth_bins, labels=all_depth_bins[:-1])
    pulses_df = pulses_df[pulses_df['string'] < 87.0]

    focus_depth_bin = -320
    filtered_doms = pulses_df[(pulses_df['depth_bin'] == focus_depth_bin)]
    
    # Split into HQE and NQE groups
    filtered_HQE = filtered_doms[filtered_doms['rde'] == 1.35]
    filtered_NQE = filtered_doms[filtered_doms['rde'] == 1.0]
    
    # Create position arrays
    pos_array_HQE = filtered_HQE[['dom_x', 'dom_y', 'dom_z']].drop_duplicates().to_numpy()
    pos_array_NQE = filtered_NQE[['dom_x', 'dom_y', 'dom_z']].drop_duplicates().to_numpy()

    # Process events in chunks
    distance_range = (5, 40)
    event_chunks = np.array_split(df_truth, 8)

    # Process HQE and NQE events separately
    results_HQE = [process_events(chunk, pulses_df, pos_array_HQE, distance_range) for chunk in event_chunks]
    results_NQE = [process_events(chunk, pulses_df, pos_array_NQE, distance_range) for chunk in event_chunks]

    # Aggregate results
    expected_hits_HQE = np.sum([result[0] for result in results_HQE], axis=0)
    actual_hits_HQE = np.sum([result[1] for result in results_HQE], axis=0)
    expected_hits_NQE = np.sum([result[0] for result in results_NQE], axis=0)
    actual_hits_NQE = np.sum([result[1] for result in results_NQE], axis=0)

    # Prepare DOM coordinates for plotting
    dom_x_HQE = filtered_HQE[['dom_x']].drop_duplicates().to_numpy().flatten()
    dom_y_HQE = filtered_HQE[['dom_y']].drop_duplicates().to_numpy().flatten()
    dom_x_NQE = filtered_NQE[['dom_x']].drop_duplicates().to_numpy().flatten()
    dom_y_NQE = filtered_NQE[['dom_y']].drop_duplicates().to_numpy().flatten()

    # Plot results
    plot_hit_fraction(
    filtered_NQE,
    filtered_HQE,
    expected_hits_NQE,
    actual_hits_NQE,
    expected_hits_HQE,
    actual_hits_HQE
    )
    plot_heatmap(filtered_HQE, expected_hits_HQE, actual_hits_HQE, focus_depth_bin, "HQE")
    plot_heatmap(filtered_NQE, expected_hits_NQE, actual_hits_NQE, focus_depth_bin, "NQE")



if __name__ == "__main__":
    main()
