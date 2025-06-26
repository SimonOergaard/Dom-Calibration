import numpy as np
import pandas as pd
import os
import sys
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from concurrent.futures import ProcessPoolExecutor
import sqlite3
from tqdm import tqdm
from line_profiler import LineProfiler
from multiprocessing import shared_memory
from itertools import repeat
import psutil

def print_memory_usage(message=""):
    """Prints the current memory usage of the script."""
    process = psutil.Process(os.getpid())
    memory_used = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    print(f"[Memory Usage] {message}: {memory_used:.2f} MB")

def create_shared_numpy_array(df):
    """Convert a DataFrame to a shared memory NumPy array safely."""
    if isinstance(df, np.ndarray):
        np_array = df  # If already a NumPy array, use it
    else:
        np_array = df.to_numpy()  # Convert DataFrame to NumPy array

    shared_mem = shared_memory.SharedMemory(create=True, size=np_array.nbytes)
    shared_np_array = np.ndarray(np_array.shape, dtype=np_array.dtype, buffer=shared_mem.buf)
    shared_np_array[:] = np_array[:]  # Copy data to shared memory
    return shared_mem, shared_np_array

def preprocess_pulses(pulses_np, pulse_event_col):
    # Get unique event numbers and their indices
    unique_events, event_indices = np.unique(pulses_np[:, pulse_event_col], return_inverse=True)
    
    # Sort pulses by event number
    sorted_indices = np.argsort(event_indices)
    sorted_pulses = pulses_np[sorted_indices]
    sorted_event_indices = event_indices[sorted_indices]
    
    # Find the start and end indices for each event
    event_start_indices = np.searchsorted(sorted_event_indices, np.arange(len(unique_events)))
    event_end_indices = np.append(event_start_indices[1:], len(sorted_event_indices))
    
    # Create a dictionary of event pulses
    event_pulse_map = {unique_events[i]: sorted_pulses[event_start_indices[i]:event_end_indices[i]] for i in range(len(unique_events))}
    
    return event_pulse_map

def process_events_with_charge_and_launches(event_chunk, shared_pulses_np, pulses_columns, pos_array, distance_range):
    """
    Optimized function to process events, counting expected hits, total charge, launches, 
    and time-centering pulses.
    """
    # Create a mapping between DOM positions and their indices
    print_memory_usage("Before creating DOM position map")
    pulses_df = pd.DataFrame(shared_pulses_np, columns=pulses_columns)
    
    pulses_np = pulses_df.to_numpy()
    pulse_event_col = pulses_df.columns.get_loc("event_no")
    print_memory_usage("After converting pulses to numpy array")
    dom_position_map = {tuple(pos): idx for idx, pos in enumerate(pos_array)}
    print_memory_usage("After creating DOM position map")
    # Initialize arrays to store results
    expected_hits = np.zeros(len(pos_array))
    total_charge = np.zeros(len(pos_array))  
    launches = np.zeros(len(pos_array), dtype=int)  
    dom_event_map = []  
    min_distances_events = []  
    starting_points_events = []
    dom_times = [[] for _ in range(len(pos_array))]
    dom_charges = [[] for _ in range(len(pos_array))]
    dom_distances = [[] for _ in range(len(pos_array))]
    print_memory_usage("After initializing arrays")
    all_dom_distances = []
    all_dom_charges = []
    
    # # Preprocess pulses for quick lookup
    pulses_np = pulses_df
    pulse_event_col = pulses_df.columns.get_loc("event_no")
    pulse_coords_cols = [pulses_df.columns.get_loc(c) for c in ["dom_x", "dom_y", "dom_z"]]
    pulse_charge_col = pulses_df.columns.get_loc("charge")
    pulse_time_col = pulses_df.columns.get_loc("corrected_dom_time")
    print_memory_usage("After converting pulses to numpy array")
    event_pulse_map = preprocess_pulses(pulses_np, pulse_event_col)
    for _, event in tqdm(event_chunk.iterrows(), total=len(event_chunk), desc="Processing Events"):
       # true_x, true_y, true_z = event['position_x'], event['position_y'], event['position_z']
        # true_x, true_y, true_z = event.position_x, event.position_y, event.position_z
        # true_zenith, true_azimuth = event.zenith, event.azimuth
        # track_length = event.track_length
        event_no = event.event_no
        
        # Calculate minimum distances to the track
        min_distances, starting_points = calculate_minimum_distance_to_track(
            pos_array, event.position_x, event.position_y, event.position_z, event.zenith, event.azimuth, event.track_length
        )
        print_memory_usage("After calculating minimum distances")
        min_distances_events.append(min_distances)
        starting_points_events.append(starting_points)
        # Identify DOMs within the distance range
        doms_in_range_mask = (distance_range[0] <= min_distances) & (min_distances < distance_range[1])
        dom_indices = np.where(doms_in_range_mask)[0]

        # Increment expected hits for DOMs in range
        expected_hits[dom_indices] += 1
        dom_event_map.extend([(dom_idx, event_no) for dom_idx in dom_indices])

        # Filter event pulses
    #    event_pulses = pulses_np[pulses_np[:, pulse_event_col] == event_no]
    #    unique_event_pulses = np.unique(event_pulses[:, pulse_coords_cols], axis=0)
        if event_no in event_pulse_map:
            event_pulses = event_pulse_map[event_no]
        else:
            #print(f"Event {event_no} not found in the event pulse map.")
            continue

        # Get unique event pulses
        unique_event_pulses = np.unique(event_pulses[:, pulse_coords_cols], axis=0)

        # Track DOM launches, aggregate total charge, and center times
        unique_positions = set(map(tuple, unique_event_pulses))
        for dom_idx in dom_indices:
            dom_position = tuple(pos_array[dom_idx])
            # Check if the DOM is in the unique pulses for the event
            #if dom_position in [tuple(pos) for pos in unique_event_pulses]:
            if dom_position in unique_positions:
                launches[dom_idx] += 1  # Increment by 1 for unique DOM launch
            # Filter pulses belonging to this DOM
            matched_pulses = event_pulses[
                (event_pulses[:, pulse_coords_cols[0]] == dom_position[0]) &
                (event_pulses[:, pulse_coords_cols[1]] == dom_position[1]) &
                (event_pulses[:, pulse_coords_cols[2]] == dom_position[2])
            ]

            if len(matched_pulses) > 0:
                # Time centering
                #corrected_times = matched_pulses[:, pulse_time_col] - earliest_time
                dom_times[dom_idx] = np.append(dom_times[dom_idx], matched_pulses[:, pulse_time_col])
                dom_charges[dom_idx] = np.append(dom_charges[dom_idx], matched_pulses[:, pulse_charge_col])
                dom_distances[dom_idx] = np.append(dom_distances[dom_idx], np.full(len(matched_pulses), min_distances[dom_idx]))
                total_charge[dom_idx] += np.sum(matched_pulses[:, pulse_charge_col])
                
                all_dom_distances.append(min_distances[dom_idx])
                all_dom_charges.append(total_charge[dom_idx])
        dom_times = [np.array(times) for times in dom_times]
        dom_charges = [np.array(charges) for charges in dom_charges]
        dom_distances = [np.array(distances) for distances in dom_distances]

    return expected_hits, total_charge, launches, dom_event_map, dom_position_map, min_distances_events, dom_times, dom_charges, dom_distances, starting_points_events, np.array(all_dom_distances), np.array(all_dom_charges)


def process_chunk(chunk, pulses_df, pulses_columns, pos_array, distance_range):
    return process_events_with_charge_and_launches(chunk, pulses_df, pulses_columns, pos_array, distance_range)

def jitter_coordinates(dom_z_values, bin_size, jitter_amount=0.1):
    """
    Add jitter to DOM z-coordinates to prevent overlapping points in plots.
    
    Args:
        dom_z_values (list): List of DOM z-coordinates.
        bin_size (float): Size of each depth bin.
        jitter_amount (float): Maximum jitter applied, as a fraction of bin_size.
        
    Returns:
        list: Jittered z-coordinates.
    """
    jittered_values = [
        z + (np.random.uniform(-1, 1) * bin_size * jitter_amount) for z in dom_z_values
    ]
    return jittered_values


def calculate_minimum_distance_to_track(pos_array, true_x, true_y, true_z, true_zenith, true_azimuth, track_length):
    """
    Vectorized calculation of minimum distance from DOMs to the muon track.
    """
   # Define the start point and direction vector
    stop_point = np.array([true_x, true_y, true_z])
    direction_vector = np.array([
        np.sin(true_zenith) * np.cos(true_azimuth),
        np.sin(true_zenith) * np.sin(true_azimuth),
        np.cos(true_zenith)
    ])
    
    start_point = stop_point + direction_vector * track_length

    #Vectorize by expanding DOM positions
    dom_vectors = pos_array - stop_point  # Vector from start to DOM positions

    #Project DOM vectors onto the track's direction vector
    projection_lengths = np.dot(dom_vectors, direction_vector)

    #Clamp the projection lengths to [0, track_length] for bounds checking
    clamped_projections = np.clip(projection_lengths, 0, track_length)

    #ompute the closest points on the track
    closest_points = stop_point + clamped_projections[:, np.newaxis] * direction_vector

    #Compute the Euclidean distances from DOMs to the closest points
    min_distances = np.linalg.norm(pos_array - closest_points, axis=1)

    return min_distances, start_point
    
def create_and_assign_depth_bins(pulses_df, bin_size, min_depth=None, max_depth=None):
    """
    Automatically create depth bins and assign `dom_z` values to them.

    Parameters:
        pulses_df (pd.DataFrame): DataFrame containing the `dom_z` column.
        bin_size (float): Size of each depth bin (e.g., 7 meters).
        min_depth (float): Minimum depth value for binning. If None, uses the minimum `dom_z`.
        max_depth (float): Maximum depth value for binning. If None, uses the maximum `dom_z`.

    Returns:
        pd.DataFrame: Updated DataFrame with an assigned `depth_bin` column.
        list: List of tuples representing bin ranges (e.g., [(-500, -493), (-493, -486), ...]).
    """
    # Use the provided range or calculate it from the data
    if min_depth is None:
        min_depth = pulses_df['dom_z'].min()
    if max_depth is None:
        max_depth = pulses_df['dom_z'].max()
    
    # Generate bins as a list of tuples
    depth_bins = [(start, start + bin_size) for start in range(int(min_depth), int(max_depth), bin_size)]
    
    # Assign depth bins to `dom_z`
    def assign_bin(dom_z):
        for i, (bin_start, bin_end) in enumerate(depth_bins):
            if bin_start <= dom_z < bin_end:
                return i
        return -1  # Return -1 if no bin matches
    
    pulses_df['depth_bin'] = pulses_df['dom_z'].apply(assign_bin)
    
    return pulses_df, depth_bins

 
import matplotlib.pyplot as plt
import os

def plot_scatter_metrics_over_depth_with_rde(
    metrics_79, metrics_80, metrics_84, output_dir
):
    """
    Generate scatter plots for metrics over depth with marker styles based on RDE.

    Args:
        metrics_79 (dict): Metrics for string 79.
        metrics_80 (dict): Metrics for string 80.
        metrics_47 (dict): Metrics for string 47.
        metrics_84 (dict): Metrics for string 84.
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    for metric_name in ["total_charge", "expected_hits", "launches"]:
        plt.figure(figsize=(12, 8))

        # Define colors for each string
        string_colors = {
            "String 79": "blue",
            "String 80": "green",
         #   "String 47": "red",
            "String 84": "purple",
        }

        # Plot data for each string
        for string_label, metrics, color in [
            ("String 79", metrics_79, "blue"),
            ("String 80", metrics_80, "green"),
           # ("String 47", metrics_47, "red"),
            ("String 84", metrics_84, "purple"),
        ]:
            # Scatter for RDE = 1.0 (circles)
            plt.scatter(
                [z for z, rde in zip(metrics["dom_z"], metrics["rde"]) if rde == 1.0],
                [v for v, rde in zip(metrics[metric_name], metrics["rde"]) if rde == 1.0],
                color=color, alpha=0.7, s=50, marker="o", label=string_label if metric_name == "total_charge" else None
            )

            # Scatter for RDE = 1.35 (triangles)
            plt.scatter(
                [z for z, rde in zip(metrics["dom_z"], metrics["rde"]) if rde == 1.35],
                [v for v, rde in zip(metrics[metric_name], metrics["rde"]) if rde == 1.35],
                color=color, alpha=0.7, s=50, marker="^"
            )

        # Add a separate legend for RDE markers
        plt.scatter([], [], color="black", marker="o", label="RDE = 1.0")
        plt.scatter([], [], color="black", marker="^", label="RDE = 1.35")

        # Finalize plot
        plt.xlabel("Depth (m)")
        plt.ylabel(metric_name.replace("_", " ").capitalize())
        plt.title(f"{metric_name.replace('_', ' ').capitalize()} vs Depth", fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        plt.grid(axis="both", linestyle="--", alpha=0.7)
        plt.legend(loc="upper right")  # Single legend per string & RDE
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(output_dir, f"{metric_name}_vs_depth_with_rde.png"))
        plt.close()




def main():
    file_path = "/groups/icecube/simon/GNN/workspace/Scripts/filtered_all_big_data.db"
    con = sqlite3.connect(file_path)
    cursor = con.cursor()
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_no ON SplitInIcePulsesSRT(event_no);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_no_truth ON truth(event_no);")
    
    con.commit()
    df_truth = pd.read_sql_query("SELECT * FROM truth", con)
    filtered_events = df_truth['event_no'].unique()
    
    pulses_df = pd.read_sql_query(
        "SELECT dom_x, dom_y, dom_z, charge, dom_time, event_no, string, rde, dom_number FROM SplitInIcePulsesSRT", 
        con
    )
    con.close()
    
    pulses_df = pulses_df[pulses_df['event_no'].isin(filtered_events)]
    
    output_dir = "/groups/icecube/simon/GNN/workspace/Plots"
    os.makedirs(output_dir, exist_ok=True)
    
    
    bin_size = 7
    min_depth = -500
    max_depth = 18

    # Automatically create depth bins and assign them
    pulses_df, depth_bins = create_and_assign_depth_bins(pulses_df, bin_size, min_depth, max_depth)

    # Update focus_depth_bins based on the created bins
    focus_depth_bins = list(range(len(depth_bins)))
    
    #all_depth_bins = np.arange(-500, 18, 7)
    #pulses_df['depth_bin'] = pd.cut(pulses_df['dom_z'], bins=all_depth_bins, labels=all_depth_bins[:-1])
    pulses_df['corrected_dom_time'] = pulses_df['dom_time'] - pulses_df.groupby('event_no')['dom_time'].transform('min')
    
    distance_range = (10, 110)
    #focus_depth_bins = all_depth_bins[:-1]
    
    # Containers for results for strings 79, 80, 47, and 27
    aggregated_metrics_79 = {"dom_z": [], "total_charge": [], "expected_hits": [], "launches": [], "rde": []}
    aggregated_metrics_80 = {"dom_z": [], "total_charge": [], "expected_hits": [], "launches": [], "rde": []}
    aggregated_metrics_47 = {"dom_z": [], "total_charge": [], "expected_hits": [], "launches": [], "rde": []}
    aggregated_metrics_84 = {"dom_z": [], "total_charge": [], "expected_hits": [], "launches": [], "rde": []}

    print_memory_usage("After initializing metrics containers")

    for i, depth_bin_index in enumerate(focus_depth_bins):
        print(f"Processing focus_depth_bin: {depth_bin_index}")

        # Filter DOMs for the current depth bin
        doms_in_range = pulses_df[pulses_df['depth_bin'] == depth_bin_index]

        for string, metrics in [
            (79, aggregated_metrics_79),
            (80, aggregated_metrics_80),
            (47, aggregated_metrics_47),
            (84, aggregated_metrics_84),
        ]:
            doms_filtered = doms_in_range[doms_in_range['string'] == string]

            if doms_filtered.empty:
                print(f"No DOMs found for string {string} in depth bin {depth_bin_index}")
                continue

            pos_array_combined = doms_filtered[['dom_x', 'dom_y', 'dom_z']].drop_duplicates().to_numpy()

            # Extract `rde` values for this string and depth bin
            rde_values = doms_filtered['rde'].unique().tolist()

            # Filter pulses for relevant events
            relevant_event_ids = doms_filtered['event_no'].unique()
            filtered_pulses_df = pulses_df[
                pulses_df['event_no'].isin(relevant_event_ids) & (pulses_df['string'] == string)
            ]
            # **Ensure `filtered_pulses_df` is a DataFrame before converting**
            
            shared_mem, shared_pulses_np = create_shared_numpy_array(filtered_pulses_df)
            pulses_columns = filtered_pulses_df.columns  # Store column labels
            # Split events into chunks
            event_chunks = np.array_split(df_truth, 16)
            print_memory_usage(f"After filtering for string {string} and depth bin {depth_bin_index}")

            # Run processing in parallel
            try:
                with ProcessPoolExecutor(max_workers=16) as executor:
                    combined_results = list(
                        executor.map(
                            process_chunk,
                            event_chunks,                    # Unique event chunks
                            repeat(shared_pulses_np),        # Shared memory NumPy array
                            repeat(pulses_columns),          # Column names for indexing
                            repeat(pos_array_combined),
                            repeat(distance_range)
                        )
                    )
            finally:
                # **Ensure shared memory cleanup**
                shared_mem.close()
                shared_mem.unlink()

            print_memory_usage("After processing events in parallel")

            # Aggregate results from all chunks
            total_charge = np.sum([np.sum(result[1]) for result in combined_results])
            expected_hits = np.sum([np.sum(result[0]) for result in combined_results])
            launches = np.sum([np.sum(result[2]) for result in combined_results])
            # Update aggregated results
            metrics["total_charge"].append(total_charge)
            metrics["expected_hits"].append(expected_hits)
            metrics["launches"].append(launches)
            bin_center = (depth_bins[depth_bin_index][0] + depth_bins[depth_bin_index][1]) / 2  # Take the middle of the bin
            metrics["dom_z"].append(bin_center)  # Assign depth bin center
            metrics["rde"].extend(rde_values)  

    print_memory_usage("After aggregating results")

    # Apply jitter to z-coordinates for each metric
    aggregated_metrics_79["dom_z"] = jitter_coordinates(aggregated_metrics_79["dom_z"], bin_size)
    aggregated_metrics_80["dom_z"] = jitter_coordinates(aggregated_metrics_80["dom_z"], bin_size)
    #aggregated_metrics_47["dom_z"] = jitter_coordinates(aggregated_metrics_47["dom_z"], bin_size)
    aggregated_metrics_84["dom_z"] = jitter_coordinates(aggregated_metrics_84["dom_z"], bin_size)

    # Generate scatter plots
    plot_scatter_metrics_over_depth_with_rde(
        aggregated_metrics_79,
        aggregated_metrics_80,
        #aggregated_metrics_47,
        aggregated_metrics_84,
        output_dir,
    )

if __name__ == "__main__":
    main()