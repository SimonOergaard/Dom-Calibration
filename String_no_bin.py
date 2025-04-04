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
    """Prints the total memory usage across all processes related to the script."""
    current_process = psutil.Process(os.getpid())
    all_processes = [current_process] + current_process.children(recursive=True)  # Get all worker processes

    total_memory = sum(proc.memory_info().rss for proc in all_processes if proc.is_running()) / (1024 * 1024)  # Convert to MB
    print(f"[Total Memory Usage] {message}: {total_memory:.2f} MB")

def create_shared_numpy_array(df):
    """Convert a DataFrame to a shared memory NumPy array, ensuring shape is correct."""
    np_array = df.to_numpy(dtype=np.float64, copy=True)  # Ensure numeric consistency
    print("DEBUG: Original df shape:", df.shape)
    print("DEBUG: Converted NumPy shape:", np_array.shape)

    shared_mem = shared_memory.SharedMemory(create=True, size=np_array.nbytes)
    shared_np_array = np.ndarray(np_array.shape, dtype=np_array.dtype, buffer=shared_mem.buf)
    shared_np_array[:] = np_array[:]  # Copy data into shared memory

    print("DEBUG: Shared memory created with shape:", shared_np_array.shape)
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

def process_events_with_charge_and_launches(event_chunk, pulses_df, pos_array, distance_range):
    """
    Optimized function to process events, counting expected hits, total charge, launches, 
    and time-centering pulses.
    """
    print_memory_usage("Before processing events")
    dom_position_map = {tuple(pos): idx for idx, pos in enumerate(pos_array)}
    #print_memory_usage("After creating DOM position map")
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
    #print_memory_usage("After initializing arrays")
    all_dom_distances = []
    all_dom_charges = []
    
    # # Preprocess pulses for quick lookup
    pulses_np = pulses_df.to_numpy()
    pulse_event_col = pulses_df.columns.get_loc("event_no")
    pulse_coords_cols = [pulses_df.columns.get_loc(c) for c in ["dom_x", "dom_y", "dom_z"]]
    pulse_charge_col = pulses_df.columns.get_loc("charge")
    pulse_time_col = pulses_df.columns.get_loc("corrected_dom_time")
    #print_memory_usage("After converting pulses to numpy array")
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
        #print_memory_usage("After calculating minimum distances")
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
    print_memory_usage("After processing events")
    return expected_hits, total_charge, launches, dom_event_map, dom_position_map, min_distances_events, dom_times, dom_charges, dom_distances, starting_points_events, np.array(all_dom_distances), np.array(all_dom_charges)

def process_chunk(event_chunk, pulses_df, pos_array, distance_range):
    
    return process_events_with_charge_and_launches(event_chunk, pulses_df, pos_array, distance_range)


def add_dynamic_jitter(df):
    """
    Add dynamic jitter to DOMs with overlapping x and y coordinates.
    """
    df = df.sort_values(by=['dom_x', 'dom_y']).reset_index(drop=True)
    df['x_jitter'] = df['dom_x']
    df['y_jitter'] = df['dom_y']

    seen_coords = {}
    for idx, row in df.iterrows():
        coord = (row['dom_x'], row['dom_y'])
        if coord in seen_coords:
            count = seen_coords[coord]
            df.at[idx, 'x_jitter'] += count * 10  # Add jitter in x
            df.at[idx, 'y_jitter'] += count * 10  # Add jitter in y
            seen_coords[coord] += 1
        else:
            seen_coords[coord] = 1

    return df

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
    

 
def plot_scatter_metrics(aggregated_metrics_79, aggregated_metrics_80, aggregated_metrics_84, output_dir):
    """
    Generate scatter plots for metrics (e.g., total charge, expected hits, or launches)
    over all DOM depths for strings 79, 80, 47, and 84 with RDE differentiation.
    """
    os.makedirs(output_dir, exist_ok=True)

    for metric_name in ["total_charge", "expected_hits", "launches"]:
        plt.figure(figsize=(12, 8))

        # Scatter plot for String 79
        for i, (z, value, rde) in enumerate(zip(
            aggregated_metrics_79["dom_z"],
            aggregated_metrics_79[metric_name],
            aggregated_metrics_79.get("rde", [1.0] * len(aggregated_metrics_79["dom_z"]))  # Default RDE to 1.0
        )):
            marker = 'o' if rde == 1.0 else '^'
            plt.scatter(z, value, label=f"String 79, RDE={rde:.2f}" if i == 0 else "", 
                        color="blue", alpha=0.7, s=50, marker=marker)

        # Scatter plot for String 80
        for i, (z, value, rde) in enumerate(zip(
            aggregated_metrics_80["dom_z"],
            aggregated_metrics_80[metric_name],
            aggregated_metrics_80.get("rde", [1.0] * len(aggregated_metrics_80["dom_z"]))  # Default RDE to 1.0
        )):
            marker = 'o' if rde == 1.0 else '^'
            plt.scatter(z, value, label=f"String 80, RDE={rde:.2f}" if i == 0 else "", 
                        color="green", alpha=0.7, s=50, marker=marker)

        # # Scatter plot for String 47
        # for i, (z, value, rde) in enumerate(zip(
        #     aggregated_metrics_47["dom_z"],
        #     aggregated_metrics_47[metric_name],
        #     aggregated_metrics_47.get("rde", [1.0] * len(aggregated_metrics_47["dom_z"]))  # Default RDE to 1.0
        # )):
        #     marker = 'o' if rde == 1.0 else '^'
        #     plt.scatter(z, value, label=f"String 47, RDE={rde:.2f}" if i == 0 else "", 
        #                 color="red", alpha=0.7, s=50, marker=marker)

        # Scatter plot for String 84
        for i, (z, value, rde) in enumerate(zip(
            aggregated_metrics_84["dom_z"],
            aggregated_metrics_84[metric_name],
            aggregated_metrics_84.get("rde", [1.0] * len(aggregated_metrics_84["dom_z"]))  # Default RDE to 1.0
        )):
            marker = 'o' if rde == 1.0 else '^'
            plt.scatter(z, value, label=f"String 84, RDE={rde:.2f}" if i == 0 else "", 
                        color="orange", alpha=0.7, s=50, marker=marker)
            
   

        # # Scatter plot for String 89
        # for i, (z, value, rde) in enumerate(zip(
        #     aggregated_metrics_89["dom_z"],
        #     aggregated_metrics_89[metric_name],
        #     aggregated_metrics_89.get("rde", [1.0] * len(aggregated_metrics_89["dom_z"]))  # Default RDE to 1.0
        # )):
        #     marker = 'o' if rde == 1.0 else '^'
        #     plt.scatter(z, value, label=f"String 89, RDE={rde:.2f}" if i == 0 else "", 
        #                 color="purple", alpha=0.7, s=50, marker=marker)
        # Plot labels and legends
        plt.xlabel("Depth (m)")
        plt.ylabel(metric_name.replace("_", " ").title())
        plt.title(f"{metric_name.replace('_', ' ').title()} vs Depth (Strings 79, 80, 84)")
        plt.legend()
        plt.grid(axis="both", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(output_dir, f"{metric_name.replace(' ', '_').lower()}_vs_depth_scatter_Check_10_110_.png"))
        plt.close()

def plot_histogram_monitored(aggregated_metrics_79, aggregated_metrics_80, aggregated_metrics_84, output_dir):
    """
    Generate histogram of total charge relative to the DOM on string 80 in the same layer.
    Only DOMs below z=0 with matching RDE values for nearest neighbors are considered.
    """
    def filter_valid_doms(metrics):
        mask = (np.array(metrics["dom_z"]) < 0)
        return {key: np.array(val)[mask] for key, val in metrics.items()}

    metrics_79 = filter_valid_doms(aggregated_metrics_79)
    metrics_80 = filter_valid_doms(aggregated_metrics_80)
    metrics_84 = filter_valid_doms(aggregated_metrics_84)


    def has_matching_neighbors(index, metrics, n_neighbors=2):
        """
        Check if the DOM at the given overall index (in metrics) has the same RDE value
        as its two closest neighbors on each side (up to 4 total) among DOMs on string 80.
        
        Parameters:
        index (int): The overall index in metrics for the DOM to check.
        metrics (dict): A dictionary containing at least 'string', 'dom_z', and 'rde' as keys.
        n_neighbors (int): Number of neighbors to check on each side (default is 2, totaling up to 4 neighbors).
        
        Returns:
        bool: True if all available neighbors have the same RDE value as the current DOM; otherwise False.
        """


        # Convert lists to numpy arrays for easy indexing.
        string_arr = np.array(metrics['string'])
        dom_z_arr = np.array(metrics['dom_z'])
        rde_arr = np.array(metrics['rde'])
        
        # Consider only DOMs on string 80.
        mask = string_arr == 80
        indices_on_80 = np.nonzero(mask)[0]
        if len(indices_on_80) == 0:
            return False  # No DOMs on string 80.
        
        # Sort these indices by dom_z.
        sorted_indices = indices_on_80[np.argsort(dom_z_arr[indices_on_80])]
        
        # Find the position of the current index in the sorted order.
        pos_arr = np.where(sorted_indices == index)[0]
        if pos_arr.size == 0:
            return False  # The provided index is not a DOM on string 80.
        pos = pos_arr[0]
        
        # Gather up to n_neighbors before and n_neighbors after (total up to 4 neighbors).
        neighbor_indices = []
        for i in range(1, n_neighbors + 1):
            if pos - i >= 0:
                neighbor_indices.append(sorted_indices[pos - i])
            if pos + i < len(sorted_indices):
                neighbor_indices.append(sorted_indices[pos + i])
        
        # Check if all gathered neighbors have the same RDE as the current DOM.
        current_rde = rde_arr[index]
        for neighbor_idx in neighbor_indices:
            if rde_arr[neighbor_idx] != current_rde:
                return False
        return True

    def find_monitor_charge(dom_z, metrics, rde_target):
        valid = np.where((metrics['rde'] == rde_target) & (metrics['string'] == 80))[0]
        if len(valid) == 0:
            return np.nan
        nearest_idx = valid[np.argmin(np.abs(metrics['dom_z'][valid] - dom_z))]
        return metrics['total_charge'][nearest_idx]

    def compute_ratios(metrics, monitor_metrics, rde_target):
        return [
            total_charge / find_monitor_charge(dom_z, monitor_metrics, rde_target)
            for idx, (dom_z, total_charge, rde) in enumerate(zip(metrics['dom_z'], metrics['total_charge'], metrics['rde']))
            if find_monitor_charge(dom_z, monitor_metrics, rde_target) > 0
            and rde == rde_target
            and has_matching_neighbors(idx, monitor_metrics)  # Note the plural
        ]

    # Loop over each RDE value you want to analyze.
    for rde_value in [1.0, 1.35]:
        
        if rde_value == 1.0:
            bin_count = 29
        elif rde_value == 1.35:
            bin_count = 9
        else:
            continue
        ratios_79 = []
        ratios_80 = []
        ratios_84 = []
        
 
        monitor_indices = [i for i, r in enumerate(metrics_80["rde"]) if r == rde_value]
        
        # Loop over the monitor DOMs (string 80) that match the current RDE.
        for idx in monitor_indices:
            # Get the DOM number and monitor charge.
            if not has_matching_neighbors(idx, metrics_80):
                continue  # Skip DOMs without matching neighbors.
            
            dom_number = metrics_80["dom_number"][idx]
            monitor_charge = metrics_80["total_charge"][idx]
            
            try:
                # We convert to list() to use the .index() method.
                idx_79 = list(metrics_79["dom_number"]).index(dom_number)
                idx_84 = list(metrics_84["dom_number"]).index(dom_number)
                
            except ValueError:
                # Skip if the matching DOM number is not found in one of the other strings.
                continue
            # Use the DOM number to locate the corresponding DOM in strings 79 and 84.
           
            
            charge_79 = metrics_79["total_charge"][idx_79]
            charge_84 = metrics_84["total_charge"][idx_84]
          
            
            # Compute ratios relative to the monitor (string 80) charge.
            if monitor_charge > 0:
                ratio_79 = charge_79 / monitor_charge
                ratio_80 = monitor_charge / monitor_charge  # This will be 1.
                ratio_84 = charge_84 / monitor_charge
          
            else:
                ratio_79 = ratio_80 = ratio_84 = 0  # or handle division by zero appropriately.
            
            ratios_79.append(ratio_79)
            ratios_80.append(ratio_80)
            ratios_84.append(ratio_84)
  
        #
        print(f"Mean total charge for string 80: {np.mean(metrics_80['total_charge'])}")
        print(f"Mean expected hits for string 80: {np.mean(metrics_80['expected_hits'])}")
        print(f"Mean launches for string 80: {np.mean(metrics_80['launches'])}")
        print(f"Mean total charge for string 79: {np.mean(metrics_79['total_charge'])}")
        print(f"Mean expected hits for string 79: {np.mean(metrics_79['expected_hits'])}")
        print(f"Mean launches for string 79: {np.mean(metrics_79['launches'])}")
        print(f"Mean total charge for string 84: {np.mean(metrics_84['total_charge'])}")
        print(f"Mean expected hits for string 84: {np.mean(metrics_84['expected_hits'])}")
        print(f"Mean launches for string 84: {np.mean(metrics_84['launches'])}")

        # Print the length of the ratios for each string.
        print(f"RDE={rde_value}: Ratios 79={len(ratios_79)}, Ratios 80={len(ratios_80)}, Ratios 84={len(ratios_84)}")
        
        # Now plot the histograms for this RDE value.
        plt.figure(figsize=(10, 6))
        plt.hist(ratios_79, bins=bin_count, alpha=0.5, label='String 79', color='blue')
        plt.hist(ratios_80, bins=bin_count, alpha=0.5, label='String 80', color='green')
        plt.hist(ratios_84, bins=bin_count, alpha=0.5, label='String 84', color='orange')
    
        plt.xlabel('Total Charge / Monitor DOM Total Charge')
        plt.ylabel('Frequency')
        plt.title(f'Charge Ratios Relative to String 80 DOM (RDE={rde_value})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{output_dir}/charge_ratio_below_rde_2Nearest_10_110_{rde_value}_no_SRT.png")
        plt.close()


def plot_histogram_monitored_expected(aggregated_metrics_79, aggregated_metrics_80, aggregated_metrics_84, output_dir):
    """
    Generate histogram of total charge relative to the DOM on string 80 in the same layer.
    Only DOMs below z=0 with matching RDE values for nearest neighbors are considered.
    """
    def filter_valid_doms(metrics):
        mask = (np.array(metrics["dom_z"]) < 0)
        return {key: np.array(val)[mask] for key, val in metrics.items()}

    metrics_79 = filter_valid_doms(aggregated_metrics_79)
    metrics_80 = filter_valid_doms(aggregated_metrics_80)
    metrics_84 = filter_valid_doms(aggregated_metrics_84)
    #metrics_85 = filter_valid_doms(aggregated_metrics_85)

    def has_matching_neighbors(index, metrics, n_neighbors=2):
        """
        Check if the DOM at the given overall index (in metrics) has the same RDE value
        as its two closest neighbors on each side (up to 4 total) among DOMs on string 80.
        
        Parameters:
        index (int): The overall index in metrics for the DOM to check.
        metrics (dict): A dictionary containing at least 'string', 'dom_z', and 'rde' as keys.
        n_neighbors (int): Number of neighbors to check on each side (default is 2, totaling up to 4 neighbors).
        
        Returns:
        bool: True if all available neighbors have the same RDE value as the current DOM; otherwise False.
        """


        # Convert lists to numpy arrays for easy indexing.
        string_arr = np.array(metrics['string'])
        dom_z_arr = np.array(metrics['dom_z'])
        rde_arr = np.array(metrics['rde'])
        
        # Consider only DOMs on string 80.
        mask = string_arr == 80
        indices_on_80 = np.nonzero(mask)[0]
        if len(indices_on_80) == 0:
            return False  # No DOMs on string 80.
        
        # Sort these indices by dom_z.
        sorted_indices = indices_on_80[np.argsort(dom_z_arr[indices_on_80])]
        
        # Find the position of the current index in the sorted order.
        pos_arr = np.where(sorted_indices == index)[0]
        if pos_arr.size == 0:
            return False  # The provided index is not a DOM on string 80.
        pos = pos_arr[0]
        
        # Gather up to n_neighbors before and n_neighbors after (total up to 4 neighbors).
        neighbor_indices = []
        for i in range(1, n_neighbors + 1):
            if pos - i >= 0:
                neighbor_indices.append(sorted_indices[pos - i])
            if pos + i < len(sorted_indices):
                neighbor_indices.append(sorted_indices[pos + i])
        
        # Check if all gathered neighbors have the same RDE as the current DOM.
        current_rde = rde_arr[index]
        for neighbor_idx in neighbor_indices:
            if rde_arr[neighbor_idx] != current_rde:
                return False
        return True

    # def find_monitor_charge(dom_z, metrics, rde_target):
    #     valid = np.where((metrics['rde'] == rde_target) & (metrics['string'] == 80))[0]
    #     if len(valid) == 0:
    #         return np.nan
    #     nearest_idx = valid[np.argmin(np.abs(metrics['dom_z'][valid] - dom_z))]
    #     return (metrics['total_charge']/metrics['expected_hits'])[nearest_idx]

    # def compute_ratios(metrics, monitor_metrics, rde_target):
    #     return [
    #         total_charge / expected_hits / find_monitor_charge(dom_z, monitor_metrics, rde_target)
    #         for idx, (dom_z, total_charge, expected_hits, rde) in enumerate(zip(metrics['dom_z'], metrics['total_charge'],metrics['expected_hits'], metrics['rde']))
    #         if find_monitor_charge(dom_z, monitor_metrics, rde_target) > 0
    #         and rde == rde_target
    #         and has_matching_neighbors(idx, monitor_metrics)  # Note the plural
    #     ]

    # Loop over each RDE value you want to analyze.
    for rde_value in [1.0, 1.35]:
        
        if rde_value == 1.0:
            bin_count = 29
        elif rde_value == 1.35:
            bin_count = 9
        else:
            continue
        ratios_79 = []
        ratios_80 = []
        ratios_84 = []
        #ratios_85 = []
 
        monitor_indices = [i for i, r in enumerate(metrics_80["rde"]) if r == rde_value]
        
        # Loop over the monitor DOMs (string 80) that match the current RDE.
        for idx in monitor_indices:
            # Get the DOM number and monitor charge.
            if not has_matching_neighbors(idx, metrics_80):
                continue  # Skip DOMs without matching neighbors.
            
            dom_number = metrics_80["dom_number"][idx]
            monitor_charge = metrics_80["total_charge"][idx]
            monitor_expected = metrics_80["expected_hits"][idx]
            monitor = monitor_charge/monitor_expected
            
            
            try:
                # We convert to list() to use the .index() method.
                idx_79 = list(metrics_79["dom_number"]).index(dom_number)
                idx_84 = list(metrics_84["dom_number"]).index(dom_number)
                #idx_85 = list(metrics_85["dom_number"]).index(dom_number)
            except ValueError:
                # Skip if the matching DOM number is not found in one of the other strings.
                continue
            # Use the DOM number to locate the corresponding DOM in strings 79 and 84.
           
            
            charge_79 = metrics_79["total_charge"][idx_79]
            expected_79 = metrics_79["expected_hits"][idx_79]
            charge_84 = metrics_84["total_charge"][idx_84]
            expected_84 = metrics_84["expected_hits"][idx_84]	
            #charge_85 = metrics_85["total_charge"][idx_85]
            #expected_85 = metrics_85["expected_hits"][idx_85]
            
            # Compute ratios relative to the monitor (string 80) charge.
            if monitor_charge > 0:
                ratio_79 = (charge_79 / expected_79) / monitor
                ratio_80 = (monitor_charge / monitor_expected) / monitor  # This will be 1.
                ratio_84 = (charge_84 / expected_84) / monitor
             #   ratio_85 = (charge_85 / expected_85) / monitor
            else:
                ratio_79 = ratio_80 = ratio_84 = 0  # or handle division by zero appropriately.
            
            ratios_79.append(ratio_79)
            ratios_80.append(ratio_80)
            ratios_84.append(ratio_84)
           # ratios_85.append(ratio_85)
        # Print the length of the ratios for each string.
        print(f"RDE={rde_value}: Ratios 79={len(ratios_79)}, Ratios 80={len(ratios_80)}, Ratios 84={len(ratios_84)}")
        
        # Now plot the histograms for this RDE value.
        plt.figure(figsize=(10, 6))
        plt.hist(ratios_79, bins=bin_count, alpha=0.5, label='String 79', color='blue')
        plt.hist(ratios_80, bins=bin_count, alpha=0.5, label='String 80', color='green')
        plt.hist(ratios_84, bins=bin_count, alpha=0.5, label='String 84', color='orange')
       # plt.hist(ratios_85, bins=bin_count, alpha=0.5, label='String 85', color='purple')
        plt.xlabel('Total Charge / expected_hits / Monitor DOM Total Charge')
        plt.ylabel('Frequency')
        plt.title(f'RIDE Relative to String 80 DOM (RDE={rde_value})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{output_dir}/RIDE_histogram_rde_2Nearest_10_110_{rde_value}_no_SRT.png")
        plt.close()


def plot_histogram_monitored_above_0(aggregated_metrics_79, aggregated_metrics_80, aggregated_metrics_84, output_dir):
    """
    Generate histogram of total charge relative to the DOM on string 80 in the same layer.
    Only DOMs below z=0 with matching RDE values for nearest neighbors are considered.
    """
    def filter_valid_doms(metrics):
        mask = (np.array(metrics["dom_z"]) > 0)
        return {key: np.array(val)[mask] for key, val in metrics.items()}

    metrics_79 = filter_valid_doms(aggregated_metrics_79)
    metrics_80 = filter_valid_doms(aggregated_metrics_80)
    metrics_84 = filter_valid_doms(aggregated_metrics_84)
 

    def has_matching_neighbors(index, metrics, n_neighbors=2):
        """
        Check if the DOM at the given overall index (in metrics) has the same RDE value
        as its two closest neighbors on each side (up to 4 total) among DOMs on string 80.
        
        Parameters:
        index (int): The overall index in metrics for the DOM to check.
        metrics (dict): A dictionary containing at least 'string', 'dom_z', and 'rde' as keys.
        n_neighbors (int): Number of neighbors to check on each side (default is 2, totaling up to 4 neighbors).
        
        Returns:
        bool: True if all available neighbors have the same RDE value as the current DOM; otherwise False.
        """


        # Convert lists to numpy arrays for easy indexing.
        string_arr = np.array(metrics['string'])
        dom_z_arr = np.array(metrics['dom_z'])
        rde_arr = np.array(metrics['rde'])
        
        # Consider only DOMs on string 80.
        mask = string_arr == 80
        indices_on_80 = np.nonzero(mask)[0]
        if len(indices_on_80) == 0:
            return False  # No DOMs on string 80.
        
        # Sort these indices by dom_z.
        sorted_indices = indices_on_80[np.argsort(dom_z_arr[indices_on_80])]
        
        # Find the position of the current index in the sorted order.
        pos_arr = np.where(sorted_indices == index)[0]
        if pos_arr.size == 0:
            return False  # The provided index is not a DOM on string 80.
        pos = pos_arr[0]
        
        # Gather up to n_neighbors before and n_neighbors after (total up to 4 neighbors).
        neighbor_indices = []
        for i in range(1, n_neighbors + 1):
            if pos - i >= 0:
                neighbor_indices.append(sorted_indices[pos - i])
            if pos + i < len(sorted_indices):
                neighbor_indices.append(sorted_indices[pos + i])
        
        # Check if all gathered neighbors have the same RDE as the current DOM.
        current_rde = rde_arr[index]
        for neighbor_idx in neighbor_indices:
            if rde_arr[neighbor_idx] != current_rde:
                return False
        return True

    def find_monitor_charge(dom_z, metrics, rde_target):
        valid = np.where((metrics['rde'] == rde_target) & (metrics['string'] == 80))[0]
        if len(valid) == 0:
            return np.nan
        nearest_idx = valid[np.argmin(np.abs(metrics['dom_z'][valid] - dom_z))]
        return metrics['total_charge'][nearest_idx]

    def compute_ratios(metrics, monitor_metrics, rde_target):
        return [
            total_charge / find_monitor_charge(dom_z, monitor_metrics, rde_target)
            for idx, (dom_z, total_charge, rde) in enumerate(zip(metrics['dom_z'], metrics['total_charge'], metrics['rde']))
            if find_monitor_charge(dom_z, monitor_metrics, rde_target) > 0
            and rde == rde_target
            and has_matching_neighbors(idx, monitor_metrics)  # Note the plural
        ]

    # Loop over each RDE value you want to analyze.
    for rde_value in [1.0, 1.35]:
        
        if rde_value == 1.0:
            bin_count = 29
        elif rde_value == 1.35:
            bin_count = 9
        else:
            continue
        ratios_79 = []
        ratios_80 = []
        ratios_84 = []
       
 
        monitor_indices = [i for i, r in enumerate(metrics_80["rde"]) if r == rde_value]
        
        # Loop over the monitor DOMs (string 80) that match the current RDE.
        for idx in monitor_indices:
            # Get the DOM number and monitor charge.
            if not has_matching_neighbors(idx, metrics_80):
                continue  # Skip DOMs without matching neighbors.
            
            dom_number = metrics_80["dom_number"][idx]
            monitor_charge = metrics_80["total_charge"][idx]
            
            try:
                # We convert to list() to use the .index() method.
                idx_79 = list(metrics_79["dom_number"]).index(dom_number)
                idx_84 = list(metrics_84["dom_number"]).index(dom_number)
             
            except ValueError:
                # Skip if the matching DOM number is not found in one of the other strings.
                continue
            # Use the DOM number to locate the corresponding DOM in strings 79 and 84.
           
            
            charge_79 = metrics_79["total_charge"][idx_79]
            charge_84 = metrics_84["total_charge"][idx_84]
         
            
            # Compute ratios relative to the monitor (string 80) charge.
            if monitor_charge > 0:
                ratio_79 = charge_79 / monitor_charge
                ratio_80 = monitor_charge / monitor_charge  # This will be 1.
                ratio_84 = charge_84 / monitor_charge
             
            else:
                ratio_79 = ratio_80 = ratio_84 = 0  # or handle division by zero appropriately.
            
            ratios_79.append(ratio_79)
            ratios_80.append(ratio_80)
            ratios_84.append(ratio_84)
       
        #

        # Print the length of the ratios for each string.
        print(f"RDE={rde_value}: Ratios 79={len(ratios_79)}, Ratios 80={len(ratios_80)}, Ratios 84={len(ratios_84)}")
        
        # Now plot the histograms for this RDE value.
        plt.figure(figsize=(10, 6))
        plt.hist(ratios_79, bins=bin_count, alpha=0.5, label='String 79', color='blue')
        plt.hist(ratios_80, bins=bin_count, alpha=0.5, label='String 80', color='green')
        plt.hist(ratios_84, bins=bin_count, alpha=0.5, label='String 84', color='orange')
        plt.xlabel('Total Charge / Monitor DOM Total Charge')
        plt.ylabel('Frequency')
        plt.title(f'Charge Ratios Relative to String 80 DOM above 0 (RDE={rde_value})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{output_dir}/charge_ratio_above_rde_2Nearest_10_110_{rde_value}.png")
        plt.close()
    

def plot_energy_distribution_from_truth(df_truth, output_dir):
    
    energy = df_truth['energy']
    plt.figure(figsize=(10, 6))
    plt.hist(energy, bins=100, alpha=0.5, color='blue', range=(0, 1000))
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    
    plt.title('Energy Distribution')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/energy_distribution.png")
    plt.close()
    
    
        

def plot_normalized_RDE_vs_dom_number_bar(aggregated_metrics_80, aggregated_metrics_83, 
                                          aggregated_metrics_84, aggregated_metrics_85, 
                                          aggregated_metrics_86, aggregated_metrics_81, output_dir):
    """
    Generate a grouped bar plot of normalized RDE values versus DOM number.
    Only consider monitor DOMs (from string 80) with RDE==1.0 and matching neighbors.
    For each such monitor DOM, locate the corresponding DOM (by dom_number)
    in the target strings (83, 84, 85, 86, and 81) and compute the normalized value:
    
        normalized_value = (target_total_charge / target_expected_hits) / (monitor_total_charge / monitor_expected_hits)
    
    The x-axis shows the DOM number, and each group has bars (color-coded by target string)
    showing the normalized value.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    def filter_valid_doms(metrics):
        # Only consider DOMs below z=0.
        mask = np.array(metrics["dom_z"]) < 0
        return {key: np.array(val)[mask] for key, val in metrics.items()}

    # Filter metrics for each string.
    metrics_80 = filter_valid_doms(aggregated_metrics_80)
    metrics_83 = filter_valid_doms(aggregated_metrics_83)
    metrics_84 = filter_valid_doms(aggregated_metrics_84)
    metrics_85 = filter_valid_doms(aggregated_metrics_85)
    metrics_86 = filter_valid_doms(aggregated_metrics_86)
    metrics_81 = filter_valid_doms(aggregated_metrics_81)

    def has_matching_neighbors(index, metrics, n_neighbors=2):
        """
        Check if the DOM (from string 80) at the given index has up to 4 neighbors 
        (2 before and 2 after when sorted by dom_z) with the same RDE value.
        """
        string_arr = np.array(metrics['string'])
        dom_z_arr = np.array(metrics['dom_z'])
        rde_arr = np.array(metrics['rde'])
        mask = string_arr == 80
        indices_on_80 = np.nonzero(mask)[0]
        if len(indices_on_80) == 0:
            return False
        sorted_indices = indices_on_80[np.argsort(dom_z_arr[indices_on_80])]
        pos_arr = np.where(sorted_indices == index)[0]
        if pos_arr.size == 0:
            return False
        pos = pos_arr[0]
        neighbor_indices = []
        for i in range(1, n_neighbors + 1):
            if pos - i >= 0:
                neighbor_indices.append(sorted_indices[pos - i])
            if pos + i < len(sorted_indices):
                neighbor_indices.append(sorted_indices[pos + i])
        current_rde = rde_arr[index]
        for neighbor_idx in neighbor_indices:
            if rde_arr[neighbor_idx] != current_rde:
                return False
        return True

    # Prepare dictionaries to store the normalized values and corresponding DOM numbers.
    results = {83: [], 84: [], 85: [], 86: [], 81: []}
    dom_numbers = {83: [], 84: [], 85: [], 86: [], 81: []}

    # Select monitor DOMs (from string 80) with RDE==1.0.
    monitor_indices = [i for i, r in enumerate(metrics_80["rde"]) if r == 1.0]
    for idx in monitor_indices:
        if not has_matching_neighbors(idx, metrics_80):
            continue
        dom_num = metrics_80["dom_number"][idx]
        monitor_charge = metrics_80["total_charge"][idx]
        monitor_expected = metrics_80["expected_hits"][idx]
        if monitor_expected <= 0:
            continue
        monitor_ratio = monitor_charge / monitor_expected

        # For each target string, try to find a matching DOM by dom_number.
        for string_id, met in zip([83, 84, 85, 86, 81],
                                  [metrics_83, metrics_84, metrics_85, metrics_86, metrics_81]):
            try:
                target_idx = list(met["dom_number"]).index(dom_num)
            except ValueError:
                continue
            target_charge = met["total_charge"][target_idx]
            target_expected = met["expected_hits"][target_idx]
            if target_expected <= 0:
                continue
            normalized_value = (target_charge / target_expected) / monitor_ratio
            results[string_id].append(normalized_value)
            dom_numbers[string_id].append(dom_num)

    # Build a union of all DOM numbers (from the monitor DOMs that have any match) and sort them.
    all_dom_numbers = set()
    for string_id in dom_numbers:
        all_dom_numbers.update(dom_numbers[string_id])
    all_dom_numbers = sorted(list(all_dom_numbers))
    
    # For easier lookup, build a mapping for each string: dom_number -> normalized value.
    mapping = {}
    for string_id in results:
        mapping[string_id] = {}
        for dom, val in zip(dom_numbers[string_id], results[string_id]):
            mapping[string_id][dom] = val

    # Create the grouped bar plot.
    n_groups = len(all_dom_numbers)
    indices = np.arange(n_groups)
    bar_width = 0.15
    # Define offsets so that the bars for different strings are grouped around the DOM number.
    offsets = {83: -2*bar_width, 84: -bar_width, 85: 0, 86: bar_width, 81: 2*bar_width}
    color_map = {83: 'blue', 84: 'green', 85: 'orange', 86: 'purple', 81: 'red'}

    plt.figure(figsize=(12, 6))
    for string_id in sorted(mapping.keys()):
        # For each DOM number group, get the normalized value if available, otherwise use 0.
        values = [mapping[string_id].get(dom, 0) for dom in all_dom_numbers]
        plt.bar(indices + offsets[string_id], values, width=bar_width, color=color_map[string_id],
                label=f'String {string_id}')
    plt.xlabel('DOM Number')
    plt.ylabel('Normalized RDE Value')
    plt.title('Normalized RDE Value vs DOM Number\n(Monitor: String 80 with RDE=1.0)')
    plt.xticks(indices, all_dom_numbers, rotation=90)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Normalized_RDE_barplot.png")
    plt.close()
    
    
def plot_track_distance_histogram(aggregated_metrics_80, aggregated_metrics_83, aggregated_metrics_84, 
                                  aggregated_metrics_85, aggregated_metrics_86, aggregated_metrics_81, 
                                  output_dir, bins=100):
    """
    Generate an overlapping histogram showing the distribution of DOM distances to the track.
    
    For each string (80, 83, 84, 85, 86, 81), the function extracts the 'distance' field from the 
    aggregated metrics (which represents the DOM's minimum distance to the track) and overlays the histograms.
    
    Parameters:
      aggregated_metrics_80, aggregated_metrics_83, aggregated_metrics_84, aggregated_metrics_85, 
      aggregated_metrics_86, aggregated_metrics_81:
          Dictionaries containing the DOM metrics for each string. Each dictionary should contain 
          at least: "dom_z" and "distance".
      output_dir: Directory path where the histogram image will be saved.
      bins: Number of bins for the histogram (default is 30).
    """
    def filter_valid_doms(metrics):
        # Only consider DOMs below z=0.
        mask = np.array(metrics["dom_z"]) < 0
        return {key: np.array(val)[mask] for key, val in metrics.items()}

    # Filter metrics for each string.
    metrics_80 = filter_valid_doms(aggregated_metrics_80)
    metrics_83 = filter_valid_doms(aggregated_metrics_83)
    metrics_84 = filter_valid_doms(aggregated_metrics_84)
    metrics_85 = filter_valid_doms(aggregated_metrics_85)
    metrics_86 = filter_valid_doms(aggregated_metrics_86)
    metrics_81 = filter_valid_doms(aggregated_metrics_81)

    # Prepare a color map for the different strings.
    color_map = {80: 'black', 83: 'blue', 84: 'green', 85: 'orange', 86: 'purple', 81: 'red'}
    # Create a mapping from string number to its filtered metrics.
    metrics_dict = {
        80: metrics_80,
        83: metrics_83,
        84: metrics_84,
        85: metrics_85,
        86: metrics_86,
        81: metrics_81
    }

    plt.figure(figsize=(10, 6))
    # Loop over each string and plot its histogram if distance data exists.
    for string_id, met in metrics_dict.items():
        # Ensure the "distance" field exists and contains data.
        if isinstance(met["distance"][0], (list, np.ndarray)):
            distances = np.hstack(met["distance"]).astype(np.float64)
        else:
            distances = np.array(met["distance"], dtype=np.float64)
        plt.hist(distances, bins=bins, alpha=0.5, label=f'String {string_id}', color=color_map[string_id])

    plt.xlabel('Distance from Track (m)')
    plt.ylabel('Frequency')
    plt.title('Distribution of DOM Distances to Track')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "DOM_track_distance_histogram.png"))
    plt.close()

global_df_truth = None
global_pulses_df = None

def init_worker(df_truth, pulses_df):
    """
    Worker initializer to store the large DataFrames as globals.
    On Unix systems using fork, these globals will be copy-on-write.
    """
    global global_df_truth, global_pulses_df
    global_df_truth = df_truth
    global_pulses_df = pulses_df

def process_zenith_bin(params):
    """
    Process one zenith bin defined by (min_zen, max_zen) using only events in that bin.
    For these events, compute the average RDE (total_charge/expected_hits) for string 79 (focus)
    and string 80 (monitor), but only using:
      - DOMs with dom_z < 0, and
      - Only those DOMs that are common between string 79 and string 80 (matched via dom_number).
    For string 80, only pulses with rde == 1.0 are used.
    
    Returns (min_zen, max_zen, normalized_rde) where normalized_rde = avg_RDE79/avg_RDE80.
    """
    import numpy as np
    # Unpack parameters
    min_zen, max_zen, distance_range = params

    global global_df_truth, global_pulses_df
    global_df_truth['zenith_degree'] = np.rad2deg(global_df_truth['zenith'])
    # Filter truth events for this zenith bin.
    subset_truth = global_df_truth[(global_df_truth['zenith_degree'] >= min_zen) &
                                   (global_df_truth['zenith_degree'] < max_zen)]
    if subset_truth.empty:
        return (min_zen, max_zen, np.nan)
    event_numbers = subset_truth['event_no'].unique()

    # --- Filter pulses for string 79 (focus) ---
    pulses_84 = global_pulses_df[
        (global_pulses_df['event_no'].isin(event_numbers)) &
        (global_pulses_df['string'] == 84) &
        (global_pulses_df['dom_z'] < 0)
    ]
    if pulses_84.empty:
        return (min_zen, max_zen, np.nan)
    # Get unique DOM info (including dom_number) for string 79.
    pos_info_84 = pulses_84[['dom_x','dom_y','dom_z','dom_number']].drop_duplicates()

    # --- Filter pulses for string 80 (monitor) ---
    pulses_80 = global_pulses_df[
        (global_pulses_df['event_no'].isin(event_numbers)) &
        (global_pulses_df['string'] == 80) &
        (global_pulses_df['dom_z'] < 0) &
        (global_pulses_df['rde'] == 1.0)
    ]
    if pulses_80.empty:
        return (min_zen, max_zen, np.nan)
    # Get unique DOM info for string 80.
    pos_info_80 = pulses_80[['dom_x','dom_y','dom_z','dom_number']].drop_duplicates()

    # --- Find common DOMs via dom_number ---
    common_dom = np.intersect1d(pos_info_84['dom_number'].values, pos_info_80['dom_number'].values)
    if common_dom.size == 0:
        return (min_zen, max_zen, np.nan)
    
    # Restrict both pos_info_79 and pos_info_80 to only common DOMs.
    pos_info_84 = pos_info_84[pos_info_84['dom_number'].isin(common_dom)].sort_values('dom_number')
    pos_info_80 = pos_info_80[pos_info_80['dom_number'].isin(common_dom)].sort_values('dom_number')
    
    # Use the filtered unique DOM positions.
    pos_array_84 = pos_info_84[['dom_x','dom_y','dom_z']].to_numpy()
    pos_array_80 = pos_info_80[['dom_x','dom_y','dom_z']].to_numpy()
    # Note: We assume that the ordering of DOMs in both strings is now the same 
    # because we sorted them by dom_number.

    # --- Process events for string 79 ---
    res84 = process_events_with_charge_and_launches(subset_truth, pulses_84[pulses_84['dom_number'].isin(common_dom)], pos_array_84, distance_range)
    expected_hits_84, total_charge_84 = res84[0], res84[1]
    rde_vals_84 = [tc / eh for tc, eh in zip(total_charge_84, expected_hits_84) if eh > 0]
    if len(rde_vals_84) == 0:
        return (min_zen, max_zen, np.nan)
    avg_rde_84= np.mean(rde_vals_84)

    # --- Process events for string 80 ---
    res80 = process_events_with_charge_and_launches(subset_truth, pulses_80[pulses_80['dom_number'].isin(common_dom)], pos_array_80, distance_range)
    expected_hits_80, total_charge_80 = res80[0], res80[1]
    rde_vals_80 = [tc / eh for tc, eh in zip(total_charge_80, expected_hits_80) if eh > 0]
    if len(rde_vals_80) == 0:
        return (min_zen, max_zen, np.nan)
    avg_rde_80 = np.mean(rde_vals_80)

    normalized_rde = avg_rde_84 / avg_rde_80 if avg_rde_80 != 0 else np.nan
    print(f"Zenith bin [{min_zen}°, {max_zen}°): {len(subset_truth)} events, "
          f"avg RDE 84 = {avg_rde_84:.3f}, avg RDE 80 = {avg_rde_80:.3f}, normalized = {normalized_rde:.3f}")
    return (min_zen, max_zen, normalized_rde)



def batched_process(func, params_list, batch_size=10, max_workers=4):
    """
    Process the params_list in batches using ProcessPoolExecutor.
    """
    from concurrent.futures import ProcessPoolExecutor
    results = []
    for i in range(0, len(params_list), batch_size):
        batch = params_list[i:i+batch_size]
        with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker,
                                  initargs=(global_df_truth, global_pulses_df)) as executor:
            batch_results = list(executor.map(func, batch))
        results.extend(batch_results)
    return results

def heatmap_normalized_RDE_focus84_monitor80(distance_range=(10, 110),
                                               zenith_bin_step=5, output_dir=None,
                                               batch_size=10, max_workers=8):
    """
    Compute and plot a heatmap of normalized RDE values.
    Uses the global DataFrames already loaded.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # Define zenith bin edges.
    min_zen_values = np.arange(0, 90, zenith_bin_step)
    max_zen_values = np.arange(zenith_bin_step, 91, zenith_bin_step)

    # Build list of parameters for each bin.
    bin_params = []
    for min_zen in min_zen_values:
        for max_zen in max_zen_values:
            if min_zen < max_zen:
                bin_params.append((min_zen, max_zen, distance_range))

    # Process bins in batches.
    results = batched_process(process_zenith_bin, bin_params, batch_size=batch_size, max_workers=max_workers)

    # Initialize heatmap matrix.
    heatmap = np.full((len(min_zen_values), len(max_zen_values)), np.nan)
    for min_zen, max_zen, norm_rde in results:
        i = np.where(min_zen_values == min_zen)[0][0]
        j = np.where(max_zen_values == max_zen)[0][0]
        heatmap[i, j] = norm_rde

    # Plot the heatmap.
    plt.figure(figsize=(10, 8))
    extent = [max_zen_values[0], max_zen_values[-1], min_zen_values[0], min_zen_values[-1]]
    im = plt.imshow(heatmap, origin='lower', extent=extent, aspect='auto', interpolation='nearest', vmin=1.3, vmax=1.55)
    plt.xlabel('Max Zenith (°)')
    plt.ylabel('Min Zenith (°)')
    plt.title('Heatmap of Normalized RDE (String 84 / String 80)\n'
              f'(Distance range: {distance_range[0]}-{distance_range[1]} m)')
    cbar = plt.colorbar(im)
    cbar.set_label('Normalized RDE')
    plt.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'Normalized_RDE_heatmap_focus84_monitor80.png'))
    plt.close()

     
def main():
    global global_df_truth, global_pulses_df
    #file_path = "/groups/icecube/simon/GNN/workspace/Scripts/filtered_all_no_cuts.db"
    #file_path = "/groups/icecube/simon/GNN/workspace/Scripts/filtered_all_big_data.db"
    file_path = '/groups/icecube/simon/GNN/workspace/data/Converted_I3_file/filtered_all_non_clean.db'
    #file_path = " /groups/icecube/petersen/GraphNetDatabaseRepository/dev_lvl3_genie_burnsample/dev_lvl3_genie_burnsample_v5.db"
    con = sqlite3.connect(file_path)

    df_truth = pd.read_sql_query("SELECT * FROM truth", con)
    filtered_events = df_truth['event_no'].unique()
    x = 1000000
    pulse_query = f"""
    SELECT dom_x, dom_y, dom_z, charge, dom_time, event_no, string, rde, dom_number
    FROM SplitInIcePulses
    WHERE event_no IN (
        SELECT DISTINCT event_no
        FROM SplitInIcePulses
        ORDER BY event_no
        LIMIT {x}
    )
    """
    pulses_df = pd.read_sql_query(pulse_query, con)
    
    con.close()
    
    pulses_df = pulses_df[pulses_df['event_no'].isin(filtered_events)]
    
    output_dir = "/groups/icecube/simon/GNN/workspace/Plots"
    os.makedirs(output_dir, exist_ok=True)
    
    pulses_df['corrected_dom_time'] = pulses_df['dom_time'] - pulses_df.groupby('event_no')['dom_time'].transform('min')
    distance_range = (10, 110)
    #focus_depth_bins = all_depth_bins[:-1]
    
    # Containers for results
    aggregated_metrics_79 = {"dom_z": [], "total_charge": [], "expected_hits": [], "launches": [], "rde": [], "string": [], "dom_number": [], "distance": []}
    aggregated_metrics_80 = {"dom_z": [], "total_charge": [], "expected_hits": [], "launches": [], "rde": [], "string": [], "dom_number": [], "distance": []}
    #aggregated_metrics_47 = {"dom_z": [], "total_charge": [], "expected_hits": [], "launches": [], "rde": []}
    aggregated_metrics_84 = {"dom_z": [], "total_charge": [], "expected_hits": [], "launches": [], "rde": [], "string": [], "dom_number": [], "distance": []}
    aggregated_metrics_85 = {"dom_z": [], "total_charge": [], "expected_hits": [], "launches": [], "rde": [], "string": [], "dom_number": [], "distance": []}
    aggregated_metrics_81 = {"dom_z": [], "total_charge": [], "expected_hits": [], "launches": [], "rde": [], "string": [], "dom_number": [], "distance": []}
    aggregated_metrics_82 = {"dom_z": [], "total_charge": [], "expected_hits": [], "launches": [], "rde": [], "string": [], "dom_number": [], "distance": []}
    aggregated_metrics_83 = {"dom_z": [], "total_charge": [], "expected_hits": [], "launches": [], "rde": [], "string": [], "dom_number": [], "distance": []}
    aggregated_metrics_86 = {"dom_z": [], "total_charge": [], "expected_hits": [], "launches": [], "rde": [], "string": [], "dom_number": [], "distance": []}
    print_memory_usage("After initializing metrics containers")

    # Iterate over strings
    for string, metrics in [(79, aggregated_metrics_79), (80, aggregated_metrics_80), (84, aggregated_metrics_84), (85, aggregated_metrics_85), (81, aggregated_metrics_81), (82, aggregated_metrics_82), (83, aggregated_metrics_83), (86, aggregated_metrics_86)]:
        print(f"Processing string {string}...")

        doms_filtered = pulses_df[pulses_df['string'] == string]
        if doms_filtered.empty:
            print(f"No DOMs found for string {string}.")
            continue

        # Convert filtered DOMs into numpy array
        pos_array_combined = doms_filtered[['dom_x', 'dom_y', 'dom_z']].drop_duplicates().to_numpy()

        # Filter `pulses_df` for relevant events
        relevant_event_ids = doms_filtered['event_no'].unique()
        filtered_pulses_df = pulses_df[pulses_df['event_no'].isin(relevant_event_ids) & (pulses_df['string'] == string)]
        
        #shared_mem, shared_pulses_np = create_shared_numpy_array(filtered_pulses_df)
        pulses_columns = filtered_pulses_df.columns
        event_chunks = np.array_split(df_truth, 12)
        pulses_chunks = [filtered_pulses_df[filtered_pulses_df["event_no"].isin(chunk["event_no"])] for chunk in event_chunks]
        cols_to_send = ["event_no", "dom_x", "dom_y", "dom_z", "charge","string", "corrected_dom_time","rde", "dom_number"]
        pulses_chunks = [chunk[cols_to_send].copy() for chunk in pulses_chunks]
        print_memory_usage(f"After filtering for string {string}")

        # Run processing in parallel
        # Step 3: Run in parallel, sending only relevant pulses to each worker
        
        try:
            with ProcessPoolExecutor(max_workers=12) as executor:
                combined_results = list(
                    executor.map(
                        process_chunk,
                        event_chunks,  # Unique event chunks
                        pulses_chunks,  # Pre-filtered pulses per chunk
                        [pos_array_combined] * 12,
                        [distance_range] * 12,
                    )
                )
        except Exception as e:
            print(f"ERROR in main process: {e}")
            import traceback
            traceback.print_exc()


        print_memory_usage("After processing events in parallel")
        
        # Aggregate results for each DOM
        for dom_idx, dom_z in enumerate(doms_filtered["dom_z"].unique()):
            dom_mask = doms_filtered["dom_z"] == dom_z
            
            rde_value = doms_filtered[dom_mask]["rde"].iloc[0]  # RDE for the current DOM
            dom_number = doms_filtered[dom_mask]["dom_number"].iloc[0]  # DOM number for the current DOM
            # Aggregate metrics for the current DOM
            dom_total_charge = np.sum([
                np.sum(result[1][dom_idx]) for result in combined_results 
                if dom_idx < len(result[1])  # Only access valid indices
            ])
            
            dom_expected_hits = np.sum([
                np.sum(result[0][dom_idx]) for result in combined_results 
                if dom_idx < len(result[0])  # Only access valid indices
            ])
            
            dom_launches = np.sum([
                np.sum(result[2][dom_idx]) for result in combined_results 
                if dom_idx < len(result[2])  # Only access valid indices
            ])
            
            dom_distances = np.hstack([np.atleast_1d(result[10]).astype(np.float64) for result in combined_results])


            
            # Store results for the current DOM
            metrics["dom_z"].append(dom_z) 
            metrics["rde"].append(rde_value)
            metrics["string"].append(string)
            metrics["total_charge"].append(dom_total_charge)
            metrics["expected_hits"].append(dom_expected_hits)
            metrics["launches"].append(dom_launches)
            metrics["dom_number"].append(dom_number)
            metrics["distance"].append(dom_distances)
            

    print_memory_usage("After aggregating metrics")
    # Generate scatter plots
    # plot_scatter_metrics(
    #     aggregated_metrics_79,
    #     aggregated_metrics_80,
    #     #aggregated_metrics_47,
    #     aggregated_metrics_84,
    #    # aggregated_metrics_85,
    #     output_dir
    # )
    plot_histogram_monitored(
        aggregated_metrics_79,
        aggregated_metrics_80,
        aggregated_metrics_84,
     #   aggregated_metrics_85,
        output_dir
    )
    plot_histogram_monitored_expected(
        aggregated_metrics_79,
        aggregated_metrics_80,
        aggregated_metrics_84,
     #   aggregated_metrics_85,
        output_dir
    )
    
    # plot_histogram_monitored_above_0(
    #     aggregated_metrics_79,
    #     aggregated_metrics_80,
    #     aggregated_metrics_84,
    #     output_dir
    # )
    #plot_energy_distribution_from_truth(df_truth, output_dir)
    # plot_normalized_RDE_vs_dom_number_bar(
    #     aggregated_metrics_80,
    #     aggregated_metrics_83,
    #     aggregated_metrics_84,
    #     aggregated_metrics_85,
    #     aggregated_metrics_86,
    #     aggregated_metrics_81,
    #     output_dir
    # )
    
    
    # plot_track_distance_histogram(
    #     aggregated_metrics_80,
    #     aggregated_metrics_83,
    #     aggregated_metrics_84,
    #     aggregated_metrics_85,
    #     aggregated_metrics_86,
    #     aggregated_metrics_81,
    #     output_dir
    # )
    
    # global_df_truth = df_truth.copy()
    # global_pulses_df = pulses_df.copy()

    # heatmap_normalized_RDE_focus84_monitor80(distance_range=(10, 110),
    #                                            zenith_bin_step=5,
    #                                            output_dir=output_dir,
    #                                            batch_size=10,
    #                                            max_workers=10)
    print_memory_usage("After plotting scatter metrics")
if __name__ == "__main__":
    main()