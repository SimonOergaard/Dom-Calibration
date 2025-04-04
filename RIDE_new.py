import numpy as np
import pandas as pd
import os
import sys
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from shapely.geometry import Polygon, Point

from matplotlib.patches import Polygon as mplPolygon
from concurrent.futures import ProcessPoolExecutor
import sqlite3
from tqdm import tqdm
from iminuit import Minuit
from line_profiler import LineProfiler

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

# -------------------------------
# Utility Functions
# -------------------------------


def process_events_with_charge_and_launches(event_chunk, pulses_df, pos_array, distance_range):
    """
    Optimized function to process events, counting expected hits, total charge, launches, 
    and time-centering pulses.
    """
    # Create a mapping between DOM positions and their indices
    dom_position_map = {tuple(pos): idx for idx, pos in enumerate(pos_array)}

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
    
    all_dom_distances = []
    all_dom_charges = []
    
    # Preprocess pulses for quick lookup
    pulses_np = pulses_df.to_numpy()
    pulse_event_col = pulses_df.columns.get_loc("event_no")
    pulse_coords_cols = [pulses_df.columns.get_loc(c) for c in ["dom_x", "dom_y", "dom_z"]]
    pulse_charge_col = pulses_df.columns.get_loc("charge")
    pulse_time_col = pulses_df.columns.get_loc("corrected_dom_time")
    
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
    dom_vectors = pos_array - stop_point  # Vector from start to DOM posit ions

    #Project DOM vectors onto the track's direction vector
    projection_lengths = np.dot(dom_vectors, direction_vector)

    #Clamp the projection lengths to [0, track_length] for bounds checking
    clamped_projections = np.clip(projection_lengths, 0, track_length)

    #ompute the closest points on the track
    closest_points = stop_point + clamped_projections[:, np.newaxis] * direction_vector

    #Compute the Euclidean distances from DOMs to the closest points
    min_distances = np.linalg.norm(pos_array - closest_points, axis=1)

    return min_distances, start_point

# def calculate_minimum_distance_to_track(pos_array, true_x, true_y, true_z, true_zenith, true_azimuth, track_length):
#     """
#     Calculate the minimum distance from DOMs to the particle track in 3D, considering cases
#     where the particle stops before the perpendicular projection point.

#     Parameters:
#         pos_array (numpy.ndarray): Array of DOM positions, shape (N, 3).
#         true_x (float): X-coordinate of the stopping point of the track.
#         true_y (float): Y-coordinate of the stopping point of the track.
#         true_z (float): Z-coordinate of the stopping point of the track.
#         true_zenith (float): Zenith angle of the particle track.
#         true_azimuth (float): Azimuth angle of the particle track.
#         track_length (float): Length of the particle track.

#     Returns:
#         min_distances (numpy.ndarray): Minimum distances from DOMs to the track, shape (N,).
#         start_point (numpy.ndarray): Starting point of the track, shape (3,).
#     """
#     # Define the stop point and direction vector
#     stop_point = np.array([true_x, true_y, true_z])
#     direction_vector = np.array([
#         np.sin(true_zenith) * np.cos(true_azimuth),
#         np.sin(true_zenith) * np.sin(true_azimuth),
#         np.cos(true_zenith)
#     ])
#     direction_vector /= np.linalg.norm(direction_vector)  # Normalize the direction vector
#     # Compute the start point using the track length
#     start_point = stop_point + direction_vector * track_length

#     # Normalize the direction vector
#     norm_direction_vec = np.dot(direction_vector, direction_vector)

#     # Vector from stop point to DOM positions
#     d = pos_array - stop_point

#     # Projection of `d` onto the direction vector
#     projection = np.dot(d, direction_vector)[:, np.newaxis] * direction_vector / norm_direction_vec

#     # Calculate scalar factors
#     dom_proj_scalar = np.dot(d, direction_vector) / norm_direction_vec
#     stop_point_scalar = np.dot(stop_point, direction_vector) / norm_direction_vec

#     # Compute distances for each DOM
#     stop_point_mask = stop_point_scalar >= dom_proj_scalar
#     min_distances = np.zeros(len(pos_array))
#     # Closest point is the stopping point
#     min_distances[stop_point_mask] = np.linalg.norm(pos_array[stop_point_mask] - stop_point, axis=1)

#     # Closest point is the perpendicular projection
#     min_distances[~stop_point_mask] = np.linalg.norm(d[~stop_point_mask] - projection[~stop_point_mask], axis=1)

#     return min_distances, start_point

def calculate_monitor_inside_polygon(df_positions, polygon_coords):
    """
    Calculate the monitor value (median of charge fraction) for DOMs inside the polygon,
    using DataFrame-based indices.
    """
    polygon = Polygon(polygon_coords)

    # Use DataFrame to calculate which DOMs are inside the polygon
    inside_indices = df_positions.apply(
        lambda row: polygon.contains(Point(row['dom_x'], row['dom_y'])),
        axis=1
    )
    
    # Filter the DataFrame to only include rows inside the polygon
    inside_positions = df_positions[inside_indices]

    # Calculate charge fractions (avoid division by zero)
    charge_fractions = inside_positions['total_charge'] / np.maximum(inside_positions['expected_hits'], 1)

    # Return the median of charge fractions
    return np.median(charge_fractions)
    
    
def linear_model(x, m, c):
    """
    Linear model for Minuit fitting.
    """
    return m * x + c



def plot_charge_fraction_and_launches(pos_array_NQE, pos_array_HQE, nqe_expected_hits, nqe_total_charge, nqe_launches, hqe_expected_hits, hqe_total_charge, hqe_launches, polygon_coords, polygon_coords_deepcore, pdf):
    """
    Plot charge fraction scatter plot, launches vs total charge scatter plot, and fits using Minuit for NQE and HQE DOMs.
    """
    
    from iminuit import Minuit

    def linear_model(x, m, c):
        """Linear model for fitting."""
        return m * x + c

    nqe_positions = pd.DataFrame(pos_array_NQE, columns=['dom_x', 'dom_y', 'dom_z','string','dom_number'])
    hqe_positions = pd.DataFrame(pos_array_HQE, columns=['dom_x', 'dom_y', 'dom_z','string','dom_number'])



    # Ensure charge fractions align with DOM positions

    nqe_positions["total_charge"] = nqe_total_charge
    nqe_positions["expected_hits"] = nqe_expected_hits
    nqe_positions["launches"] = nqe_launches

    
    hqe_positions["total_charge"] = hqe_total_charge
    hqe_positions["expected_hits"] = hqe_expected_hits
    hqe_positions["launches"] = hqe_launches
    
    # Calculate charge fractions
    nqe_positions["charge_fraction"] = nqe_positions["total_charge"] / nqe_positions["expected_hits"]
    nqe_positions.loc[nqe_positions["expected_hits"] == 0, "charge_fraction"] = 0

    hqe_positions["charge_fraction"] = hqe_positions["total_charge"] / hqe_positions["expected_hits"]
    hqe_positions.loc[hqe_positions["expected_hits"] == 0, "charge_fraction"] = 0

    nqe_positions_string = nqe_positions.loc[nqe_positions['string'] == 79]
    nqe_positions_charge_fraction_79 = np.median(nqe_positions_string['charge_fraction'])
    
    charge_fraction_min = min(nqe_positions['charge_fraction'].min(), hqe_positions['charge_fraction'].min())
    charge_fraction_max = max(nqe_positions['charge_fraction'].max(), hqe_positions['charge_fraction'].max())
    total_charge_min = min(nqe_positions['total_charge'].min(), hqe_positions['total_charge'].min())
    total_charge_max = max(nqe_positions['total_charge'].max(), hqe_positions['total_charge'].max())
    launches_min = min(nqe_positions['launches'].min(), hqe_positions['launches'].min())
    launches_max = max(nqe_positions['launches'].max(), hqe_positions['launches'].max())
    expected_hits_min = min(nqe_positions['expected_hits'].min(), hqe_positions['expected_hits'].min())
    expected_hits_max = max(nqe_positions['expected_hits'].max(), hqe_positions['expected_hits'].max())
    
    # Apply jitter to HQE and NQE positions
    nqe_positions = add_dynamic_jitter(nqe_positions)
    hqe_positions = add_dynamic_jitter(hqe_positions)

    # Plot charge fraction scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        nqe_positions['x_jitter'], nqe_positions['y_jitter'],
        c=nqe_positions['charge_fraction'], cmap="viridis", s=40, label='NQE', vmin=charge_fraction_min, vmax=charge_fraction_max
    )
    plt.scatter(
        hqe_positions['x_jitter'], hqe_positions['y_jitter'],
        c=hqe_positions['charge_fraction'], cmap="viridis", s=40, edgecolors='red', label='HQE', facecolors='none', vmin=charge_fraction_min, vmax=charge_fraction_max
    )
    
    polygon_patch = mplPolygon(polygon_coords, closed=True, fill = None , edgecolor='black', linewidth=2)
    polygon_patch_deepcore = mplPolygon(polygon_coords_deepcore, closed=True, fill = None , edgecolor='black', linewidth=2)
    plt.gca().add_patch(polygon_patch)
    plt.gca().add_patch(polygon_patch_deepcore)
    plt.colorbar(scatter, label="Charge Fraction (Σ(Total charge) / Σ(expected_hits))")
    plt.xlabel("DOM x-coordinate (jitter applied to HQE)")
    plt.ylabel("DOM y-coordinate (jitter applied to HQE)")
    plt.title("DOM Charge Fraction Scatter Plot (NQE & HQE)")
    plt.legend()
    pdf.savefig()
    plt.close()

    # Plot total charge scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        nqe_positions['x_jitter'], nqe_positions['y_jitter'],
        c=nqe_positions['total_charge'], cmap="viridis", s=40, label='NQE', vmin=total_charge_min, vmax=total_charge_max
    )
    plt.scatter(
        hqe_positions['x_jitter'], hqe_positions['y_jitter'],
        c=hqe_positions['total_charge'] , cmap="viridis", s=40, edgecolors='red', label='HQE', facecolors='none', vmin=total_charge_min, vmax=total_charge_max
    )
    plt.colorbar(scatter, label="Total Charge [C]")
    plt.xlabel("DOM x-coordinate (jitter applied to HQE)")
    plt.ylabel("DOM y-coordinate (jitter applied to HQE)")
    plt.title("DOM Total Charge Scatter Plot (NQE & HQE)")
    plt.legend()
    pdf.savefig()
    plt.close()
    
    # Plot launches vs total charge
    plt.figure(figsize=(10, 8))
    plt.scatter(nqe_positions['total_charge'], nqe_positions["launches"], label="NQE", alpha=0.7, c="blue")
    plt.scatter(hqe_positions['total_charge'], hqe_positions["launches"], label="HQE", alpha=0.7, c="orange", edgecolors='black')

    # Define cost functions for NQE and HQE
    def cost_function_nqe(m, c):
        return np.sum((nqe_launches - linear_model(nqe_total_charge, m, c))**2)

    def cost_function_hqe(m, c):
        return np.sum((hqe_launches - linear_model(hqe_total_charge, m, c))**2)

    # # Fit NQE data
    # minuit_nqe = Minuit(cost_function_nqe, m=1.0, c=0.0)
    # minuit_nqe.errordef = Minuit.LEAST_SQUARES
    # minuit_nqe.migrad()  # Perform the fit
    # slope_nqe, intercept_nqe = minuit_nqe.values["m"], minuit_nqe.values["c"]
    # slope_nqe_err, intercept_nqe_err = minuit_nqe.errors["m"], minuit_nqe.errors["c"]

    # # Fit HQE data
    # minuit_hqe = Minuit(cost_function_hqe, m=1.0, c=0.0)
    # minuit_hqe.errordef = Minuit.LEAST_SQUARES
    # minuit_hqe.migrad()  # Perform the fit
    # slope_hqe, intercept_hqe = minuit_hqe.values["m"], minuit_hqe.values["c"]
    # slope_hqe_err, intercept_hqe_err = minuit_hqe.errors["m"], minuit_hqe.errors["c"]

    # # Plot fitted lines
    # x_line_nqe = np.linspace(nqe_total_charge.min(), nqe_total_charge.max(), 500)
    # y_line_nqe = linear_model(x_line_nqe, slope_nqe, intercept_nqe)
    # plt.plot(
    #     x_line_nqe, y_line_nqe, label=f"NQE Fit: Slope = {slope_nqe:.2f} ± {slope_nqe_err:.2f}", color="blue", linestyle="--"
    # )

    # x_line_hqe = np.linspace(hqe_total_charge.min(), hqe_total_charge.max(), 500)
    # y_line_hqe = linear_model(x_line_hqe, slope_hqe, intercept_hqe)
    # plt.plot(
    #     x_line_hqe, y_line_hqe, label=f"HQE Fit: Slope = {slope_hqe:.2f} ± {slope_hqe_err:.2f}", color="orange", linestyle="--"
    # )

    plt.xlabel("Total Charge")
    plt.ylabel("Launches")
    plt.title("Launches vs Total Charge with Minuit Fits")
    plt.legend()
    plt.grid()
    pdf.savefig()
    plt.close()

    #print(f"NQE Fit: Slope = {slope_nqe:.2f} ± {slope_nqe_err:.2f}, Intercept = {intercept_nqe:.2f} ± {intercept_nqe_err:.2f}")
    #print(f"HQE Fit: Slope = {slope_hqe:.2f} ± {slope_hqe_err:.2f}, Intercept = {intercept_hqe:.2f} ± {intercept_hqe_err:.2f}")
    
       # Plot launches scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        nqe_positions['x_jitter'], nqe_positions['y_jitter'],
        c=nqe_positions['launches'], cmap="viridis", s=40, label='NQE', vmin=launches_min, vmax=launches_max
    )
    plt.scatter(
        hqe_positions['x_jitter'], hqe_positions['y_jitter'],
        c=hqe_positions['launches'], cmap="viridis", s=40, edgecolors='red', label='HQE', facecolors='none', vmin=launches_min, vmax=launches_max
    )
    plt.colorbar(scatter, label="Number of Events (Launches)")
    plt.xlabel("DOM x-coordinate (jitter applied to HQE)")
    plt.ylabel("DOM y-coordinate (jitter applied to HQE)")
    plt.title("DOM Launches Scatter Plot (NQE & HQE)")
    plt.legend()
    pdf.savefig()
    plt.close()

    # Plot expected hits scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        nqe_positions['x_jitter'], nqe_positions['y_jitter'],
        c=nqe_positions['expected_hits'], cmap="viridis", s=40, label='NQE', vmin=expected_hits_min, vmax=expected_hits_max
    )
    plt.scatter(
        hqe_positions['x_jitter'], hqe_positions['y_jitter'],
        c=hqe_positions['expected_hits'], cmap="viridis", s=40, edgecolors='red', label='HQE', facecolors='none', vmin=expected_hits_min, vmax=expected_hits_max
    )
    plt.colorbar(scatter, label="Expected Hits")
    plt.xlabel("DOM x-coordinate (jitter applied to HQE)")
    plt.ylabel("DOM y-coordinate (jitter applied to HQE)")
    plt.title("DOM Expected Hits Scatter Plot (NQE & HQE)")
    plt.legend()
    pdf.savefig()
    plt.close()

    # Histogram for DOM participation (Launches)
    plt.figure(figsize=(10, 6))
    plt.hist(nqe_positions['launches'], bins=20, alpha=0.6, label="NQE", color='blue', edgecolor='black')
    plt.hist(hqe_positions['launches'], bins=20, alpha=0.6, label="HQE", color='orange', edgecolor='black')
    plt.xlabel("Number of Events (Launches)")
    plt.ylabel("Number of DOMs")
    plt.title("Distribution of Events DOMs Participate In")
    plt.legend()
    plt.grid()
    pdf.savefig()
    plt.close()
    print("Saved histogram plot: launches_histogram.pdf")
    
    # # Split NQE into three groups and HQE into one group
    # nqe_group1 = (nqe_expected_hits < 100)
    # nqe_group2 = (100 <= nqe_expected_hits) & (nqe_expected_hits < 300)
    # nqe_group3 = (300 <= nqe_expected_hits) & (nqe_expected_hits < 600)
    # hqe_group = hqe_expected_hits > 0  # All HQE DOMs

    # # Prepare data for each group
    # groups = [
    #     (nqe_positions['expected_hits'][nqe_group1], nqe_positions['launches'][nqe_group1], "NQE < 100", "blue"),
    #     (nqe_positions['expected_hits'][nqe_group2], nqe_positions['launches'][nqe_group2], "100 ≤ NQE < 300", "green"),
    #     (nqe_positions['expected_hits'][nqe_group3], nqe_positions['launches'][nqe_group3], "300 ≤ NQE < 600", "purple"),
    #     (hqe_positions['expected_hits'][hqe_group], hqe_positions['launches'][hqe_group], "HQE", "orange")
    # ]
    # def group_cost_function(m, c, x, y):
    #     return np.sum((y - linear_model(x, m, c))**2)
    
    # fits = []
    # for x, y, label, color in groups:
    #     def cost_for_groyp(m,c):
    #         return group_cost_function(m,c,x,y)
    #     # Fit the group
    #     minuit = Minuit(cost_for_groyp, m=1.0, c=0.0)
    #     minuit.errordef = Minuit.LEAST_SQUARES
    #     minuit.migrad()
    #     m, c = minuit.values["m"], minuit.values["c"]
    #     print(f"Fit for {label}: Slope = {m:.2f}, Intercept = {c:.2f}")
    #     m_err, c_err = minuit.errors["m"], minuit.errors["c"]
    #     fits.append((m, c, m_err, c_err, label, color))

    
    
    plt.figure(figsize=(10, 8))
    plt.scatter(nqe_positions['expected_hits'], nqe_positions['launches'], label="NQE", alpha=0.7, c="blue")
    plt.scatter(hqe_positions['expected_hits'], hqe_positions['launches'], label="HQE", alpha=0.7, c="orange", edgecolors='black')
    # # Plot fitted lines
    # for (x, _, label, color), (m, c, m_err, c_err, _, _) in zip(groups, fits):
    #     # Define the range of x values for this group
    #     if x.size == 0:
    #         continue
    #     x_min, x_max = x.min(), x.max()
    #     x_group = np.linspace(x_min, x_max, 100)  # Generate x values within the group's range

    #     # Calculate the corresponding y values for the fitted line
    #     y_line = linear_model(x_group, m, c)

    #     # Plot the line segment for this group
    #     plt.plot(x_group, y_line, label=f"{label}: Slope = {m:.2f} ± {m_err:.2f}", color=color, linestyle="--")

    plt.xlabel("Expected Hits")
    plt.ylabel("Launches")
    plt.title("Launches vs Expected Hits")
    plt.legend()
    plt.grid()
    pdf.savefig()
    plt.close()
    print("Saved scatter plot: launches_vs_expected_hits.pdf")
    
    # Plot showing RIDE for NQE and HQE with RIDE values on y-axis and x-position on x-axis
    
    nqe_positions.loc[nqe_positions['expected_hits'] == 0, 'expected_hits'] = 1  # Avoid division by zero
    hqe_positions.loc[hqe_positions['expected_hits'] == 0, 'expected_hits'] = 1  # Avoid division by zero
    
    monitor_old = np.median(nqe_positions['total_charge'] / nqe_positions['expected_hits'])
    
    #monitor = calculate_monitor_inside_polygon(pos_array_NQE, nqe_positions['total_charge'], nqe_positions['expected_hits'], polygon_coords)
    #monitor_deepcore = calculate_monitor_inside_polygon(pos_array_NQE, nqe_positions['total_charge'], nqe_positions['expected_hits'], polygon_coords_deepcore)
    monitor = calculate_monitor_inside_polygon(nqe_positions, polygon_coords)
    monitor_deepcore = calculate_monitor_inside_polygon(nqe_positions, polygon_coords_deepcore)
 
    plt.figure(figsize=(10, 8))
    plt.scatter(nqe_positions['dom_x'], (nqe_positions['total_charge'] / nqe_positions['expected_hits']) / monitor_deepcore, label="NQE", alpha=0.7, c="blue")
    plt.scatter(hqe_positions['dom_x'], (hqe_positions['total_charge'] / hqe_positions['expected_hits']) / monitor_deepcore, label="HQE", alpha=0.7, c="orange", edgecolors='black')
    plt.xlabel("DOM x-coordinate")
    plt.ylabel("RIDE Deepcore (Monitored to NQE)") 
    plt.title("RIDE vs DOM x-coordinate")
    plt.legend()
    plt.grid()
    pdf.savefig()
    
    plt.figure(figsize=(10, 8))
    plt.scatter(nqe_positions['dom_x'], (nqe_positions['total_charge'] / nqe_positions['expected_hits']) / monitor, label="NQE", alpha=0.7, c="blue")
    plt.scatter(hqe_positions['dom_x'], (hqe_positions['total_charge'] / hqe_positions['expected_hits']) / monitor, label="HQE", alpha=0.7, c="orange", edgecolors='black')
    
    plt.xlabel("DOM x-coordinate")
    plt.ylabel("RIDE (Monitored to NQE)")
    plt.title("RIDE vs DOM x-coordinate")
    plt.legend()
    plt.grid()
    pdf.savefig()
    print("Saved scatter plot: ride_vs_dom_x.pdf")
    
    
    plt.figure(figsize=(10, 8))
    plt.scatter(nqe_positions['dom_x'], (nqe_positions['total_charge'] / nqe_positions['expected_hits']) / nqe_positions_charge_fraction_79, label="NQE", alpha=0.7, c="blue")
    plt.scatter(hqe_positions['dom_x'], (hqe_positions['total_charge'] / hqe_positions['expected_hits']) / nqe_positions_charge_fraction_79, label="HQE", alpha=0.7, c="orange", edgecolors='black')
    
    plt.xlabel("DOM x-coordinate")
    plt.ylabel("RIDE (Monitored to string 79)")
    plt.title("RIDE (Monitored to string 79) vs DOM x-coordinate")
    plt.legend()
    plt.grid()
    pdf.savefig()
    print("Saved scatter plot: ride_vs_dom_x.pdf")
    
    # Scatter plot showing 
    # Calculate charge fractions
    
    
    
    # nqe_launche_charge_fraction = np.where(nqe_launches > 0, nqe_total_charge / nqe_launches, 0)
    # hqe_launce_charge_fraction = np.where(hqe_launches > 0, hqe_total_charge / hqe_launches, 0)
    # nqe_positions['charge_fraction_launche'] = nqe_launche_charge_fraction
    # hqe_positions['charge_fraction_launche'] = hqe_launce_charge_fraction
    # charge_launche_fraction_min = min(nqe_launche_charge_fraction.min(), hqe_launce_charge_fraction.min())
    # charge_launche_fraction_max = max(nqe_launche_charge_fraction.max(), hqe_launce_charge_fraction.max())
    
    # plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(
    #     nqe_positions['x_jitter'], nqe_positions['y_jitter'],
    #     c=nqe_positions['charge_fraction_launche'], cmap="viridis", s=40, label='NQE', vmin=charge_launche_fraction_min, vmax=charge_launche_fraction_max
    # )
    # plt.scatter(
    #     hqe_positions['x_jitter'], hqe_positions['y_jitter'],
    #     c=hqe_positions['charge_fraction_launche'], cmap="viridis", s=40, edgecolors='red', label='HQE', facecolors='none', vmin=charge_launche_fraction_min, vmax=charge_launche_fraction_max
    # )
    # plt.colorbar(scatter, label="Charge Fraction (Σ(Total charge) / Σ(launches))")
    # plt.xlabel("DOM x-coordinate (jitter applied to HQE)")
    # plt.ylabel("DOM y-coordinate (jitter applied to HQE)")
    # plt.title("DOM Total Charge Scatter Plot (NQE & HQE)")
    # plt.legend()
    # pdf.savefig()
    # plt.close()
    
    # Make a plot showing efficiency ratio for NQE and HQE DOMs (Launches / Expected Hits) as a function of x-coordinate
    nqe_efficiency_ratio = nqe_positions['launches'] / nqe_positions['expected_hits']
    hqe_efficiency_ratio = hqe_positions['launches'] / hqe_positions['expected_hits']
    nqe_positions['efficiency_ratio'] = nqe_efficiency_ratio
    hqe_positions['efficiency_ratio'] = hqe_efficiency_ratio
    
    plt.figure(figsize=(10, 8))
    plt.scatter(nqe_positions['dom_x'], nqe_positions['efficiency_ratio'], label="NQE", alpha=0.7, c="blue")
    plt.scatter(hqe_positions['dom_x'], hqe_positions['efficiency_ratio'], label="HQE", alpha=0.7, c="orange", edgecolors='black')
    
    plt.xlabel("DOM x-coordinate")
    plt.ylabel("Efficiency Ratio (Launches / Expected Hits)")
    plt.title("Efficiency Ratio vs DOM x-coordinate")
    plt.legend()
    plt.grid()
    pdf.savefig()
    plt.close()
    
    

def plot_distance_vs_charge(pulses_df, pos_array, true_x, true_y, true_z, true_zenith, true_azimuth, track_length):
    """
    Plots total DOM charge vs. distance from track.
    Args:
        pulses_df (pd.DataFrame): DataFrame containing DOM positions and observed charges.
        pos_array (np.ndarray): Array of DOM positions.
        true_x, true_y, true_z (float): Stopping position of the muon.
        true_zenith, true_azimuth (float): Muon track angles.
        track_length (float): Length of the muon track.
    """
    # Calculate distances of DOMs to the track
    min_distances, _ = calculate_minimum_distance_to_track(
        pos_array, true_x, true_y, true_z, true_zenith, true_azimuth, track_length
    )

    # Aggregate charges for DOMs
    dom_positions = pulses_df[["dom_x", "dom_y", "dom_z"]].to_numpy()
    charges = pulses_df["charge"].to_numpy()
    
    # Map DOMs in pulses_df to indices in pos_array
    pos_indices = np.array([
        np.where((pos_array == dom_pos).all(axis=1))[0][0] for dom_pos in dom_positions
    ])

    # Aggregate total charges for each DOM
    total_charges = np.zeros(len(pos_array))
    np.add.at(total_charges, pos_indices, charges)

    # Prepare data for plotting
    distances = min_distances[pos_indices]
    charges_for_plot = total_charges[pos_indices]

    # Generate scatter plot
    plt.figure(figsize=(10, 6))
    # Make a scatter plot of total DOM charge vs. distance to track using a log color scale
    plt.scatter(distances, charges_for_plot, alpha=0.7, s=20, c='blue')
    plt.xlabel("Distance to Track [m]")
    plt.ylabel("Total DOM Charge [PE]")
    plt.title("Total DOM Charge vs. Distance to Track")
    plt.grid(True)
    plt.show()


def plot_charge_distribution_by_DOM(doms_in_range, pdf):
    """
    Plot charge distributions for all DOMs in the given range, grouped by string, dom_number, and rde.
    Uses a custom color cycle for DOMs and splits into multiple plots if necessary.
    """
    # Group DOMs by string, dom_number, and rde
    grouped_doms = doms_in_range.groupby(['string', 'dom_number', 'rde'])

    # Convert the group keys into a list for iteration
    dom_keys = list(grouped_doms.groups.keys())

    # Split DOMs into chunks (e.g., 10 DOMs per plot) for clarity
    chunk_size = 5
    dom_chunks = [dom_keys[i:i + chunk_size] for i in range(0, len(dom_keys), chunk_size)]

    # Plot each chunk of DOMs
    for chunk_idx, chunk in enumerate(dom_chunks):
        plt.figure(figsize=(10, 6))
        for i, (string, dom_number, rde) in enumerate(chunk):
            dom_charges = grouped_doms.get_group((string, dom_number, rde))['charge']
            linestyle = "solid" if rde == 1.0 else "dashed"  # Solid for rde == 1.0, dashed for rde == 1.35
            label = f"String {string}, DOM {dom_number}, RDE {rde}"

            # Use custom colors and cycle them
            #color = custom_colors[i % len(custom_colors)]

            plt.hist(
                dom_charges, bins=50, histtype='step', lw=2,
                label=label, linestyle=linestyle
            )

        plt.xlabel("Charge [C]")
        plt.ylabel("Counts")
        plt.title(f"Charge Distribution for DOMs (Chunk {chunk_idx + 1})")
        plt.legend(fontsize=8)
        plt.grid()
        pdf.savefig()  # Save the current figure to the PDF
        plt.close()
    
def plot_dom_participation_heatmap(pulses_df, pdf):
    """
    Plot a heatmap showing DOM participation in events.
    """
    heatmap_data = pulses_df.groupby(['string', 'dom_number']).size().unstack(fill_value=0)

    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap_data, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label="Number of Events")
    plt.ylabel("String")
    plt.xlabel("DOM Number")
    plt.title("DOM Participation Heatmap")
    pdf.savefig()
    plt.close()
    
def plot_distance_histogram(min_distances, pdf):
    """
    Plot a histogram of minimum distances to the track for all DOMs.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(min_distances, bins=50, alpha=0.6, color='blue', edgecolor='black')
    plt.xlabel("Minimum Distance to Track [m]")
    plt.ylabel("Counts")
    plt.title("Minimum Distance to Track Histogram")
    plt.grid()
    pdf.savefig()
    plt.close()

def plot_total_charge_vs_distance(dom_distances, dom_charges, pdf):
    """
    Plot total DOM charge vs. distance to track.

    Args:
        dom_distances (np.ndarray): Array of DOM distances.
        dom_charges (np.ndarray): Array of DOM charges.
    """
    plt.figure(figsize=(10, 6))
    plt.hist2d(dom_charges, dom_distances, bins=(50, 50), cmap='viridis')
    plt.ylabel("Distance to Track [m]")
    plt.xlabel("Total DOM Charge [PE]")
    plt.title("Total DOM Charge vs. Distance to Track")
    plt.grid(True)
    pdf.savefig()
    plt.close()



def plot_starting_points(start_xs, start_ys, start_zs, pdf):
    """
    Plot the starting points of the muon track using imshow, first in the x-y plane and then in the x-z plane.
    """
    heatmap, xedges, yedges = np.histogram2d(start_xs, start_ys, bins=100)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis')
    plt.colorbar(label="Number of Events")
    plt.xlabel("X-coordinate [m]")
    plt.ylabel("Y-coordinate [m]")
    plt.title("Starting Points of Muon Track (X-Y Plane)")
    pdf.savefig()
    plt.close()
    
    heatmap, xedges, zedges = np.histogram2d(start_xs, start_zs, bins=100)
    extent = [xedges[0], xedges[-1], zedges[0], zedges[-1]]
    
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis')
    plt.colorbar(label="Number of Events")
    plt.xlabel("X-coordinate [m]")
    plt.ylabel("Z-coordinate [m]")
    plt.title("Starting Points of Muon Track (X-Z Plane)")
    pdf.savefig()
    plt.close()
    
def plot_unique_dom_launches_histogram(pulses_df, pdf=None):
    """
    Plot a histogram of unique DOM launches per event.
    
    Parameters:
        pulses_df: pandas.DataFrame
            The DataFrame containing DOM pulse data.
        pdf: PdfPages or None
            If provided, save the histogram to the PDF.
    """
    # Group by event and DOM identifiers to count unique launches
    dom_launch_counts = pulses_df.groupby(['event_no', 'dom_x', 'dom_y', 'dom_z']).size()

    # Group by event to find the number of unique DOM launches per event
    launches_per_event = dom_launch_counts.groupby('event_no').sum()

    # Plot the histogram of unique DOM launches per event
    plt.figure(figsize=(10, 6))
    plt.hist(launches_per_event, bins=50, color='blue', range = (0,200), alpha=0.7, edgecolor='black')
    plt.xlabel("Number of Unique DOM Launches per Event")
    plt.ylabel("Frequency (Number of Events)")
    plt.title("Histogram of Unique DOM Launches Per Event")
    plt.grid()

    # Save to PDF if provided
    if pdf:
        pdf.savefig()
        plt.close()
    else:
        plt.show()



def plot_high_expected_hits_histogram(
    nqe_positions, nqe_expected_hits, nqe_launches, dom_times_combined, dom_charges_combined, dom_distances_combined, total_charge_NQE, pdf
):
    """
    Plot histograms of charge and timing for DOMs in two groups of strings.
    """
    # Convert the NQE positions into a DataFrame with the correct columns
    nqe_df = pd.DataFrame(nqe_positions, columns=['dom_x', 'dom_y', 'dom_z', 'string', 'dom_number'])
    nqe_df["expected_hits"] = nqe_expected_hits
    nqe_df["launches"] = nqe_launches
    nqe_df["total_charge"] = total_charge_NQE
    nqe_df["dom_idx"] = nqe_df.index

    # Define the two groups of strings
    group1_strings = [79, 80, 25, 26, 47, 35, 34, 36, 46, 44, 37, 54]
    group2_strings = [18, 55, 56, 17]

    # Filter for Group 1 strings
    group1_doms = nqe_df[nqe_df["string"].isin(group1_strings)]
    if group1_doms.empty:
        print(f"No DOMs found in Group 1 strings: {group1_strings}")
    else:
        # Plot Charge Distribution for Group 1
        plt.figure(figsize=(12, 8))
        for _, dom in group1_doms.iterrows():
            dom_idx = int(dom["dom_idx"])
            plt.hist(
                dom_charges_combined[dom_idx], bins=50, histtype='step', lw=2,
                label=f"String {int(dom['string'])}, DOM {int(dom['dom_number'])}, Total Charge = {dom['total_charge']:.2f}, Max_charge = {np.max(dom_charges_combined[dom_idx]):.2f}"
            )
        plt.xlabel("Charge [C]")
        plt.ylabel("Counts")
        plt.title(f"Charge Distribution for DOMs in Strings: {group1_strings}")
        plt.legend(fontsize=8)
        plt.grid()
        pdf.savefig()
        plt.close()

        # Plot Timing Distribution for Group 1
        plt.figure(figsize=(12, 8))
        for _, dom in group1_doms.iterrows():
            dom_idx = int(dom["dom_idx"])
            plt.hist(
                dom_times_combined[dom_idx], bins=50, histtype='step', range =(0,7500), lw=2,
                label=f"String {int(dom['string'])}, DOM {int(dom['dom_number'])}, Total Charge = {dom['total_charge']:.2f}"
            )
        plt.xlabel("Time (ns)")
        plt.ylabel("Counts")
        plt.title(f"Time Distribution for DOMs in Strings: {group1_strings}")
        plt.legend(fontsize=8)
        plt.grid()
        pdf.savefig()
        plt.close()

    # Filter for Group 2 strings
    group2_doms = nqe_df[nqe_df["string"].isin(group2_strings)]
    if group2_doms.empty:
        print(f"No DOMs found in Group 2 strings: {group2_strings}")
    else:
        # Plot Charge Distribution for Group 2
        plt.figure(figsize=(12, 8))
        for _, dom in group2_doms.iterrows():
            dom_idx = int(dom["dom_idx"])
            plt.hist(
                dom_charges_combined[dom_idx], bins=50, histtype='step', lw=2,
                label=f"String {int(dom['string'])}, DOM {int(dom['dom_number'])}, Total Charge = {dom['total_charge']:.2f}, Max_charge = {np.max(dom_charges_combined[dom_idx]):.2f}"
            )
        plt.xlabel("Charge [C]")
        plt.ylabel("Counts")
        plt.title(f"Charge Distribution for DOMs in Strings: {group2_strings}")
        plt.legend(fontsize=8)
        plt.grid()
        pdf.savefig()
        plt.close()

        # Plot Timing Distribution for Group 2
        plt.figure(figsize=(12, 8))
        for _, dom in group2_doms.iterrows():
            dom_idx = int(dom["dom_idx"])
            plt.hist(
                dom_times_combined[dom_idx], bins=50, histtype='step',range =(0,7500), lw=2,
                label=f"String {int(dom['string'])}, DOM {int(dom['dom_number'])}, Total Charge = {dom['total_charge']:.2f}"
            )
        plt.xlabel("Time (ns)")
        plt.ylabel("Counts")
        plt.title(f"Time Distribution for DOMs in Strings: {group2_strings}")
        plt.legend(fontsize=8)
        plt.grid()
        pdf.savefig()
        plt.close()

    all_times = np.concatenate(dom_times_combined)
    all_charges = np.concatenate(dom_charges_combined)
    plt.figure(figsize=(12, 8))
    plt.hist2d(all_times, all_charges, bins=50, cmap='viridis', range=[[0, 7500], [0, 5]])
    # Find the index of the maximum charge in `all_charges`
    max_charge_idx = np.argmax(all_charges)

    # Use the corresponding index from `nqe_df`
    if max_charge_idx < len(nqe_df):
        max_charge_row = nqe_df.iloc[max_charge_idx]  # Use iloc to get the row by position
        print(f"Max charge observed by a DOM: {all_charges[max_charge_idx]} string {max_charge_row['string']} dom {max_charge_row['dom_number']}")
    else:
        print(f"Error: max_charge_idx {max_charge_idx} is out of bounds for nqe_df with length {len(nqe_df)}")
    plt.colorbar(label="Counts")
    plt.xlabel("Time (ns)")
    plt.ylabel("Charge [C]")
    plt.title("Charge vs Time for All DOMs")
    pdf.savefig()
    plt.close()

# -------------------------------
# Main Function
# -------------------------------
def main():
    file_path = '/groups/icecube/simon/GNN/workspace/Scripts/filtered_all_no_cuts.db'
    #file_path = '/groups/icecube/simon/GNN/workspace/filtered.db'
    con = sqlite3.connect(file_path)
    cursor = con.cursor()
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_no ON SplitInIcePulsesSRT(event_no);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_no_truth ON truth(event_no);")
    
    con.commit()
    df_truth = pd.read_sql_query("SELECT * FROM truth", con)
    filtered_events = df_truth['event_no'].unique()
    
    pulses_df = pd.read_sql_query("SELECT  dom_x, dom_y, dom_z, charge, dom_time, event_no, string,rde, dom_number FROM SplitInIcePulsesSRT", con)
    con.close()
    
    pulses_df = pulses_df[pulses_df['event_no'].isin(filtered_events)]
    
    output_dir = "/groups/icecube/simon/GNN/workspace/Plots"
    os.makedirs(output_dir, exist_ok=True)
    # Filter DOMs by depth and strings
    all_depth_bins = np.arange(-500, 300, 10)
    pulses_df['depth_bin'] = pd.cut(pulses_df['dom_z'], bins=all_depth_bins, labels=all_depth_bins[:-1])
    pulses_df = pulses_df[pulses_df['string'] < 87.0]

    # Define focus depth bins and distance ranges
    focus_depth_bins = [140, -210]#, -310, -320]
    distance_ranges = [(10, 110)]#, (5,20)]#, (5, 100), (5, 200)]
    pulses_df['corrected_dom_time'] = pulses_df['dom_time'] - pulses_df.groupby('event_no')['dom_time'].transform('min')


    for focus_depth_bin in focus_depth_bins:
        for distance_range in distance_ranges:
            print(f"Processing focus_depth_bin: {focus_depth_bin}, distance_range: {distance_range}")

            # Filter DOMs based on depth_bin and rde
            doms_in_range = pulses_df[(pulses_df['depth_bin'] == focus_depth_bin)]

            # Separate NQE and HQE DOMs
            filtered_NQE = doms_in_range[(doms_in_range['rde'] == 1.0)]
            filtered_HQE = doms_in_range[(doms_in_range['rde'] == 1.35)]

            # Combine N QE and HQE DOM positions into a single array
            pos_array_combined = pd.concat([filtered_NQE, filtered_HQE])[['dom_x', 'dom_y', 'dom_z','string','dom_number']].drop_duplicates().to_numpy()
            pos_array_combined_coords = pos_array_combined[:, :3]
            # Map combined DOM positions to their respective groups (NQE or HQE)
            nqe_positions_set = set(filtered_NQE[['dom_x', 'dom_y', 'dom_z','string','dom_number']].apply(tuple, axis=1).tolist())
            group_map = {tuple(pos): "NQE" if tuple(pos) in nqe_positions_set else "HQE" for pos in pos_array_combined}

            # Split events into chunks for parallel processing
            event_chunks = np.array_split(df_truth, 12)
            filtered_pulses_df = pulses_df[pulses_df["event_no"].isin(df_truth["event_no"])]
            pulses_chunks = [filtered_pulses_df[filtered_pulses_df["event_no"].isin(chunk["event_no"])] for chunk in event_chunks]
            cols_to_send = ["event_no", "dom_x", "dom_y", "dom_z", "charge", "corrected_dom_time","rde"]
            pulses_chunks = [chunk[cols_to_send].copy() for chunk in pulses_chunks]
            # Parallel processing for all DOMs
            try:
                with ProcessPoolExecutor(max_workers=12) as executor:
                    combined_results = list(
                        executor.map(
                            process_chunk,
                            event_chunks,  # Unique event chunks
                            pulses_chunks,  # Pre-filtered pulses per chunk
                            [pos_array_combined_coords] * 12,
                            [distance_range] * 12,
                        )
                    )
            except Exception as e:
                print(f"ERROR in main process: {e}")
                import traceback
                traceback.print_exc()
            # Aggregate results
            expected_hits_combined = np.sum([result[0] for result in combined_results], axis=0)
            total_charge_combined = np.sum([result[1] for result in combined_results], axis=0)
            launches_combined = np.sum([result[2] for result in combined_results], axis=0)
            combined_dom_event_map = np.concatenate([result[3] for result in combined_results])
            dom_position_map = combined_results[0][4]
            distances_combined = np.concatenate([result[5] for result in combined_results])
            distances_combined_flat = distances_combined.flatten()
            starting_points_events = np.concatenate([result[9] for result in combined_results], axis=0)
            
            dom_times_combined = [[] for _ in range(len(pos_array_combined))]
            dom_charges_combined = [[] for _ in range(len(pos_array_combined))]
            dom_distance_combined = [[] for _ in range(len(pos_array_combined))]
            # Loop through results and collect both time and charge data
            for result in combined_results:
                for idx, (times, charges, distances) in enumerate(zip(result[6], result[7], result[8])):
                    if len(times) > 0 and len(charges) > 0 and len(distances) > 0:
                        dom_times_combined[idx].extend(times)
                        dom_charges_combined[idx].extend(charges)
                        dom_distance_combined[idx].extend(distances)


            # Convert to numpy arrays for consistency
            dom_times_combined = [np.array(times) for times in dom_times_combined]
            dom_charges_combined = [np.array(charges) for charges in dom_charges_combined]
            dom_distance_combined = [np.array(distances) for distances in dom_distance_combined]
            all_dom_distances = np.concatenate([res for res in [result[10] for result in combined_results] if res.size > 0], axis=0)
            all_dom_charges = np.concatenate([res for res in [result[11] for result in combined_results] if res.size > 0], axis=0)
            start_xs = starting_points_events[:, 0]
            start_ys = starting_points_events[:, 1]
            start_zs = starting_points_events[:, 2]
                             
            # Split results into NQE and HQE based on group map
            nqe_mask = [group_map[tuple(pos)] == "NQE" for pos in pos_array_combined]
            expected_hits_NQE = expected_hits_combined[nqe_mask]
            total_charge_NQE = total_charge_combined[nqe_mask]
            launches_NQE = launches_combined[nqe_mask]

            hqe_mask = [group_map[tuple(pos)] == "HQE" for pos in pos_array_combined]
            expected_hits_HQE = expected_hits_combined[hqe_mask]
            total_charge_HQE = total_charge_combined[hqe_mask]
            launches_HQE = launches_combined[hqe_mask]

            pos_array_NQE = pos_array_combined[nqe_mask]
            pos_array_HQE = pos_array_combined[hqe_mask]
            
            # Generate PDF filename
            pdf_filename = os.path.join(output_dir,f"combined_plots_part_check_{abs(focus_depth_bin)}_{distance_range[0]}_{distance_range[1]}.pdf")
            polygon_coords = [(-510,-110),(-300,405),(210,520),(520,130),(300,-390),(-220,-490)]
            polygon_coords_deepcore = [(-100,-320),(-260,-100),(-120,220),(170,280),(360,50),(210,-280)]
            # Generate all plots into the PDF
            with PdfPages(pdf_filename) as pdf:
                plot_charge_fraction_and_launches(
            pos_array_NQE, pos_array_HQE, expected_hits_NQE, total_charge_NQE, launches_NQE,
            expected_hits_HQE, total_charge_HQE, launches_HQE, polygon_coords, polygon_coords_deepcore, pdf
                )
                #plot_charge_distribution_by_DOM(doms_in_range, pdf)
                plot_dom_participation_heatmap(pulses_df, pdf)
                #plot_efficiency_ratio(launches_NQE, expected_hits_NQE, launches_HQE, expected_hits_HQE, pdf)
                plot_distance_histogram(distances_combined_flat, pdf)
                plot_high_expected_hits_histogram(pos_array_NQE, expected_hits_NQE, launches_NQE, dom_times_combined, dom_charges_combined, dom_distance_combined, total_charge_NQE,  pdf)
                plot_unique_dom_launches_histogram(pulses_df, pdf)
                #plot_starting_points(start_xs, start_ys, start_zs, pdf)
                plot_total_charge_vs_distance(all_dom_distances, all_dom_charges, pdf)

            print(f"Saved PDF: {pdf_filename}")
            
if __name__ == "__main__":
    lp = LineProfiler()
    lp.add_function(process_events_with_charge_and_launches)
    lp.add_function(calculate_minimum_distance_to_track)
    lp_wrapper = lp(main)
    lp_wrapper()
    lp.print_stats()
