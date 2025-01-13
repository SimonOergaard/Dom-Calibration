import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from shapely.geometry import Polygon, Point

from matplotlib.patches import Polygon as mplPolygon
from concurrent.futures import ProcessPoolExecutor
import sqlite3
from tqdm import tqdm
from iminuit import Minuit


# -------------------------------
# Utility Functions
# -------------------------------


def process_events_with_charge_and_launches(event_chunk, pulses_df, pos_array, distance_range):
    """
    Optimized function to process events, counting expected hits, total charge, and launches.
    """
    # Create a mapping between DOM positions and their indices
    dom_position_map = {tuple(pos): idx for idx, pos in enumerate(pos_array)}

    # Initialize arrays to store results
    expected_hits = np.zeros(len(pos_array))
    total_charge = np.zeros(len(pos_array))  # Reflects cumulative charge over all pulses
    launches = np.zeros(len(pos_array), dtype=int)  # Reflects unique event counts for each DOM
    dom_event_map = []  # Store mappings between DOM indices and event numbers

    # Preprocess pulses for quick lookup
    pulses_np = pulses_df.to_numpy()
    pulse_event_col = pulses_df.columns.get_loc("event_no")
    pulse_coords_cols = [pulses_df.columns.get_loc(c) for c in ["dom_x", "dom_y", "dom_z"]]
    pulse_charge_col = pulses_df.columns.get_loc("charge")

    for _, event in tqdm(event_chunk.iterrows(), total=len(event_chunk), desc="Processing Events"):
        true_x, true_y, true_z = event[['position_x', 'position_y', 'position_z']]
        true_zenith, true_azimuth = event[['zenith', 'azimuth']]
        track_length = event['track_length']
        event_no = event['event_no']

        # Calculate minimum distances to the track
        min_distances = calculate_minimum_distance_to_track(
            pos_array, true_x, true_y, true_z, true_zenith, true_azimuth, track_length
        )

        # Identify DOMs within the distance range
        doms_in_range_mask = (distance_range[0] <= min_distances) & (min_distances < distance_range[1])
        dom_indices = np.where(doms_in_range_mask)[0]

        # Increment expected hits for DOMs in range
        expected_hits[dom_indices] += 1
        dom_event_map.extend([(dom_idx, event_no) for dom_idx in dom_indices])

        # Filter event pulses
        event_pulses = pulses_np[pulses_np[:, pulse_event_col] == event_no]
        unique_pulses = np.unique(event_pulses[:, pulse_coords_cols], axis=0)

        # Create a set of unique DOM positions for quick lookup
        unique_positions = {tuple(pos) for pos in unique_pulses}

        # Track DOM launches and aggregate total charge
        for dom_idx in dom_indices:
            dom_position = tuple(pos_array[dom_idx])

            # Check if the DOM is in the unique positions
            if dom_position in unique_positions:
                launches[dom_idx] += 1

            # Aggregate total charge for this DOM
            matched_pulses = event_pulses[
                (event_pulses[:, pulse_coords_cols[0]] == dom_position[0]) &
                (event_pulses[:, pulse_coords_cols[1]] == dom_position[1]) &
                (event_pulses[:, pulse_coords_cols[2]] == dom_position[2])
            ]
            total_charge[dom_idx] += np.sum(matched_pulses[:, pulse_charge_col])

    return expected_hits, total_charge, launches, dom_event_map


def process_chunk(chunk, pulses_df, pos_array, distance_range):
    return process_events_with_charge_and_launches(chunk, pulses_df, pos_array, distance_range)

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

def calculate_minimum_distance_to_track(pos_array, true_x, true_y, true_z, true_zenith, true_azimuth, track_length, steps=10000):
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

def calculate_monitor_inside_polygon(pos_array, total_charge, expected_hits, polygon_coords):
    """
    Calculate the monitor value (median of charge fraction) for DOMs inside the polygon.
    """
    polygon = Polygon(polygon_coords)
    inside_indices = [
        idx for idx, (x, y, _) in enumerate(pos_array) if polygon.contains(Point(x, y))
    ]
    charge_fractions = total_charge[inside_indices] / np.maximum(expected_hits[inside_indices], 1)
    return np.median(charge_fractions)

def linear_model(x, m, c):
    """
    Linear model for Minuit fitting.
    """
    return m * x + c






def plot_charge_fraction_and_launches(
            pos_array_NQE, pos_array_HQE, nqe_expected_hits, nqe_total_charge, nqe_launches,
            hqe_expected_hits, hqe_total_charge, hqe_launches, polygon_coords, pdf
):
    """
    Plot charge fraction scatter plot, launches vs total charge scatter plot, and fits using Minuit for NQE and HQE DOMs.
    """
    
    from iminuit import Minuit

    def linear_model(x, m, c):
        """Linear model for fitting."""
        return m * x + c

    nqe_positions = pd.DataFrame(pos_array_NQE, columns=['dom_x', 'dom_y', 'dom_z'])
    hqe_positions = pd.DataFrame(pos_array_HQE, columns=['dom_x', 'dom_y', 'dom_z'])

    # Calculate charge fractions
    nqe_charge_fraction = np.where(nqe_expected_hits > 0, nqe_total_charge / nqe_expected_hits, 0)
    hqe_charge_fraction = np.where(hqe_expected_hits > 0, hqe_total_charge / hqe_expected_hits, 0)

    # Calculate global min and max for consistent colorbar scaling
    charge_fraction_min = min(nqe_charge_fraction.min(), hqe_charge_fraction.min())
    charge_fraction_max = max(nqe_charge_fraction.max(), hqe_charge_fraction.max())
    total_charge_min = min(nqe_total_charge.min(), hqe_total_charge.min())
    total_charge_max = max(nqe_total_charge.max(), hqe_total_charge.max())
    launches_min = min(nqe_launches.min(), hqe_launches.min())
    launches_max = max(nqe_launches.max(), hqe_launches.max())
    expected_hits_min = min(nqe_expected_hits.min(), hqe_expected_hits.min())
    expected_hits_max = max(nqe_expected_hits.max(), hqe_expected_hits.max())

    # Ensure charge fractions align with DOM positions
    nqe_positions['charge_fraction'] = nqe_charge_fraction
    hqe_positions['charge_fraction'] = hqe_charge_fraction
    nqe_positions['expected_hits'] = nqe_expected_hits
    hqe_positions['expected_hits'] = hqe_expected_hits

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
    plt.gca().add_patch(polygon_patch)
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
        c=nqe_total_charge, cmap="viridis", s=40, label='NQE', vmin=total_charge_min, vmax=total_charge_max
    )
    plt.scatter(
        hqe_positions['x_jitter'], hqe_positions['y_jitter'],
        c=hqe_total_charge, cmap="viridis", s=40, edgecolors='red', label='HQE', facecolors='none', vmin=total_charge_min, vmax=total_charge_max
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
    plt.scatter(nqe_total_charge, nqe_launches, label="NQE", alpha=0.7, c="blue")
    plt.scatter(hqe_total_charge, hqe_launches, label="HQE", alpha=0.7, c="orange", edgecolors='black')

    # Define cost functions for NQE and HQE
    def cost_function_nqe(m, c):
        return np.sum((nqe_launches - linear_model(nqe_total_charge, m, c))**2)

    def cost_function_hqe(m, c):
        return np.sum((hqe_launches - linear_model(hqe_total_charge, m, c))**2)

    # Fit NQE data
    minuit_nqe = Minuit(cost_function_nqe, m=1.0, c=0.0)
    minuit_nqe.errordef = Minuit.LEAST_SQUARES
    minuit_nqe.migrad()  # Perform the fit
    slope_nqe, intercept_nqe = minuit_nqe.values["m"], minuit_nqe.values["c"]
    slope_nqe_err, intercept_nqe_err = minuit_nqe.errors["m"], minuit_nqe.errors["c"]

    # Fit HQE data
    minuit_hqe = Minuit(cost_function_hqe, m=1.0, c=0.0)
    minuit_hqe.errordef = Minuit.LEAST_SQUARES
    minuit_hqe.migrad()  # Perform the fit
    slope_hqe, intercept_hqe = minuit_hqe.values["m"], minuit_hqe.values["c"]
    slope_hqe_err, intercept_hqe_err = minuit_hqe.errors["m"], minuit_hqe.errors["c"]

    # Plot fitted lines
    x_line_nqe = np.linspace(nqe_total_charge.min(), nqe_total_charge.max(), 500)
    y_line_nqe = linear_model(x_line_nqe, slope_nqe, intercept_nqe)
    plt.plot(
        x_line_nqe, y_line_nqe, label=f"NQE Fit: Slope = {slope_nqe:.2f} ± {slope_nqe_err:.2f}", color="blue", linestyle="--"
    )

    x_line_hqe = np.linspace(hqe_total_charge.min(), hqe_total_charge.max(), 500)
    y_line_hqe = linear_model(x_line_hqe, slope_hqe, intercept_hqe)
    plt.plot(
        x_line_hqe, y_line_hqe, label=f"HQE Fit: Slope = {slope_hqe:.2f} ± {slope_hqe_err:.2f}", color="orange", linestyle="--"
    )

    plt.xlabel("Total Charge")
    plt.ylabel("Launches")
    plt.title("Launches vs Total Charge with Minuit Fits")
    plt.legend()
    plt.grid()
    pdf.savefig()
    plt.close()

    print(f"NQE Fit: Slope = {slope_nqe:.2f} ± {slope_nqe_err:.2f}, Intercept = {intercept_nqe:.2f} ± {intercept_nqe_err:.2f}")
    print(f"HQE Fit: Slope = {slope_hqe:.2f} ± {slope_hqe_err:.2f}, Intercept = {intercept_hqe:.2f} ± {intercept_hqe_err:.2f}")
    
       # Plot launches scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        nqe_positions['x_jitter'], nqe_positions['y_jitter'],
        c=nqe_launches, cmap="viridis", s=40, label='NQE', vmin=launches_min, vmax=launches_max
    )
    plt.scatter(
        hqe_positions['x_jitter'], hqe_positions['y_jitter'],
        c=hqe_launches, cmap="viridis", s=40, edgecolors='red', label='HQE', facecolors='none', vmin=launches_min, vmax=launches_max
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
    plt.hist(nqe_launches, bins=20, alpha=0.6, label="NQE", color='blue', edgecolor='black')
    plt.hist(hqe_launches, bins=20, alpha=0.6, label="HQE", color='orange', edgecolor='black')
    plt.xlabel("Number of Events (Launches)")
    plt.ylabel("Number of DOMs")
    plt.title("Distribution of Events DOMs Participate In")
    plt.legend()
    plt.grid()
    pdf.savefig()
    plt.close()
    print("Saved histogram plot: launches_histogram.pdf")
    
    # Split NQE into three groups and HQE into one group
    nqe_group1 = (nqe_expected_hits < 100)
    nqe_group2 = (100 <= nqe_expected_hits) & (nqe_expected_hits < 300)
    nqe_group3 = (300 <= nqe_expected_hits) & (nqe_expected_hits < 600)
    hqe_group = hqe_expected_hits > 0  # All HQE DOMs

    # Prepare data for each group
    groups = [
        (nqe_expected_hits[nqe_group1], nqe_launches[nqe_group1], "NQE < 100", "blue"),
        (nqe_expected_hits[nqe_group2], nqe_launches[nqe_group2], "100 ≤ NQE < 300", "green"),
        (nqe_expected_hits[nqe_group3], nqe_launches[nqe_group3], "300 ≤ NQE < 600", "purple"),
        (hqe_expected_hits[hqe_group], hqe_launches[hqe_group], "HQE", "orange")
    ]
    def group_cost_function(m, c, x, y):
        return np.sum((y - linear_model(x, m, c))**2)
    
    fits = []
    for x, y, label, color in groups:
        def cost_for_groyp(m,c):
            return group_cost_function(m,c,x,y)
        # Fit the group
        minuit = Minuit(cost_for_groyp, m=1.0, c=0.0)
        minuit.errordef = Minuit.LEAST_SQUARES
        minuit.migrad()
        m, c = minuit.values["m"], minuit.values["c"]
        print(f"Fit for {label}: Slope = {m:.2f}, Intercept = {c:.2f}")
        m_err, c_err = minuit.errors["m"], minuit.errors["c"]
        fits.append((m, c, m_err, c_err, label, color))

    
    
    plt.figure(figsize=(10, 8))
    plt.scatter(nqe_expected_hits, nqe_launches, label="NQE", alpha=0.7, c="blue")
    plt.scatter(hqe_expected_hits, hqe_launches, label="HQE", alpha=0.7, c="orange", edgecolors='black')
    # Plot fitted lines
    for (x, _, label, color), (m, c, m_err, c_err, _, _) in zip(groups, fits):
        # Define the range of x values for this group
        if x.size == 0:
            continue
        x_min, x_max = x.min(), x.max()
        x_group = np.linspace(x_min, x_max, 100)  # Generate x values within the group's range

        # Calculate the corresponding y values for the fitted line
        y_line = linear_model(x_group, m, c)

        # Plot the line segment for this group
        plt.plot(x_group, y_line, label=f"{label}: Slope = {m:.2f} ± {m_err:.2f}", color=color, linestyle="--")

    plt.xlabel("Expected Hits")
    plt.ylabel("Launches")
    plt.title("Launches vs Expected Hits")
    plt.legend()
    plt.grid()
    pdf.savefig()
    plt.close()
    print("Saved scatter plot: launches_vs_expected_hits.pdf")
    
    # Plot showing RIDE for NQE and HQE with RIDE values on y-axis and x-position on x-axis
    nqe_expected_hits[nqe_expected_hits == 0] = 1  # Avoid division by zero
    hqe_expected_hits[hqe_expected_hits == 0] = 1  # Avoid division by zero
    monitor_old = np.median(nqe_total_charge / nqe_expected_hits)
    
    monitor = calculate_monitor_inside_polygon(pos_array_NQE, nqe_total_charge, nqe_expected_hits, polygon_coords)

    plt.figure(figsize=(10, 8))
    plt.scatter(nqe_positions['dom_x'], (nqe_total_charge / nqe_expected_hits) / monitor, label="NQE", alpha=0.7, c="blue")
    plt.scatter(hqe_positions['dom_x'], (hqe_total_charge / hqe_expected_hits) / monitor, label="HQE", alpha=0.7, c="orange", edgecolors='black')
    
    plt.xlabel("DOM x-coordinate")
    plt.ylabel("RIDE (Monitored to NQE)")
    plt.title("RIDE vs DOM x-coordinate")
    plt.legend()
    plt.grid()
    pdf.savefig()
    print("Saved scatter plot: ride_vs_dom_x.pdf")
    
    
    plt.figure(figsize=(10, 8))
    plt.scatter(nqe_positions['dom_x'], (nqe_total_charge / nqe_expected_hits) / monitor_old, label="NQE", alpha=0.7, c="blue")
    plt.scatter(hqe_positions['dom_x'], (hqe_total_charge / hqe_expected_hits) / monitor_old, label="HQE", alpha=0.7, c="orange", edgecolors='black')
    
    plt.xlabel("DOM x-coordinate")
    plt.ylabel("RIDE Old (Monitored to NQE)")
    plt.title("RIDE Old vs DOM x-coordinate")
    plt.legend()
    plt.grid()
    pdf.savefig()
    print("Saved scatter plot: ride_vs_dom_x.pdf")
    
    # Scatter plot showing 
    # Calculate charge fractions
    
    
    
    nqe_launche_charge_fraction = np.where(nqe_launches > 0, nqe_total_charge / nqe_launches, 0)
    hqe_launce_charge_fraction = np.where(hqe_launches > 0, hqe_total_charge / hqe_launches, 0)
    
    charge_launche_fraction_min = min(nqe_launche_charge_fraction.min(), hqe_launce_charge_fraction.min())
    charge_launche_fraction_max = max(nqe_launche_charge_fraction.max(), hqe_launce_charge_fraction.max())
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        nqe_positions['x_jitter'], nqe_positions['y_jitter'],
        c=nqe_launche_charge_fraction, cmap="viridis", s=40, label='NQE', vmin=charge_launche_fraction_min, vmax=charge_launche_fraction_max
    )
    plt.scatter(
        hqe_positions['x_jitter'], hqe_positions['y_jitter'],
        c=hqe_launce_charge_fraction, cmap="viridis", s=40, edgecolors='red', label='HQE', facecolors='none', vmin=charge_launche_fraction_min, vmax=charge_launche_fraction_max
    )
    plt.colorbar(scatter, label="Charge Fraction (Σ(Total charge) / Σ(launches))")
    plt.xlabel("DOM x-coordinate (jitter applied to HQE)")
    plt.ylabel("DOM y-coordinate (jitter applied to HQE)")
    plt.title("DOM Total Charge Scatter Plot (NQE & HQE)")
    plt.legend()
    pdf.savefig()
    plt.close()
    
    # Make a plot showing efficiency ratio for NQE and HQE DOMs (Launches / Expected Hits) as a function of x-coordinate
    nqe_efficiency_ratio = nqe_launches / nqe_expected_hits
    hqe_efficiency_ratio = hqe_launches / hqe_expected_hits
    
    plt.figure(figsize=(10, 8))
    plt.scatter(nqe_positions['dom_x'], nqe_efficiency_ratio, label="NQE", alpha=0.7, c="blue")
    plt.scatter(hqe_positions['dom_x'], hqe_efficiency_ratio, label="HQE", alpha=0.7, c="orange", edgecolors='black')
    
    plt.xlabel("DOM x-coordinate")
    plt.ylabel("Efficiency Ratio (Launches / Expected Hits)")
    plt.title("Efficiency Ratio vs DOM x-coordinate")
    plt.legend()
    plt.grid()
    pdf.savefig()
    plt.close()
    
    
    

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
    plt.xlabel("String")
    plt.ylabel("DOM Number")
    plt.title("DOM Participation Heatmap")
    pdf.savefig()
    plt.close()

def plot_efficiency_ratio(launches_nqe, expected_hits_nqe, launches_hqe, expected_hits_hqe, pdf):
    """
    Plot the efficiency ratio (Launches / Expected Hits) for NQE and HQE DOMs.
    """
    ratio_nqe = launches_nqe / expected_hits_nqe
    ratio_hqe = launches_hqe / expected_hits_hqe

    plt.figure(figsize=(10, 6))
    plt.hist(ratio_nqe, bins=20, alpha=0.6, label="NQE (RDE=1.0)", color='blue', edgecolor='black')
    plt.hist(ratio_hqe, bins=20, alpha=0.6, label="HQE (RDE=1.35)", color='orange', edgecolor='black')
    plt.xlabel("Efficiency Ratio (Launches / Expected Hits)")
    plt.ylabel("Counts")
    plt.title("Efficiency Ratio Comparison: NQE vs HQE")
    plt.legend()
    plt.grid()
    pdf.savefig()
    plt.close()


def plot_muon_direction_heatmap(df_truth, pulses_df, pdf, disk_depth, azimuth_bins=30, zenith_bins=30):
    """
    Generate a heatmap showing the density of muon directions (azimuth vs zenith)
    for events that trigger at least one DOM in the specified disk depth.
    """
    # Identify events with at least one DOM triggered in the specified disk
    events_in_disk = pulses_df[pulses_df['dom_z'] == disk_depth]['event_no'].unique()
    muons_in_disk = df_truth[df_truth['event_no'].isin(events_in_disk)]

    # Extract azimuth and zenith angles
    azimuths = muons_in_disk['azimuth'].values  # Azimuthal angles (0 to 2π)
    zeniths = muons_in_disk['zenith'].values    # Zenith angles (0 to π)

    # Create a 2D histogram (heatmap)
    heatmap, xedges, yedges = np.histogram2d(
        azimuths, zeniths, bins=[azimuth_bins, zenith_bins], range=[[0, 2 * np.pi], [0, np.pi]]
    )

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(
        heatmap.T, origin='lower', aspect='auto', 
        extent=[0, 2 * np.pi, 0, np.pi], cmap='viridis'
    )
    plt.colorbar(label="Muon Count")
    plt.xlabel("Azimuthal Angle (φ) [radians]")
    plt.ylabel("Zenith Angle (θ) [radians]")
    plt.title(f"Muon Direction Heatmap Triggering DOMs at Depth {disk_depth} m")

    # Overlay grid lines for better visualization
    plt.xticks(np.linspace(0, 2 * np.pi, 9), labels=["0", "π/4", "π/2", "3π/4", "π", "5π/4", "3π/2", "7π/4", "2π"])
    plt.yticks(np.linspace(0, np.pi, 7), labels=["0", "π/6", "π/3", "π/2", "2π/3", "5π/6", "π"])
    plt.grid(color='white', linestyle='--', alpha=0.5)

    # Save to PDF
    pdf.savefig()
    plt.close()


# -------------------------------
# Main Function
# -------------------------------
def main():
    file_path = '/groups/icecube/simon/GNN/workspace/filtered.db'
    con = sqlite3.connect(file_path)
    df_truth = pd.read_sql_query("SELECT * FROM truth", con)
    pulses_df = pd.read_sql_query("SELECT * FROM SplitInIcePulsesSRT", con)
    con.close()

    df_truth = df_truth
    filtered_events = df_truth['event_no'].unique()
    pulses_df = pulses_df[pulses_df['event_no'].isin(filtered_events)]

    # Filter DOMs by depth and strings
    all_depth_bins = np.arange(-500, 0, 10)
    pulses_df['depth_bin'] = pd.cut(pulses_df['dom_z'], bins=all_depth_bins, labels=all_depth_bins[:-1])
    pulses_df = pulses_df[pulses_df['string'] < 87.0]

    # Define focus depth bins and distance ranges
    focus_depth_bins = [-200, -250, -310, -320]
    distance_ranges = [(5, 40), (5, 100), (5, 200)]
    

    for focus_depth_bin in focus_depth_bins:
        for distance_range in distance_ranges:
            print(f"Processing focus_depth_bin: {focus_depth_bin}, distance_range: {distance_range}")

            # Filter DOMs based on depth_bin and rde
            doms_in_range = pulses_df[(pulses_df['depth_bin'] == focus_depth_bin)]

            # Separate NQE and HQE DOMs
            filtered_NQE = doms_in_range[(doms_in_range['rde'] == 1.0)]
            filtered_HQE = doms_in_range[(doms_in_range['rde'] == 1.35)]

            # Combine NQE and HQE DOM positions into a single array
            pos_array_combined = pd.concat([filtered_NQE, filtered_HQE])[['dom_x', 'dom_y', 'dom_z']].drop_duplicates().to_numpy()

            # Map combined DOM positions to their respective groups (NQE or HQE)
            nqe_positions_set = set(filtered_NQE[['dom_x', 'dom_y', 'dom_z']].apply(tuple, axis=1).tolist())
            group_map = {tuple(pos): "NQE" if tuple(pos) in nqe_positions_set else "HQE" for pos in pos_array_combined}

            # Split events into chunks for parallel processing
            event_chunks = np.array_split(df_truth, 8)
            
            # Parallel processing for all DOMs
            with ProcessPoolExecutor(max_workers=8) as executor:
                combined_results = list(executor.map(process_chunk, event_chunks, [pulses_df] * 8, [pos_array_combined] * 8, [distance_range] * 8))

            # Aggregate results
            expected_hits_combined = np.sum([result[0] for result in combined_results], axis=0)
            total_charge_combined = np.sum([result[1] for result in combined_results], axis=0)
            launches_combined = np.sum([result[2] for result in combined_results], axis=0)
            combined_dom_event_map = np.concatenate([result[3] for result in combined_results])

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
            pdf_filename = f"combined_plots_{abs(focus_depth_bin)}_{distance_range[0]}_{distance_range[1]}.pdf"
            polygon_coords = [(-510,-110),(-300,405),(210,520),(520,130),(300,-390),(-220,-490)]
            # Generate all plots into the PDF
            with PdfPages(pdf_filename) as pdf:
                plot_charge_fraction_and_launches(
            pos_array_NQE, pos_array_HQE, expected_hits_NQE, total_charge_NQE, launches_NQE,
            expected_hits_HQE, total_charge_HQE, launches_HQE, polygon_coords, pdf
            )
                plot_charge_distribution_by_DOM(doms_in_range, pdf)
                plot_dom_participation_heatmap(pulses_df, pdf)
                plot_efficiency_ratio(launches_NQE, expected_hits_NQE, launches_HQE, expected_hits_HQE, pdf)
                
            print(f"Saved PDF: {pdf_filename}")
            
if __name__ == "__main__":
    main()
