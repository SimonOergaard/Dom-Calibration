import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import os
import matplotlib.colors as colors
from concurrent.futures import ProcessPoolExecutor
import seaborn as sns

def classify_dom_strings(x, y):
    return 79 if x < 0 else 80

def aggregate_charge_dict(df, key_cols, value_col):
    keys = df[key_cols].to_numpy()
    unique_keys, inv = np.unique(keys, axis=0, return_inverse=True)
    sums = np.bincount(inv, weights=df[value_col].to_numpy())
    return {tuple(unique_keys[i]): sums[i] for i in range(len(unique_keys))}

def preprocess_pulses(pulses_np, pulse_event_col):
    unique_events, inverse = np.unique(pulses_np[:, pulse_event_col], return_inverse=True)
    sorted_indices = np.argsort(inverse)
    sorted_pulses = pulses_np[sorted_indices]
    sorted_inverse = inverse[sorted_indices]
    event_start = np.searchsorted(sorted_inverse, np.arange(len(unique_events)))
    event_end = np.append(event_start[1:], len(sorted_inverse))
    return {unique_events[i]: sorted_pulses[event_start[i]:event_end[i]] for i in range(len(unique_events))}

def calculate_minimum_distance_to_track(pos_array, true_x, true_y, true_z, true_zenith, true_azimuth, track_length):
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
    return np.linalg.norm(pos_array[:, :3] - closest_points, axis=1)

def calculate_expected_arrival_time(x0, y0, z0, zenith, azimuth, pos_array, dom_string, dom_number, 
                                    track_start_time=0, c=0.2998, n=1.33):
    dom_mask = (pos_array[:, 3] == dom_string) & (pos_array[:, 4] == dom_number)
    if not np.any(dom_mask):
        return np.inf
    dom_pos = pos_array[dom_mask][0, :3]

    v = np.array([
        np.sin(zenith) * np.cos(azimuth),
        np.sin(zenith) * np.sin(azimuth),
        np.cos(zenith)
    ])
    
    r0 = np.array([x0, y0, z0])
    r_rel = dom_pos - r0
    proj_length = np.dot(r_rel, v)
    r_para = proj_length * v
    r_perp = r_rel - r_para
    d_perp = np.linalg.norm(r_perp)

    theta_c = np.arccos(1 / n)
    d_track = d_perp / np.tan(theta_c)
    emission_point = r0 + v * d_track
    d_mu = d_track
    d_gamma = np.linalg.norm(dom_pos - emission_point)

    t_mu = d_mu / c
    t_gamma = n * d_gamma / c
    return track_start_time + t_mu + t_gamma

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

def collect_distance_vs_delta_t(chunk, pulses_df_np, pos_array, pulse_event_col, col_string, col_dom_number, col_dom_time, col_rde,
                                distance_lower, upper_bounds):
    event_map = preprocess_pulses(pulses_df_np, pulse_event_col)
    records = []

    for event in chunk.itertuples(index=False):
        event_no = event.event_no
        if event_no not in event_map:
            continue
        
        t0 = compute_t0_from_earliest_photon(
            pulse_rows=event_map[event_no],
            pos_array=pos_array,
            event=event,
            col_string=col_string,
            col_dom_number=col_dom_number,
            col_dom_time=col_dom_time
        )
        if t0 is None:
            continue

        for row in event_map[event_no]:
            dom_string = int(row[col_string])
            dom_number = int(row[col_dom_number])
            dom_time = row[col_dom_time]
            rde = row[col_rde]	
            dom_mask = (pos_array[:, 3] == dom_string) & (pos_array[:, 4] == dom_number)
            if not np.any(dom_mask):
                continue

            dom_pos = pos_array[dom_mask][0, :3]
            dist = calculate_minimum_distance_to_track(
                np.array([dom_pos]),
                event.position_x, event.position_y, event.position_z,
                event.zenith, event.azimuth, event.track_length
            )[0]

            expected_time = calculate_expected_arrival_time(
                event.position_x, event.position_y, event.position_z,
                event.zenith, event.azimuth,
                pos_array,
                dom_string, dom_number,
                track_start_time=0
            )

            delta_t = (dom_time - t0) - expected_time
            records.append((dist, delta_t, rde))
            print(f"rde={rde:.2f} | distance={dist:.1f} | dom_time={dom_time:.2f} | t0={t0:.2f} | expected={expected_time:.2f} | delta_t={delta_t:.2f} ns")
            
    return records

def collect_distance_vs_delta_t_wrapper(args):
    return collect_distance_vs_delta_t(*args)

def plot_ridge_delta_t_vs_distance(distances, deltas, output_dir, filename="ridge_delta_t_vs_distance.png"):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.DataFrame({"distance": distances, "delta_t": deltas})
    df = df[(df["delta_t"] > -200) & (df["delta_t"] < 800)]
    df = df.dropna(subset=["distance", "delta_t"])
    df["bin"] = pd.cut(df["distance"], bins=np.arange(0, 160, 20), include_lowest=True)
    df = df.dropna(subset=["bin"])
    # Filter to bins with enough samples

    bin_order = sorted(df["bin"].unique(), key=lambda x: x.left)

    g = sns.FacetGrid(
        df, row="bin", hue="bin",
        row_order=bin_order,  # Highest bin on top
        aspect=10, height=0.5, palette="viridis"
    )
    g.map(sns.kdeplot, "delta_t",
          bw_adjust=1.0, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "delta_t", clip_on=False, color="w", lw=2)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    for ax, label in zip(g.axes.flat, bin_order):
        ax.set_ylabel(str(label), rotation=0, ha='right', va='center', fontsize=9)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.set_yticks([])  # Hide KDE y-ticks

    g.set_titles("")
    g.despine(bottom=True, left=True)
    g.fig.subplots_adjust(hspace=0.5)

    plt.xlabel("Δt = (t_dom - t₀) - t_expected (ns)")
    plt.suptitle("Δt Distributions by DOM Distance Bin", fontsize=16)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    plt.figure(figsize=(8, 4))
    plt.hist(df["delta_t"], bins=100, range=(-200, 800), color='navy', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', label='Zero (expected match)')
    plt.xlabel("Δt = (t_dom - t₀) - t_expected (ns)")
    plt.ylabel("Count")
    plt.xlim(-200, 800)
    plt.title("Distribution of Δt values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "delta_t_histogram.png"), dpi=300)
    plt.close()

def plot_ridge_histogram_delta_t_by_distance(df, output_dir, filename="ridge_histogram_delta_t_by_distance.png"):
    import matplotlib.pyplot as plt

    # Filter and bin
    df = df[(df["delta_t"] > -200) & (df["delta_t"] < 800)]
    df = df.dropna(subset=["distance", "delta_t"])
    df["bin"] = pd.cut(df["distance"], bins=np.arange(0, 160, 10), include_lowest=True)

    df = df.dropna(subset=["bin"])
    bin_order = sorted(df["bin"].unique(), key=lambda x: x.left)
    
    fig, ax = plt.subplots(figsize=(10, len(bin_order) * 0.4))

    # Plot a histogram per bin as a ridge (vertically offset)
    for i, b in enumerate(bin_order):
        subset = df[df["bin"] == b]["delta_t"]
        hist, edges = np.histogram(subset, bins=np.linspace(-200, 800, 100))
        hist = hist / hist.max()  # normalize
        centers = (edges[:-1] + edges[1:]) / 2
        ax.fill_between(centers, i, i + hist, step="mid", alpha=0.8)

    ax.set_yticks(np.arange(len(bin_order)) + 0.5)
    ax.set_yticklabels([f"{b.left:.0f}–{b.right:.0f} m" for b in bin_order])
    ax.set_xlabel("Δt = (t_dom - t₀) - t_expected (ns)")
    ax.set_ylabel("Distance Bin")
    ax.set_title("Δt Histogram Ridge Plot by DOM Distance Bin")
    ax.set_xlim(-200, 800)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def plot_delta_t_vs_rde(distances, deltas, rdes, output_dir, filename="delta_t_vs_rde.png"):
    # Create DataFrame
    df = pd.DataFrame({
        "delta_t": deltas,
        "rde": rdes
    })

    # Apply reasonable delta_t limits to remove outliers
    df = df[(df["delta_t"] > -200) & (df["delta_t"] < 800)]
    df = df.dropna(subset=["delta_t", "rde"])

    # Convert rde to string category for labeling
    df["rde"] = df["rde"].astype(str)

    # Initialize the plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="rde", y="delta_t", inner="box", palette="Set2")

    # Labels and style
    plt.xlabel("DOM RDE")
    plt.ylabel("Δt = (t_dom - t₀) - t_expected (ns)")
    plt.title("Δt Distributions by DOM RDE Value")
    plt.grid(True, axis="y")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def plot_distances(distances,output_dir, filename="distances.png"):
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=100, color='blue', alpha=0.7)
    plt.xlabel('Distance (m)')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution')
    plt.grid()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
def main():
    file_path = '/groups/icecube/simon/GNN/workspace/Scripts/filtered_all_big_data.db'
    con = sqlite3.connect(file_path)
    df_truth = pd.read_sql_query("SELECT * FROM truth", con)
    pulses_df = pd.read_sql_query(
        "SELECT dom_x, dom_y, dom_z, charge, event_no, string, dom_time, rde, dom_number FROM SplitInIcePulsesSRT",
        con
    )
    con.close()

    pulses_df = pulses_df[
        pulses_df['string'].isin([80, 81, 82, 83, 84, 85])# & (pulses_df['dom_number'] < 11)
    ]
    pos_array = pulses_df[['dom_x','dom_y','dom_z','string','dom_number']].drop_duplicates().to_numpy()
    event_chunks = np.array_split(df_truth, 12)

    bin_edges = [(lo, lo + 5) for lo in range(10, 160, 20)]
    output_dir = '/groups/icecube/simon/GNN/workspace/Plots/'

    pulses_df_np = pulses_df.to_numpy()
    pulse_event_col = pulses_df.columns.get_loc("event_no")
    col_string = pulses_df.columns.get_loc("string")
    col_dom_number = pulses_df.columns.get_loc("dom_number")
    col_dom_time = pulses_df.columns.get_loc("dom_time")
    col_rde = pulses_df.columns.get_loc("rde")

    all_records = []
    with ProcessPoolExecutor(max_workers=12) as executor:
        args_list = [
            (chunk, pulses_df_np, pos_array, pulse_event_col,
             col_string, col_dom_number, col_dom_time, col_rde,
             bin_edges[0][0], [b[1] for b in bin_edges])
            for chunk in event_chunks
        ]
        for result in tqdm(executor.map(collect_distance_vs_delta_t_wrapper, args_list), total=len(args_list)):
            all_records.extend(result)

    distances, deltas, rdes = zip(*all_records)
    plot_ridge_delta_t_vs_distance(distances, deltas, output_dir)
    plot_ridge_histogram_delta_t_by_distance(pd.DataFrame({"distance": distances, "delta_t": deltas}), output_dir)
    plot_delta_t_vs_rde(distances, deltas, rdes, output_dir)
    #plot_distances(distances, output_dir)
if __name__ == "__main__":
    main()
