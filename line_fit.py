import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import matplotlib.colors as colors

def weighted_pca(points, weights):
    """
    Perform a weighted PCA on a set of 3D points.
    Returns the weighted mean (as a point) and the principal component (direction).
    """
    wm = np.average(points, axis=0, weights=weights)
    centered = points - wm
    # Normalize weights
    w_norm = weights / np.sum(weights)
    cov = np.dot((centered * w_norm[:, np.newaxis]).T, centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Principal component corresponds to the largest eigenvalue
    idx = np.argmax(eigvals)
    direction = eigvecs[:, idx]
    return wm, direction

def make_cherenkov_cone(
    apex,            # 3D apex coordinate [x, y, z]
    direction,       # 3D direction vector (will be normalized)
    speed,           # photon speed in ice, e.g. c/n
    cherenkov_angle, # half-angle of the cone in radians (e.g., ~41 deg in ice)
    time_val,        # time in *seconds* at which we draw the cone
    resolution=32
):
    """
    Generate (X, Y, Z) arrays for a Cherenkov cone in 3D at a specific time 'time_val'.

    The cone extends from 'apex' (s=0) to the ring at s=1, forming a truncated cone.

    Returns:
        X, Y, Z : 2D arrays of shape (resolution+1, resolution+1)
                  that can be passed to ax.plot_surface(...).
                  If time_val <= 0 or the cone is degenerate, returns (None, None, None).
    """
    apex = np.asarray(apex, dtype=float)
    direction = np.asarray(direction, dtype=float)
    norm_dir = np.linalg.norm(direction)
    if norm_dir < 1e-12:
        print("Cherenkov cone: direction vector is near zero-length.")
        return None, None, None
    direction /= norm_dir  # normalize

    # Distance along the track axis at time_val
    length = speed * time_val * np.cos(cherenkov_angle)
    # Radius of the cone at time_val
    radius = speed * time_val * np.sin(cherenkov_angle)

    if time_val <= 0 or length < 0 or radius < 0:
        # If time is negative or zero, no cone to draw
        return None, None, None

    # Build orthonormal vectors {direction, n1, n2}
    def find_perp(vec):
        # Return a vector not parallel to 'vec'
        if abs(vec[0]) < abs(vec[1]):
            return np.array([1.0, 0.0, 0.0])
        else:
            return np.array([0.0, 1.0, 0.0])

    perp = find_perp(direction)
    n1 = np.cross(direction, perp)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(direction, n1)

    # Parameter s in [0..1], theta in [0..2π]
    s_vals = np.linspace(0, 1, resolution+1)
    theta_vals = np.linspace(0, 2*np.pi, resolution+1)
    S, Theta = np.meshgrid(s_vals, theta_vals, indexing='ij')

    # Center line from apex to apex + length*direction
    center_line = apex[None,None,:] + (S[...,None]*length)*direction[None,None,:]

    # radius at each s => radius*s
    r_s = radius * S
    cross = (r_s[...,None]*np.cos(Theta[...,None])*n1
             + r_s[...,None]*np.sin(Theta[...,None])*n2)

    coords = center_line + cross
    X = coords[..., 0]
    Y = coords[..., 1]
    Z = coords[..., 2]
    return X, Y, Z


def create_mp4_for_event(event_df, event_no, output_dir,
                         time_bin=5.0, framerate=95, size_scale=2000.0):
    """
    Create an MP4 animation for one event using 3D data, where:
      - Time is normalized to [0, 1].
      - Marker diameter is proportional to each DOM's partial-summed charge
        as a fraction of the *total* event charge.
      - The line is computed from the total-summed charges over the entire event
        and is gradually revealed in time.
    """
    # Make a copy so we don't modify the original DataFrame
    event_df = event_df.copy()
    
    # 1) Normalize time:
    #    Subtract the minimum dom_time to start at 0,
    #    then divide by the maximum to get [0..1].
    event_df['dom_time_centered'] = event_df['dom_time'] - event_df['dom_time'].min()
    
    max_time = event_df['dom_time_centered'].max()
    if max_time > 0:
        event_df['dom_time_norm'] = event_df['dom_time_centered'] / max_time
    else:
        # Edge case: if all times are the same, just set them to 0
        event_df['dom_time_norm'] = 0.0
    
    
    min_x, max_x = event_df['dom_x'].min(), event_df['dom_x'].max()
    min_y, max_y = event_df['dom_y'].min(), event_df['dom_y'].max()
    min_z, max_z = event_df['dom_z'].min(), event_df['dom_z'].max()
    
    # Sort by normalized time
    event_df = event_df.sort_values('dom_time_norm')
    
    # 2) Compute the total charge of the entire event (for sizing).
    total_event_charge = event_df['charge'].sum()
    
    # 3) For the line fit, group by DOM to get total-summed charge per DOM
    #    so the PCA sees each DOM only once with its total charge.
    event_summed = event_df.groupby(['dom_x','dom_y','dom_z'], as_index=False)['charge'].sum()
    points_all = event_summed[['dom_x', 'dom_y', 'dom_z']].values
    weights_all = event_summed['charge'].values
    
    # Weighted PCA for the entire event
    wm, direction = weighted_pca(points_all, weights_all)
    
    # Determine line extent
    projections = np.dot(points_all - wm, direction)
    proj_min, proj_max = projections.min(), projections.max()
    line_vals = np.linspace(proj_min, proj_max, 100)
    line_points = wm + np.outer(line_vals, direction)
    
    apex0 = line_points[0]
    line_length = np.linalg.norm(line_points[-1] - line_points[0])
    # 4) Prepare frames from time 0..1 in increments of time_bin/max_time
    #    (Because the original time was normalized to [0..1], we scale accordingly.)
    if max_time == 0:
        # Degenerate case: everything at one time
        time_frames = [0.0]
    else:
        # time_bin is in original units; fraction is time_bin / max_time
        frame_step = time_bin / max_time
        # Go from 0..1 with that step
        time_frames = np.arange(0, 1.0 + frame_step, frame_step)
    
    # 4) Physical parameters
    c = 3e8
    n = 1.31
    cher_speed = c / n         # speed of light in ice
    muon_speed = 0.99*c        # e.g. 0.99 c
    angle = np.radians(41.0)   # Cherenkov angle

    direction_hat = direction / np.linalg.norm(direction)
    
    # Temporary directory to store frames
    temp_dir = os.path.join(output_dir, f"temp_event_{event_no}")
    os.makedirs(temp_dir, exist_ok=True)
    vmin, vmax = 0, 1
    
    for i, t in enumerate(time_frames):
        # 4a) Partial hits up to time t
        df_t = event_df[event_df['dom_time_norm'] <= t]
        
        # Group by DOM => partial-summed charge, plus average time for coloring
        df_t_summed = df_t.groupby(['dom_x','dom_y','dom_z'], as_index=False).agg({
            'charge': 'sum',
            'dom_time_centered': 'mean'  # average time among the partial hits
        })

        print(f"Event {event_no} frame {i:03d}: {len(df_t_summed)} DOMs at time {t:.1f}")

        # 4b) Determine fraction of the line to show
        if max_time > 0:
           df_t_summed['dom_time_norm'] = df_t_summed['dom_time_centered'] / max_time
        else:
            df_t_summed['dom_time_norm'] = 0.0
            
        # current_proj_max = proj_min + t*(proj_max - proj_min)
        
        # #Generate partial line points
        # partial_line_vals = np.linspace(proj_min, current_proj_max, 100)
        # partial_line_points = wm + np.outer(partial_line_vals, direction)
        
        real_time_s = (t * max_time) *1e-9
        dist_travelled = muon_speed * real_time_s
        
        param_vals = np.linspace(0, dist_travelled, 100)
        partial_line_points = apex0[None, :] + param_vals[:, None] * direction_hat[None, :]
        
        # 4c) Plot
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')

        # Marker size = partial-summed charge
        partial_total = df_t_summed['charge'].sum()
        if partial_total > 0:
            size_fraction = df_t_summed['charge'] / partial_total
            sizes = size_scale * size_fraction
        else:
            sizes = 50  # fallback

        # Marker color = average time, normalized to [0..1]
        # We'll pass 'time_norm' to c=...
        sc = ax.scatter(
            df_t_summed['dom_x'], 
            df_t_summed['dom_y'], 
            df_t_summed['dom_z'],
            c=df_t_summed['dom_time_norm'], 
            cmap='viridis',
            s=sizes,
            edgecolor='k',
            alpha=0.8, 
            zorder=2,
            vmin=vmin, vmax=vmax
        )

        # partial line
        ax.plot(
            partial_line_points[:,0],
            partial_line_points[:,1],
            partial_line_points[:,2],
            'r-', linewidth=2,
            zorder=3,
            label='Muon Track (best-fit)'
        )
        
        apex_t = apex0 + (muon_speed * real_time_s) * direction_hat
        
        Xc, Yc, Zc = make_cherenkov_cone(
            apex=apex_t,
            direction=-direction_hat,
            speed=cher_speed,
            cherenkov_angle=angle,
            time_val=real_time_s,
            resolution=32
        )
        if Xc is not None:
            ax.plot_surface(Xc, Yc, Zc, color='cyan', alpha=0.3, zorder=1)

        # Axis limits
        ax.set_xlim(-600, 600)
        ax.set_ylim(-600, 600)
        ax.set_zlim(-600, 600)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Event {event_no} at t={t:.1f} / {max_time:.1f}")
        cb = fig.colorbar(sc, ax=ax, label="Normalized Time")
        ax.legend()

        frame_filename = os.path.join(temp_dir, f"frame_{i:03d}.png")
        plt.savefig(frame_filename, dpi=150)
        plt.close(fig)

    # 5) Stitch frames into MP4
    output_file = os.path.join(output_dir, f"event_{event_no}_animation.mp4")
    ffmpeg_cmd = [
        "/groups/icecube/simon/bin/ffmpeg", "-y",
        "-framerate", str(framerate),
        "-i", os.path.join(temp_dir, "frame_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        output_file
    ] 
    try:
        result = subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("ffmpeg output:", result.stdout.decode())
        print("ffmpeg errors:", result.stderr.decode())
    except Exception as e:
        print("Error running ffmpeg:", e)

    # Optionally remove the temp directory
    import shutil
    shutil.rmtree(temp_dir)

    print(f"Saved event {event_no} animation to {output_file}")

def make_cylinder(p0, p1, radius=60.0, resolution=32):
    """
    Generate (X, Y, Z) arrays for a cylinder of given 'radius'
    around the line segment from p0 to p1 in 3D.

    Returns:
        X, Y, Z : 2D arrays of shape (resolution+1, resolution+1)
                  that can be passed to ax.plot_surface(X, Y, Z).
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v = p1 - p0
    length = np.linalg.norm(v)
    if length < 1e-12:
        # Degenerate case: p0 == p1
        raise ValueError("Cylinder axis has zero length (p0 == p1).")

    # Unit vector along cylinder axis
    vhat = v / length

    # Find two perpendicular vectors n1, n2 to define the cross-section plane
    def find_perp(vec):
        """Return a vector not parallel to 'vec'."""
        # If x-component is smaller, we pick x-axis to cross with, else y-axis
        if abs(vec[0]) < abs(vec[1]):
            return np.array([1.0, 0.0, 0.0])
        else:
            return np.array([0.0, 1.0, 0.0])

    perp = find_perp(vhat)
    n1 = np.cross(vhat, perp)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(vhat, n1)
    # Now vhat, n1, n2 is an orthonormal basis

    # Parameter ranges
    s_vals = np.linspace(0, 1, resolution+1)      # along the cylinder's length
    theta_vals = np.linspace(0, 2*np.pi, resolution+1)  # around the circumference

    # Make 2D grids: shape => (resolution+1, resolution+1)
    S, Theta = np.meshgrid(s_vals, theta_vals, indexing='ij')
    # S(i,j) goes in [0..1], Theta(i,j) in [0..2pi]

    # S is fraction of the axis length => actual "center-line" offset
    # shape (res+1, res+1)
    # We'll build a 3D array for the center line
    # center_line(i,j,:) = p0 + S(i,j)*length*vhat
    # Expand dims so we can broadcast
    center_line = p0[None, None, :] + (S[..., None] * length) * vhat[None, None, :]

    # For the circular cross-section
    # cross(i,j,:) = radius*cos(Theta(i,j))*n1 + radius*sin(Theta(i,j))*n2
    cosT = np.cos(Theta)
    sinT = np.sin(Theta)
    cross = radius * (cosT[..., None]*n1 + sinT[..., None]*n2)

    # Combine center line + cross => final shape (res+1, res+1, 3)
    coords = center_line + cross

    # Separate into X, Y, Z
    X = coords[..., 0]
    Y = coords[..., 1]
    Z = coords[..., 2]
    return X, Y, Z

def plot_event_with_fit(
    geometry_df,   # full detector geometry: columns [string_id, dom_x, dom_y, dom_z, ...]
    df,            # event data: columns [dom_x, dom_y, dom_z, charge, dom_time, ...]
    event_no, 
    output_dir, 
    size_scale=2000.0
):
    """
    Plots a 3D event display with:
      1) All strings from geometry_df (gray lines).
      2) Summed charges per DOM from df (so each DOM appears once).
         - Marker diameter proportional to fraction of total event charge.
         - Marker color by average normalized time [0..1].
      3) A weighted PCA fitted line (based on the DOMs' total-summed charges).
    Saves the figure as a PNG in output_dir.

    Parameters:
      geometry_df : pd.DataFrame
          Must contain at least: 'string_id', 'dom_x', 'dom_y', 'dom_z'
          (one row per DOM in the detector). 
      df          : pd.DataFrame
          Contains the *event hits*, with columns:
            'dom_x', 'dom_y', 'dom_z', 'charge', 'dom_time', ...
          Possibly multiple rows per DOM if it was hit multiple times.
      event_no    : (int or str) Identifier for the event (used in saved filename).
      output_dir  : (str) Directory to save the output PNG file.
      size_scale  : (float) Factor to multiply fraction-of-total-charge for marker size.
    """
    # Make a copy so we don't modify the original event data
    df = df.copy()

    # 1) Plot setup
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # 2) Plot all strings from geometry_df
    #    We assume each row in geometry_df is one DOM, so group by string_id, sort by z.
    for s_id in geometry_df['string'].unique():
        str_df = geometry_df[geometry_df['string'] == s_id].sort_values('dom_z')
        ax.plot(
            str_df['dom_x'], 
            str_df['dom_y'], 
            str_df['dom_z'], 
            color='gray', 
            linewidth=1, 
            alpha=0.5
        )

    # 3) Normalize time in the event DataFrame
    df['dom_time_centered'] = df['dom_time'] - df['dom_time'].min()
    max_time = df['dom_time_centered'].max()
    if max_time > 0:
        df['dom_time_norm'] = df['dom_time_centered'] / max_time
    else:
        # If all times are identical, just set them to 0
        df['dom_time_norm'] = 0.0

    # 4) Group the event hits by DOM to sum charges, get an average (or min/max) of normalized time
    #    We'll use the mean normalized time for coloring.
    grouped = df.groupby(['dom_x','dom_y','dom_z'], as_index=False).agg({
        'charge': 'sum',
        'dom_time_norm': 'mean'
    })

    # 5) Determine total event charge, for scaling marker sizes
    total_event_charge = grouped['charge'].sum()
    if total_event_charge > 0:
        size_fraction = grouped['charge'] / total_event_charge
        sizes = size_fraction * size_scale
    else:
        # fallback if total charge is 0
        sizes = 50

    # 6) Weighted PCA for the entire event, using total-summed charges
    points = grouped[['dom_x','dom_y','dom_z']].values
    weights = grouped['charge'].values
    wm, direction = weighted_pca(points, weights)

    # Determine extent along the principal direction
    projections = np.dot(points - wm, direction)
    t_min, t_max = projections.min(), projections.max()

    # Generate a set of points along the fit line
    line_t_vals = np.linspace(t_min, t_max, 100)
    line_points = wm + np.outer(line_t_vals, direction)

    # 7) Scatter the DOM hits
    #    Color by average normalized time, size by fraction of total event charge
    

    
    sc = ax.scatter(
        grouped['dom_x'], grouped['dom_y'], grouped['dom_z'],
        c=grouped['dom_time_norm'],  # color by mean normalized time
        s=sizes,
        cmap='viridis',
        alpha=0.8,
        edgecolor='k'
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label('Normalized Time')
    
    

    # 8) Plot the fitted line in red
    ax.plot(
        line_points[:,0], 
        line_points[:,1], 
        line_points[:,2],
        'r-', lw=2, label='Fitted Line'
    )
    #  Draw a cylinder along the fitted line with a radius of 60 m
    p0 = line_points[0]
    p1 = line_points[-1]

    Xcyl, Ycyl, Zcyl = make_cylinder(p0, p1, radius=60, resolution=32)
    ax.plot_surface(Xcyl, Ycyl, Zcyl, color='blue', alpha=0.2)
    
    # Axis labels, title, legend
    ax.set_xlabel('DOM X')
    ax.set_ylabel('DOM Y')
    ax.set_zlabel('DOM Z')
    ax.set_title(f'Event {event_no} (Summed Charges, Normalized Time)')
    ax.legend()

    # 9) Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"event_{event_no}_display.png")
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"Saved event {event_no} display to {output_file}")
    
def main():
    # Database file path (adjust as needed)
    db_path = "/groups/icecube/simon/GNN/workspace/Scripts/filtered_all.db"
    con = sqlite3.connect(db_path)
    # Assume the table is named 'SplitInIcePulsesSRT' and contains at least:
    # event_no, dom_x, dom_y, dom_z, charge, dom_time
    query = "SELECT event_no, dom_x, dom_y, dom_z, charge, dom_time, string, dom_number FROM SplitInIcePulsesSRT;"
    df = pd.read_sql_query(query, con)
    con.close()
    
    # Get the first 10 unique events.
    unique_events = df['event_no'].unique()[16:20]
    geometry_df = df[['string', 'dom_x', 'dom_y', 'dom_z']].drop_duplicates()
    output_dir = "/groups/icecube/simon/GNN/workspace/Plots/animation"
    os.makedirs(output_dir, exist_ok=True)
    
    for event_no in unique_events:
        event_df = df[df['event_no'] == event_no]
        print(f"Processing event {event_no}...")
        #create_mp4_for_event(event_df, event_no, output_dir, time_bin=5.0, framerate=95, size_scale=2000.0)
        plot_event_with_fit(geometry_df, event_df, event_no, output_dir, size_scale=2000.0)

if __name__ == "__main__":
    main()
