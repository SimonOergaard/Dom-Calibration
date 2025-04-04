import sqlite3
import pandas as pd
import numpy as np


def center_dom_time_by_event(dom_df, pulse_event_col="event_no", dom_time_col="dom_time"):
    """
    Center the DOM time at 0 for each event by subtracting the earliest time for each event.

    Parameters:
        dom_df (pd.DataFrame): DataFrame containing DOM pulse data.
        pulse_event_col (str): Column name for event numbers.
        dom_time_col (str): Column name for DOM time.

    Returns:
        pd.DataFrame: Updated DataFrame with centered `dom_time`.
    """
    # Group by event_no and subtract the minimum time within each event
    dom_df[dom_time_col] = dom_df.groupby(pulse_event_col)[dom_time_col].transform(lambda x: x - x.min())
    return dom_df


def preprocess_features(dom_df, detector_bounds):
    """
    Normalize DOM features based on detector bounds and center `dom_time` for each event.

    Parameters:
        dom_df (pd.DataFrame): DataFrame containing DOM pulse data.
        detector_bounds (dict): Dictionary with `min` and `max` bounds for DOM positions.

    Returns:
        Tuple of normalized positions, normalized dom_time, and normalized charge.
    """
    # Normalize positions
    normalized_positions = (dom_df[["dom_x", "dom_y", "dom_z"]] - detector_bounds["min"]) / (
        detector_bounds["max"] - detector_bounds["min"]
    )
    
    # Center dom_time for each event
    dom_df = center_dom_time_by_event(dom_df, pulse_event_col="event_no", dom_time_col="dom_time")
    
    # Normalize dom_time and charge
    normalized_dom_time = (dom_df["dom_time"] - dom_df["dom_time"].mean()) / (dom_df["dom_time"].std() + 1e-8)
    normalized_charge = (dom_df["charge"] - dom_df["charge"].mean()) / (dom_df["charge"].std() + 1e-8)
    
    # Update DataFrame with normalized features
    dom_df["dom_x"], dom_df["dom_y"], dom_df["dom_z"] = normalized_positions["dom_x"], normalized_positions["dom_y"], normalized_positions["dom_z"]
    dom_df["dom_time"] = normalized_dom_time
    dom_df["charge"] = normalized_charge

    return dom_df



def calculate_start_point(row):
    """Calculate the starting position of the muon using the stopping position, direction, and track length."""
    stop_point = row[["position_x", "position_y", "position_z"]].values
    direction_vector = [
        np.sin(row["zenith"]) * np.cos(row["azimuth"]),
        np.sin(row["zenith"]) * np.sin(row["azimuth"]),
        np.cos(row["zenith"]),
    ]
    start_point = stop_point + np.array(direction_vector) * row["track_length"]
    return start_point


def calculate_direction(row):
    """Calculate the direction vector of the muon using the starting and stopping positions."""
    stop_point = row[["position_x", "position_y", "position_z"]].values
    start_point = row[["start_position_x", "start_position_y", "start_position_z"]].values
    direction_vector = stop_point - start_point
    direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize
    return direction_vector


def preprocess_truth_table(df_truth):
    """Add calculated start positions and direction vectors to the truth table."""
    if "start_position_x" not in df_truth.columns:
        print("Calculating start positions...")
        start_points = df_truth.apply(calculate_start_point, axis=1)
        df_truth[["start_position_x", "start_position_y", "start_position_z"]] = pd.DataFrame(
            start_points.tolist(), index=df_truth.index
        )

    if "direction_x" not in df_truth.columns:
        print("Calculating direction vectors...")
        directions = df_truth.apply(calculate_direction, axis=1)
        df_truth[["direction_x", "direction_y", "direction_z"]] = pd.DataFrame(
            directions.tolist(), index=df_truth.index
        )

    return df_truth


def normalize_and_save(input_db_path, output_db_path, pulsemap, truth_table, detector_bounds):
    """Normalize data and save it to a new SQLite database."""
    # Connect to the input database
    conn = sqlite3.connect(input_db_path)

    # Load and preprocess the pulsemap table
    print("Processing pulsemap table...")
    dom_df = pd.read_sql(f"SELECT * FROM {pulsemap};", conn)
    
    # Pass the DOM data through preprocessing
    normalized_dom_df = preprocess_features(dom_df, detector_bounds)
    
    # Load and preprocess the truth table
    print("Processing truth table...")
    truth_df = pd.read_sql(f"SELECT * FROM {truth_table};", conn)
    truth_df = preprocess_truth_table(truth_df)

    # Close the input database connection
    conn.close()

    # Save the processed data to a new SQLite database
    print("Saving normalized data...")
    conn_out = sqlite3.connect(output_db_path)

    # Save the pulsemap and truth table
    normalized_dom_df.to_sql(pulsemap, conn_out, if_exists="replace", index=False)
    truth_df.to_sql(truth_table, conn_out, if_exists="replace", index=False)

    # Close the output database connection
    conn_out.close()
    print(f"Normalized dataset saved to {output_db_path}")



if __name__ == "__main__":
    # File paths and table names
    input_db_path = "/groups/icecube/simon/GNN/workspace/Scripts/filtered_all.db"
    output_db_path = "/groups/icecube/simon/GNN/workspace/Scripts/normalized_dataset.db"
    pulsemap = "SplitInIcePulsesSRT"
    truth_table = "truth"

    # Detector bounds
    detector_bounds = {
        "min": np.array([-500, -500, -500]),  # Replace with actual detector min bounds
        "max": np.array([500, 500, 500]),    # Replace with actual detector max bounds
    }

    normalize_and_save(input_db_path, output_db_path, pulsemap, truth_table, detector_bounds)
