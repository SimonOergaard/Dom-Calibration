import argparse
import sqlite3
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeUpgrade
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.task.reconstruction import StopPositionReconstruction, StartPositionReconstruction, DirectionReconstruction, DirectionReconstructionWithUncertainty
from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import SQLiteDataset, SQLiteDataset_new, worker_init_fn
from graphnet.training.loss_functions import LogCoshLoss, EuclideanDistanceLoss, vMFNegativeLogLikelihood, VonMisesFisher3DLoss, MSELoss, HuberLoss,CosineAngleLoss,CosineAngleLoss_2, DirectionalCosineLossWithKappa
from graphnet.training.callbacks import PiecewiseLinearLR
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from line_profiler import LineProfiler
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch

import torch.nn.functional as F
from torch.optim.adam import Adam
from torch import set_float32_matmul_precision
from torch.optim.lr_scheduler import OneCycleLR

set_float32_matmul_precision('medium')  # 'medium' for performance vs 'high' for precision

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train or predict using GNN.")
    parser.add_argument("mode", choices=["train", "predict"], help="Mode to run: train or predict.")
    parser.add_argument("--weights", choices=["best_weights", "last_weights"], default = None, help="Which weights to use for prediction or continuing training.")
    return parser.parse_args()


def ensure_index_on_event_no(db_path):
    """
    Ensures the SQLite database has an index on the event_no column for faster queries.
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_no ON SplitInIcePulsesSRT(event_no);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_no_truth ON truth(event_no);")
    connection.commit()
    connection.close()
    print("Index on 'event_no' ensured.")

def calculate_start_point(row):
    """
    Calculate the starting position of the muon using the stopping position, direction, and track length.
    """
    stop_point = row[["position_x", "position_y", "position_z"]].values
    direction_vector = [
        np.sin(row["zenith"]) * np.cos(row["azimuth"]),
        np.sin(row["zenith"]) * np.sin(row["azimuth"]),
        np.cos(row["zenith"]),
    ]
    start_point = stop_point + np.array(direction_vector) * row["track_length"]
    return start_point


def calculate_direction(row):
    """
    Calculate the direction vector of the muon using the starting and stopping positions.
    """
    stop_point = row[["position_x", "position_y", "position_z"]].values
    start_point = row[["start_position_x", "start_position_y", "start_position_z"]].values
    direction_vector = start_point - stop_point
    direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize
    return direction_vector

def preprocess_truth_table(db_path, truth_table):
    """
    Overwrites or adds starting position, direction vectors, and normalized position columns in the truth table.
    """
    connection = sqlite3.connect(db_path)

    # Load the truth table into a DataFrame
    df_truth = pd.read_sql(f"SELECT * FROM {truth_table};", connection)

    # Always overwrite start positions
    # print("Calculating start positions...")
    # start_points = df_truth.apply(calculate_start_point, axis=1)
    # df_truth[["start_position_x", "start_position_y", "start_position_z"]] = pd.DataFrame(
    #     start_points.tolist(), index=df_truth.index
    # )
    # print("Start positions updated successfully.")

    # Always overwrite normalised positions
    print("Calculating normalised positions...")
    df_truth["normalised_position_x"] = df_truth["position_x"] / 500.0
    df_truth["normalised_position_y"] = df_truth["position_y"] / 500.0
    df_truth["normalised_position_z"] = df_truth["position_z"] / 500.0
    print("Normalised positions updated successfully.")

    # Always overwrite direction vectors
    # print("Calculating direction vectors...")
    # directions = df_truth.apply(calculate_direction, axis=1)
    # df_truth[["direction_x", "direction_y", "direction_z"]] = pd.DataFrame(
    #     directions.tolist(), index=df_truth.index
    # )
    # print("Direction vectors updated successfully.")

    # Replace the table in the database with the updated DataFrame
    df_truth.to_sql(truth_table, connection, if_exists="replace", index=False)
    print(f"`{truth_table}` table updated with new calculated columns.")

    connection.close()



# def preprocess_truth_table(db_path, truth_table):
#     """
#     Adds starting position and direction vector columns to the truth table if they are missing.
#     """
#     connection = sqlite3.connect(db_path)

#     # Load the truth table into a DataFrame
#     df_truth = pd.read_sql(f"SELECT * FROM {truth_table};", connection)

#     # Check and calculate starting positions
#     if "start_position_x" not in df_truth.columns or \
#        "start_position_y" not in df_truth.columns or \
#        "start_position_z" not in df_truth.columns:
#         print("Calculating start positions...")
#         start_points = df_truth.apply(calculate_start_point, axis=1)
#         df_truth[["start_position_x", "start_position_y", "start_position_z"]] = pd.DataFrame(
#             start_points.tolist(), index=df_truth.index
#         )
#         print("Start positions added successfully.")
#     if "normalised_position_x" not in df_truth.columns or \
#         "normalised_position_y" not in df_truth.columns or \
#         "normalised_position_z" not in df_truth.columns:
#         print("Calculating normalised positions...")
#         df_truth["normalised_position_x"] = df_truth["position_x"] / 500.0
#         df_truth["normalised_position_y"] = df_truth["position_y"] / 500.0
#         df_truth["normalised_position_z"] = df_truth["position_z"] / 500.0
#         print("Normalised positions added successfully.")
#     # Check and calculate direction vectors
#     if "direction_x" not in df_truth.columns or \
#        "direction_y" not in df_truth.columns or \
#        "direction_z" not in df_truth.columns:
#         print("Calculating direction vectors...")
#         directions = df_truth.apply(calculate_direction, axis=1)
#         df_truth[["direction_x", "direction_y", "direction_z"]] = pd.DataFrame(
#             directions.tolist(), index=df_truth.index
#         )

#     # Replace the table in the database with the updated DataFrame
#     df_truth.to_sql(truth_table, connection, if_exists="replace", index=False)
#     print(f"`{truth_table}` table updated with starting positions and direction vectors.")
#     connection.close()


def compute_angles_between_vectors(true_directions, predicted_directions):
    """
    Compute the angle between true and predicted direction vectors.
    """
    # Normalize the vectors
    true_directions = true_directions / np.linalg.norm(true_directions, axis=1, keepdims=True)
    predicted_directions = predicted_directions / np.linalg.norm(predicted_directions, axis=1, keepdims=True)
    
    # Compute dot product
    cos_theta = np.sum(true_directions * predicted_directions, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Clip to avoid numerical issues

    # Compute angles in radians and convert to degrees
    angles = np.arccos(cos_theta) * (180 / np.pi)
    return angles

def plot_angle_distribution(true_directions, predicted_directions, output_dir):
    """
    Plot the distribution of angles between true and predicted direction vectors.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute angles
    angles = compute_angles_between_vectors(true_directions, predicted_directions)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(angles, bins=50, range=(0, 180), color="skyblue", alpha=0.7, edgecolor="black")
    plt.xlabel("Angle between True and Predicted Directions (degrees)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Angles Between True and Predicted Directions")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "angle_distribution.png"))
    plt.close()



def plot_stopping_predictions(true_positions, predicted_positions, output_dir):
    """
    Plots heatmaps of predicted vs true stopping positions (x, y, z).
    """
    os.makedirs(output_dir, exist_ok=True)

    labels = ["x", "y", "z"]
    for i, label in enumerate(labels):
        plt.figure(figsize=(8, 8))
        plt.hexbin(
            true_positions[:, i],
            predicted_positions[:, i],
            gridsize=50,  # Adjust grid size for resolution
            cmap="viridis",
            mincnt=1,
        )
        plt.colorbar(label="Density")
        plt.plot(
            [true_positions[:, i].min(), true_positions[:, i].max()],
            [true_positions[:, i].min(), true_positions[:, i].max()],
            color="red",
            linestyle="--",
            label="y = x",
        )
        plt.xlabel(f"True {label.upper()} Stopping Position")
        plt.ylabel(f"Predicted {label.upper()} Stopping Position")
        plt.title(f"Predicted vs True {label.upper()} Stopping Position")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, f"Euclidian_heatmap_pred_vs_true_stopping_{label}_1.png"))
        plt.close()


def plot_direction_predictions(true_directions, predicted_directions, output_dir):
    """
    Plots heatmaps of predicted vs true direction vector components (x, y, z).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Normalize the vectors
    true_directions = normalize_vectors(true_directions)
    predicted_directions = normalize_vectors(predicted_directions)

    labels = ["x", "y", "z"]
    for i, label in enumerate(labels):
        plt.figure(figsize=(8, 8))
        plt.hexbin(
            true_directions[:, i],
            predicted_directions[:, i],
            gridsize=50,  # Adjust grid size for resolution
            cmap="viridis",
            mincnt=1,
        )
        plt.colorbar(label="Density")
        plt.plot(
            [-1, 1],
            [-1, 1],
            color="red",
            linestyle="--",
            label="y = x",
        )
        
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.xlabel(f"True {label.upper()} Direction")
        plt.ylabel(f"Predicted {label.upper()} Direction")
        plt.title(f"Predicted vs True {label.upper()} Direction")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, f"heatmap_pred_vs_true_direction_{label}.png"))
        plt.close()

import matplotlib.pyplot as plt

def plot_zenith_azimuth_predictions(pred_zenith, pred_azimuth, true_zenith, true_azimuth, zenith_sigma, azimuth_sigma, output_dir):
    """
    Plots predicted vs. true zenith and azimuth angles with uncertainties.
    
    Args:
        pred_zenith (array): Predicted zenith angles.
        pred_azimuth (array): Predicted azimuth angles.
        true_zenith (array): True zenith angles.
        true_azimuth (array): True azimuth angles.
        zenith_sigma (array): Predicted uncertainty (std deviation) on zenith.
        azimuth_sigma (array): Predicted uncertainty (std deviation) on azimuth.
        output_dir (str): Path to save the plot.
    """

    plt.figure(figsize=(12, 5))

    # Zenith angle plot
    plt.subplot(1, 2, 1)
    plt.errorbar(true_zenith, pred_zenith, yerr=zenith_sigma, fmt="o", alpha=0.5, label="Predicted Zenith")
    plt.plot([true_zenith.min(), true_zenith.max()], [true_zenith.min(), true_zenith.max()], "--k", label="Perfect Prediction")
    plt.xlabel("True Zenith (rad)")
    plt.ylabel("Predicted Zenith (rad)")
    plt.title("Zenith Angle Reconstruction")
    plt.legend()
    plt.grid()

    # Azimuth angle plot
    plt.subplot(1, 2, 2)
    plt.errorbar(true_azimuth, pred_azimuth, yerr=azimuth_sigma, fmt="o", alpha=0.5, label="Predicted Azimuth")
    plt.plot([true_azimuth.min(), true_azimuth.max()], [true_azimuth.min(), true_azimuth.max()], "--k", label="Perfect Prediction")
    plt.xlabel("True Azimuth (rad)")
    plt.ylabel("Predicted Azimuth (rad)")
    plt.title("Azimuth Angle Reconstruction")
    plt.legend()
    plt.grid()

    # Save plot
    plt.tight_layout()
    plot_path = f"{output_dir}/zenith_azimuth_uncertainty.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved Zenith/Azimuth prediction plot with uncertainties to: {plot_path}")

def plot_opening_angle_distribution(true_directions, predicted_directions, output_dir):
    """
    Plots the distribution of opening angles between true and predicted direction vectors.
    
    Args:
        true_directions (array): True direction vectors.
        predicted_directions (array): Predicted direction vectors.
        output_dir (str): Path to save the plot.
    """
    # Normalize the vectors
    true_directions = normalize_vectors(true_directions)
    predicted_directions = normalize_vectors(predicted_directions)
    
    # Compute angles
    angles = compute_angles_between_vectors(true_directions, predicted_directions)
    
    # Plot histogram
    plt.figure(figsize=(6, 5))
    plt.hist(angles, bins=50, range=(0, 180), color="skyblue", alpha=0.7, edgecolor="black")
    plt.yscale("log")
    plt.xlabel("Opening Angle Between True and Predicted Directions (degrees)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Opening Angles Between True and Predicted Directions")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "opening_angle_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved opening angle distribution plot to: {plot_path}")

def load_model_weights(model, weights, weights_path, trainer=None):
    """
    Loads weights into the model based on the user-specified option.
    If weights are missing, starts training from scratch.
    """
    if weights == "best_weights":
        if os.path.exists(weights_path):
            print(f"Loading best model weights from {weights_path}...")
            checkpoint = torch.load(weights_path)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            print(f"Best model weights not found at {weights_path}. Starting from scratch.")
    elif weights == "last_weights" and trainer is not None:
        print("Using last weights from training (trainer state).")
        # The trainer will use its own mechanism for managing the last checkpoint.
        print(f"Invalid weights option or path not found: {weights}. Starting from scratch.")
    return model

def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def plot_hexbin(true_vals, pred_vals, angle_type, output_dir, gridsize=30):
    
    plt.figure(figsize=(6, 5))
    plt.hexbin(true_vals, pred_vals, gridsize=gridsize, cmap="viridis", mincnt=1)
    plt.plot([true_vals.min(), true_vals.max()], [true_vals.min(), true_vals.max()], '--k')
    plt.xlabel(f"True {angle_type} (rad)")
    plt.ylabel(f"Predicted {angle_type} (rad)")
    plt.title(f"{angle_type} Prediction Hexbin")
    cb = plt.colorbar()
    cb.set_label("Counts")
    # Set empty bins to white
    cb.cmap.set_under("white")
    
    plt.grid()
    plot_path = f"{output_dir}/{angle_type.lower()}_hexbin.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved {angle_type} hexbin plot to: {plot_path}")

def calibration_plot(pred, true, uncertainty, angle_type, output_dir, bins=10):
    """
    Creates a calibration plot for uncertainty estimates.
    
    The function bins the data based on predicted uncertainty, computes the average
    predicted uncertainty and average absolute error within each bin, and plots these
    values against each other. An ideal calibration (where the predicted uncertainty
    matches the actual error) is shown as a y=x reference line.
    
    Args:
        pred (array-like): Predicted angle values.
        true (array-like): True angle values.
        uncertainty (array-like): Predicted uncertainties (e.g., sigma or kappa converted appropriately).
        angle_type (str): Label for the angle type (e.g., "Zenith" or "Azimuth").
        output_dir (str): Directory path to save the plot.
        bins (int, optional): Number of bins to use for calibration. Default is 10.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Calculate absolute error between predictions and true values
    abs_error = np.abs(np.array(pred) - np.array(true))
    uncertainty = np.array(uncertainty)
    
    # Define bins spanning the range of predicted uncertainties
    bin_edges = np.linspace(uncertainty.min(), uncertainty.max(), bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Lists to store mean predicted uncertainty and mean absolute error in each bin
    mean_pred_uncertainty = []
    mean_abs_error = []
    
    # Bin the data and compute averages in each bin
    for i in range(bins):
        # Select data points that fall into the current bin
        bin_mask = (uncertainty >= bin_edges[i]) & (uncertainty < bin_edges[i+1])
        if np.any(bin_mask):
            mean_pred_uncertainty.append(np.mean(uncertainty[bin_mask]))
            mean_abs_error.append(np.mean(abs_error[bin_mask]))
        else:
            # Skip bins with no data
            mean_pred_uncertainty.append(np.nan)
            mean_abs_error.append(np.nan)
    
    # Remove bins that had no data
    mean_pred_uncertainty = np.array(mean_pred_uncertainty)
    mean_abs_error = np.array(mean_abs_error)
    valid = ~np.isnan(mean_pred_uncertainty)
    bin_centers = bin_centers[valid]
    mean_pred_uncertainty = mean_pred_uncertainty[valid]
    mean_abs_error = mean_abs_error[valid]
    
    # Create the calibration plot
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred_uncertainty, mean_abs_error, 'o-', label='Calibration Curve')
    
    # Plot ideal calibration line (y=x)
    min_val = min(mean_pred_uncertainty.min(), mean_abs_error.min())
    max_val = max(mean_pred_uncertainty.max(), mean_abs_error.max())
    x_line = np.linspace(min_val, max_val, 100)
    plt.plot(x_line, x_line, '--', color='black', label='Ideal Calibration')
    
    plt.xlabel("Predicted Uncertainty")
    plt.ylabel("Average Absolute Error")
    plt.title(f"{angle_type} Uncertainty Calibration")
    plt.legend()
    plt.grid()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{angle_type.lower()}_uncertainty_calibration.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved {angle_type} calibration plot to: {plot_path}")

def merge_pulses_with_predictions(dataset, predictions_df):
    """
    1. Collect n_pulses from the dataset.
    2. Merge them with the predictions_df on 'event_no'.
    """
    df_pulses = collect_pulses_from_dataset(dataset)
    
    # Merge so that each event_no in predictions_df gets the matching n_pulses
    df_merged = pd.merge(predictions_df, df_pulses, on="event_no", how="left")
    return df_merged


def collect_pulses_from_dataset(dataset):
    """
    Loops over a dataset of PyG Data objects and collects (event_no, n_pulses).
    Returns a DataFrame with columns ["event_no", "n_pulses"].
    """
    event_n_pulses = []
    for i in range(len(dataset)):
        data = dataset[i]
        # event_no is often a 1-element tensor or list, so extract the integer:
        e_no = int(data.event_no[0])   # or data.event_no.item()
        
        # n_pulses is presumably an integer or 1-element tensor
        n_pulses = int(data.n_pulses)
        
        event_n_pulses.append((e_no, n_pulses))
    
    # Convert to DataFrame
    df_pulses = pd.DataFrame(event_n_pulses, columns=["event_no", "n_pulses"])
    return df_pulses


def plot_good_versus_bad_predictions(
    event_no,
    true_z,
    pred_z,
    n_pulses,
    output_dir,
    threshold=40.0,
    line_value = -260.0,
    mean_threshold = 20.0,
):
    """
    Identify and plot 'good' vs. 'bad' predictions based on an absolute error threshold,
    and save two CSV files with the event numbers for bad and good events.
    
    Args:
        event_no (array-like): Array of event numbers corresponding to each prediction.
        true_z (array-like): Array of true Z values.
        pred_z (array-like): Array of predicted Z values.
        output_dir (str): Directory to save the plot and CSV files.
        threshold (float): Absolute error threshold for classifying events as 'bad'.
                           Defaults to 10.0 (units depend on your data).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1) Compute absolute error for each event
    error = np.abs(pred_z - true_z)
    
    # 2) Define 'bad' vs. 'good' based on threshold
    bad_mask = error > threshold
    good_mask = ~bad_mask
    
    # For the horizontal line, identify events near line_value (e.g. -260)
    line_mask = np.abs(pred_z - line_value) < mean_threshold
    
    # 3) Separate event numbers
    bad_event_no = event_no[line_mask]
    good_event_no = event_no[good_mask]
    
    # 4) Save CSV files for bad and good events
    bad_events_csv = os.path.join(output_dir, "bad_events.csv")
    good_events_csv = os.path.join(output_dir, "good_events.csv")
    
    pd.DataFrame({"event_no": bad_event_no}).to_csv(bad_events_csv, index=False)
    pd.DataFrame({"event_no": good_event_no}).to_csv(good_events_csv, index=False)
    
    print(f"Saved bad events to: {bad_events_csv}")
    print(f"Saved good events to: {good_events_csv}")
    
    # 5) Plot the good vs. bad predictions
    plt.figure(figsize=(8, 6))
    
    # Plot "bad" (line_mask) in red
    plt.scatter(true_z[line_mask], pred_z[line_mask], color='red', s=10, alpha=0.5, label='Bad fits')
    
    # Plot "good" (good_mask) in green
    plt.scatter(true_z[good_mask], pred_z[good_mask], color='green', s=10, alpha=0.5, label='Good fits')
    
    # Reference line y = x
    min_val = min(true_z.min(), pred_z.min())
    max_val = max(true_z.max(), pred_z.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
    
    plt.xlabel('True Z Stopping Position')
    plt.ylabel('Predicted Z Stopping Position')
    plt.legend()
    plt.title(f'Good vs. Bad Fits (threshold={threshold})')
    
    plot_path = os.path.join(output_dir, "good_vs_bad_fits.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved good vs. bad fits plot to: {plot_path}")
    
    # 6) Plot histogram of n_pulses for good vs. bad
    plt.figure(figsize=(8, 6))
    plt.hist(n_pulses[good_mask], bins=50, range=(0, 100), color='green', alpha=0.5, label='Good fits')
    plt.hist(n_pulses[line_mask], bins=50, range=(0, 100), color='red', alpha=0.5, label='Bad fits')
    plt.xlabel('Number of Pulses')
    plt.ylabel('Frequency')
    plt.title('Number of Pulses Distribution')
    plt.legend()
    plt.grid()
    
    plot_path = os.path.join(output_dir, "n_pulses_distribution.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved pulses distribution plot to: {plot_path}")
    

def copy_table_in_chunks(table, conn_in, conn_out, chunk_size=500_000, modify_fn=None):
    """
    Stream a large table from conn_in to conn_out in chunks.
    Optionally modifies the DataFrame using modify_fn(df_chunk) before writing.
    """
    offset = 0
    first_chunk = True
    while True:
        query = f"SELECT * FROM {table} LIMIT {chunk_size} OFFSET {offset};"
        df_chunk = pd.read_sql(query, conn_in)
        if df_chunk.empty:
            break

        if modify_fn:
            df_chunk = modify_fn(df_chunk)

        df_chunk.to_sql(
            table, conn_out, index=False, if_exists="replace" if first_chunk else "append"
        )
        offset += chunk_size
        first_chunk = False


def main():
    args = parse_arguments()
    # Paths
    #db_path = "/groups/icecube/simon/GNN/workspace/Scripts/filtered_all_big_data.db"
    db_path = f"{os.environ['SCRATCH']}/filtered_all_big_data.db"

    #db_path = "/groups/icecube/simon/GNN/workspace/Scripts/filtered_all_sub.db"
    pulsemap = "SplitInIcePulsesSRT"
    truth_table = "truth"
    output_dir = "/groups/icecube/simon/GNN/workspace/Plots"
    #best_model_path = "/groups/icecube/simon/GNN/workspace/Models/best_model-v23.ckpt" # For angles
    train_model_path = "/groups/icecube/simon/GNN/workspace/Models/model_state.pth" 
    #best_model_path = "/groups/icecube/simon/GNN/workspace/Models/best_model-v27.ckpt" # Euclidian_distance
    best_model_path = "/groups/icecube/simon/GNN/workspace/Models/best_model-v31.ckpt" # Kappa_weigthed mean squared error.  
    #best_model_path = "/groups/icecube/simon/GNN/workspace/Models/best_model-v32.ckpt"
    #ensure_index_on_event_no(db_path)
    #preprocess_truth_table(db_path, truth_table)
    
    # Initialize WandbLogger
    wandb.init(project="GNN_Muon_Regression", name="DynEdge_Position_Prediction")
    wandb_logger = WandbLogger(
        project="GNN_Muon_Regression",
        name="DynEdge_Position_Prediction",
        log_model=True,
        save_dir="/groups/icecube/simon/GNN/workspace/Logs",
    )
    
    threshold = 30.0
    
    with sqlite3.connect(db_path) as conn:
        df_pulses = pd.read_sql(f"""
            SELECT event_no, COUNT(*) AS n_pulses
            FROM {pulsemap}
            GROUP BY event_no
        """, conn)
    
    df_filtered = df_pulses[df_pulses["n_pulses"] >= threshold]
    allowed_event_nos = set(df_filtered["event_no"].tolist())

    print(f"Number of events with >= {threshold} pulses: {len(allowed_event_nos)}")
    
    # Split dataset into training and validation events
    connection = sqlite3.connect(db_path)
    event_numbers = pd.read_sql("SELECT DISTINCT event_no FROM truth", connection)
    train_events = event_numbers.iloc[:1300000].values.flatten().tolist()
    val_events = event_numbers.iloc[1300000:].values.flatten().tolist()
    #test_events = event_numbers.iloc[800000:].values.flatten().tolist()
    
    connection.close()
    
    train_events_filtered = list(set(train_events).intersection(allowed_event_nos))
    val_events_filtered   = list(set(val_events).intersection(allowed_event_nos))

    print(f"Filtered train size: {len(train_events_filtered)}")
    print(f"Filtered val size: {len(val_events_filtered)}")
        
    
    features = ["dom_x", "dom_y", "dom_z", "charge", "dom_time", "rde", "pmt_area","pmt_dir_x", "pmt_dir_y", "pmt_dir_z","hlc","string","pmt_number","dom_number","dom_type"]
    truth_labels = [
       # "zenith", "azimuth",  # Direction
        "normalised_position_x", "normalised_position_y", "normalised_position_z",  # Stopping positions
    ]

    
    graph_definition = KNNGraph(
        detector=IceCubeUpgrade(),
        nb_nearest_neighbours=8,
        node_definition=NodesAsPulses(),
        input_feature_names=features,
    )

    train_dataset = SQLiteDataset(
        path=db_path,
        pulsemaps=pulsemap,
        features=features,
        truth=truth_labels,
        selection=train_events_filtered,
        truth_table=truth_table,
        graph_definition=graph_definition,
    )
    val_dataset = SQLiteDataset(
        path=db_path,
        pulsemaps=pulsemap,
        features=features,
        truth=truth_labels,
        selection=val_events_filtered,
        truth_table=truth_table,
        graph_definition=graph_definition,
    )
    
    
    torch.cuda.empty_cache()
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8, worker_init_fn=worker_init_fn)

    val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=False, num_workers=8)

    gnn = DynEdge(nb_inputs=graph_definition.nb_outputs, global_pooling_schemes=["min", "max", "mean", "sum"])

    direction_task = DirectionReconstructionWithUncertainty(
        hidden_size=gnn.nb_outputs,
        target_labels=["zenith", "azimuth"],
        loss_function=DirectionalCosineLossWithKappa(),
    )#CosineAngleLoss_2()
    stop_task = StopPositionReconstruction(
        hidden_size=gnn.nb_outputs,
        target_labels=["normalised_position_x", "normalised_position_y", "normalised_position_z"],
        loss_function=EuclideanDistanceLoss(),    
    )
    model = StandardModel(
        graph_definition=graph_definition,
        backbone=gnn,
        tasks=[stop_task],#, direction_task, stop_task
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=OneCycleLR, 
        #verbose=True,
        scheduler_kwargs={
            "max_lr": 5e-5,
            "steps_per_epoch": len(train_loader),
            "epochs": 200,
            "anneal_strategy": "cos",
            "pct_start": 0.25,
            "div_factor": 100.0,
            "final_div_factor": 100.0,
        },
        scheduler_config={"interval": "step"},
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=200, verbose=True, mode="min"),
        ModelCheckpoint(dirpath="/groups/icecube/simon/GNN/workspace/Models", filename="best_model", monitor="val_loss", save_top_k=1, mode="min"),
    ]
    trainer = Trainer(max_epochs=200, accelerator="gpu", devices=1, log_every_n_steps=10, callbacks=callbacks, logger=wandb_logger)

    if args.mode == "train":
        # Load best weights if specified
        if args.weights == "best_weights":
            model = load_model_weights(model, "best_weights", best_model_path)
        elif args.weights == "last_weights":
            print("Starting training with the last weights (if resuming).")
        else:
            print("No weights loaded. Starting training from scratch.")

        # Train the model
        trainer.fit(model, train_loader, val_loader)
        
    elif args.mode == "predict":
        # Load weights for prediction
        model = load_model_weights(model, args.weights, best_model_path, trainer=trainer)

        # Generate predictions

        predictions = model.predict_as_dataframe(
            val_loader,
            gpus = 1,
            prediction_columns=[
               # "zenith_pred", "azimuth_pred","zenith_kappa", "azimuth_kappa",
                 "stop_position_x_pred", "stop_position_y_pred", "stop_position_z_pred",
                 "x_pred_kappa", "y_pred_kappa", "z_pred_kappa",
               
            ],
            additional_attributes=[
                "event_no",
              #  "n_pulses",
              # "zenith", "azimuth",
                 "normalised_position_x", "normalised_position_y", "normalised_position_z",
            ],
        )

        #pred_zenith,pred_azimuth = predictions["zenith_pred"].values, predictions["azimuth_pred"].values
        #true_zenith,true_azimuth = predictions["zenith"].values, predictions["azimuth"].values
        #zenith_sigma, azimuth_sigma = predictions["zenith_kappa"].values, predictions["azimuth_kappa"].values
        predictions["position_x_pred_scaled"] = predictions["stop_position_x_pred"] * 500.0
        predictions["position_y_pred_scaled"] = predictions["stop_position_y_pred"] * 500.0
        predictions["position_z_pred_scaled"] = predictions["stop_position_z_pred"] * 500.0
        
        predictions["position_x_true_scaled"] = predictions["normalised_position_x"] * 500.0
        predictions["position_y_true_scaled"] = predictions["normalised_position_y"] * 500.0
        predictions["position_z_true_scaled"] = predictions["normalised_position_z"] * 500.0
        
        #Extract true and predicted values for stopping points
        true_stopping = predictions[["position_x_true_scaled", "position_y_true_scaled", "position_z_true_scaled"]].values
        pred_stopping = predictions[["position_x_pred_scaled", "position_y_pred_scaled", "position_z_pred_scaled"]].values
        #Print a sample of pred_stopping
        print(f'pred_stopping: {pred_stopping[:10]}')

        
        # df_merged = merge_pulses_with_predictions(val_dataset, predictions)
        # arr_event_no = df_merged["event_no"].values
        # arr_true_z = df_merged["position_z_true_scaled"].values      # or "position_z_true_scaled"
        # arr_pred_z = df_merged["position_z_pred_scaled"].values      # or "position_z_pred_scaled"
        # arr_n_pulses = df_merged["n_pulses"].values
        
        # # Plot stopping points
        plot_stopping_predictions(true_stopping, pred_stopping, output_dir)
        # plot_good_versus_bad_predictions(
        #     event_no=arr_event_no,
        #     true_z=arr_true_z,
        #     pred_z=arr_pred_z,
        #     n_pulses=arr_n_pulses,
        #     output_dir=output_dir,
        #     threshold=40.0,
        #     line_value=-260.0,
        #     mean_threshold=20.0,
        # )
        # plot_zenith_azimuth_predictions(
        #     pred_zenith, pred_azimuth, true_zenith, true_azimuth, zenith_sigma, azimuth_sigma, output_dir
        # )
        
        #plot_hexbin(true_zenith, pred_zenith, "Zenith", output_dir)
        #plot_hexbin(true_azimuth, pred_azimuth, "Azimuth", output_dir)

        
        # # #print the mean opening angle
        #true_vectors = np.array([np.sin(true_zenith) * np.cos(true_azimuth), np.sin(true_zenith) * np.sin(true_azimuth), np.cos(true_zenith)]).T
        #pred_vectors = np.array([np.sin(pred_zenith) * np.cos(pred_azimuth), np.sin(pred_zenith) * np.sin(pred_azimuth), np.cos(pred_zenith)]).T
        #angles = compute_angles_between_vectors(true_vectors, pred_vectors)
        #print(f"Mean opening angle: {np.degrees(angles).mean()} degrees")
        #plot_opening_angle_distribution(true_vectors, pred_vectors, output_dir)        
        #print the mean opening angle
        
        # Plot direction vectors
        #plot_direction_predictions(true_directions, pred_directions, output_dir)

        #plot_angle_distribution(true_directions, pred_directions, output_dir)
        
        #Make a file with the predictions it should also have event_no, position_x, position_y, position_z, zenith, azimuth
      #  predictions.to_csv(f"{output_dir}/predictions.csv", index=False)
       # print("Predictions saved to CSV file.")
       
       
        # Prepare prediction DataFrame
        # prediction_cols = ["event_no", "zenith_pred", "azimuth_pred", "zenith_kappa", "azimuth_kappa"]
        # prediction_cols = []
        # prediction_df = predictions[prediction_cols].copy()

        # # Output file path
    #     output_pred = "/groups/icecube/simon/GNN/workspace/data/Stopped_muons/filtered_all_big_data_with_predictions_and_flag.db"
    #     #db_name = os.path.basename(db_path).replace(".db", "_with_predictions_and_flag.db")
    #     flagged_output_path = output_pred#os.path.join(output_pred, db_name)
    #     os.makedirs(os.path.dirname(flagged_output_path), exist_ok=True)
    #     # with sqlite3.connect(flagged_output_path) as conn_out, sqlite3.connect(db_path) as conn_in:
    #     #     # Stream + flag the truth table
    #     #     def add_flag(df):
    #     #         df["is_train"] = df["event_no"].isin(train_events_filtered).astype(int)
    #     #         return df

    #     #     copy_table_in_chunks("truth", conn_in, conn_out, chunk_size=100_000, modify_fn=add_flag)
    #     #     print("Copied `truth` table with `is_train` flag.")

    #     #     # Stream the pulsemap without modification
    #     #     copy_table_in_chunks("SplitInIcePulsesSRT", conn_in, conn_out, chunk_size=100_000)
    #     #     print("Copied `SplitInIcePulsesSRT` table.")

    #     #     # Write prediction table (small, in memory is fine)
    #     #     prediction_df.to_sql("prediction", conn_out, index=False, if_exists="replace")
    #     #     print("Added `prediction` table.")


    #     # print(f"Saved new DB to: {flagged_output_path}")

    # # Define only the new columns to add
    # cols_to_merge = [
    #     "event_no",
    #     "position_x_true_scaled",
    #     "position_y_true_scaled",
    #     "position_z_true_scaled",
    #     "x_pred_kappa",
    #     "y_pred_kappa",
    #     "z_pred_kappa",
    # ]

    # # Stream update in batches
    # chunk_size = 100_000
    # offset = 0
    # updated_chunks = []

    # with sqlite3.connect(flagged_output_path) as conn:
    #     print("🔄 Reading and updating prediction table in batches...")

    #     while True:
    #         chunk = pd.read_sql(f"SELECT * FROM prediction LIMIT {chunk_size} OFFSET {offset};", conn)
    #         if chunk.empty:
    #             break

    #         merged_chunk = pd.merge(chunk, predictions[cols_to_merge], on="event_no", how="left")
    #         updated_chunks.append(merged_chunk)
    #         offset += chunk_size

    #     # Combine all updated chunks and write back
    #     full_prediction_updated = pd.concat(updated_chunks, ignore_index=True)
    #     full_prediction_updated.to_sql("prediction", conn, index=False, if_exists="replace")
    #     print("✅ Updated prediction table with new kappa + scaled true position columns.")

    #     # --- Now update `truth` table with new flag ---
    #     def add_is_train_v2_flag(df):
    #         df["is_train_pos"] = df["event_no"].isin(train_events_filtered).astype(int)
    #         return df

    #     print("🔄 Updating truth table with `is_train_v2` flag...")

    #     offset = 0
    #     first_chunk = True
    #     while True:
    #         chunk = pd.read_sql(f"SELECT * FROM truth LIMIT {chunk_size} OFFSET {offset};", conn)
    #         if chunk.empty:
    #             break

    #         chunk = add_is_train_v2_flag(chunk)
    #         chunk.to_sql("truth", conn, index=False, if_exists="replace" if first_chunk else "append")
    #         first_chunk = False
    #         offset += chunk_size

    #     print("✅ Updated truth table with `is_train_v2` flag.")
if __name__ == "__main__":
    main()
