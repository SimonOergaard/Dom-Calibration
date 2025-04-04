import argparse
import sqlite3
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeUpgrade
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph

from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.task.reconstruction import StopPositionReconstruction, StartPositionReconstruction, DirectionReconstruction
from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import SQLiteDataset
from graphnet.training.loss_functions import LogCoshLoss, StraightLineLoss, EuclideanDistanceLoss
from graphnet.training.callbacks import PiecewiseLinearLR

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from pytorch_lightning.loggers import WandbLogger
import wandb
import torch

import torch.nn.functional as F
from torch.optim.adam import Adam
from torch import set_float32_matmul_precision
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset

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
    direction_vector = stop_point - start_point
    direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize
    return direction_vector



from typing import List, Union, Optional, Tuple, Any
import sqlite3
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data


class BatchedSQLiteDataset(Dataset):
    def __init__(self, db_path, pulsemaps, features, truth, selection, truth_table):
        self.db_path = db_path
        self.pulsemaps = pulsemaps
        self.features = features
        self.truth = truth
        self.selection = selection
        self.truth_table = truth_table

    def __len__(self):
        return len(self.selection)

    def __getitem__(self, idx):
        event_no = self.selection[idx]
        connection = sqlite3.connect(self.db_path)

        # Pulse data
        query_pulse = f"""
        SELECT {','.join(self.features)}
        FROM {self.pulsemaps}
        WHERE event_no = {event_no};
        """
        pulse_data = pd.read_sql(query_pulse, connection)

        # Truth data
        query_truth = f"""
        SELECT {','.join(self.truth)}
        FROM {self.truth_table}
        WHERE event_no = {event_no};
        """
        truth_data = pd.read_sql(query_truth, connection)
        connection.close()

        # Features as tensor
        x = torch.tensor(pulse_data.values, dtype=torch.float32)

        # Generate edge index (fully connected for now)
        n_nodes = x.size(0)
        edge_index = torch.combinations(torch.arange(n_nodes), r=2).t()

        # Truth values as tensor
        y = torch.tensor(truth_data.values, dtype=torch.float32).squeeze()

        # Create graph
        graph = Data(x=x, edge_index=edge_index, y=y)
        graph.event_no = torch.tensor([event_no])
        return graph








def preprocess_truth_table(db_path, truth_table):
    """
    Adds starting position and direction vector columns to the truth table if they are missing.
    """
    connection = sqlite3.connect(db_path)

    # Load the truth table into a DataFrame
    df_truth = pd.read_sql(f"SELECT * FROM {truth_table};", connection)

    # Check and calculate starting positions
    if "start_position_x" not in df_truth.columns or \
       "start_position_y" not in df_truth.columns or \
       "start_position_z" not in df_truth.columns:
        print("Calculating start positions...")
        start_points = df_truth.apply(calculate_start_point, axis=1)
        df_truth[["start_position_x", "start_position_y", "start_position_z"]] = pd.DataFrame(
            start_points.tolist(), index=df_truth.index
        )
        print("Start positions added successfully.")

    # Check and calculate direction vectors
    if "direction_x" not in df_truth.columns or \
       "direction_y" not in df_truth.columns or \
       "direction_z" not in df_truth.columns:
        print("Calculating direction vectors...")
        directions = df_truth.apply(calculate_direction, axis=1)
        df_truth[["direction_x", "direction_y", "direction_z"]] = pd.DataFrame(
            directions.tolist(), index=df_truth.index
        )

    # Replace the table in the database with the updated DataFrame
    df_truth.to_sql(truth_table, connection, if_exists="replace", index=False)
    print(f"`{truth_table}` table updated with starting positions and direction vectors.")
    connection.close()


# def preprocess_truth_table_with_directions(db_path, truth_table):
#     """
#     Adds direction vector columns (direction_x, direction_y, direction_z) to the truth table 
#     if they are missing, based on start and stop positions.
#     """
#     connection = sqlite3.connect(db_path)

#     # Load the truth table into a DataFrame
#     df_truth = pd.read_sql(f"SELECT * FROM {truth_table};", connection)

#     # Add new columns for the direction vector if they don't already exist
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
#     print(f"`{truth_table}` table updated with direction vector columns.")
#     connection.close()


def plot_stopping_predictions(true_positions, predicted_positions, output_dir):
    """
    Plots scatter plots of predicted vs true stopping positions (x, y, z).
    """
    os.makedirs(output_dir, exist_ok=True)

    labels = ["x", "y", "z"]
    for i, label in enumerate(labels):
        plt.figure(figsize=(8, 8))
        plt.scatter(
            true_positions[:, i],
            predicted_positions[:, i],
            alpha=0.5,
            s=10,
            label=f"{label.upper()}",
        )
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
        plt.savefig(os.path.join(output_dir, f"pred_vs_true_stopping_{label}.png"))
        plt.close()

def plot_direction_predictions(true_directions, predicted_directions, output_dir):
    """
    Plots scatter plots of predicted vs true direction vector components (x, y, z).
    """
    os.makedirs(output_dir, exist_ok=True)

    labels = ["x", "y", "z"]
    for i, label in enumerate(labels):
        plt.figure(figsize=(8, 8))
        plt.scatter(
            true_directions[:, i],
            predicted_directions[:, i],
            alpha=0.5,
            s=10,
            label=f"{label.upper()}",
        )
        plt.plot(
            [true_directions[:, i].min(), true_directions[:, i].max()],
            [true_directions[:, i].min(), true_directions[:, i].max()],
            color="red",
            linestyle="--",
            label="y = x",
        )
        plt.xlabel(f"True {label.upper()} Direction")
        plt.ylabel(f"Predicted {label.upper()} Direction")
        plt.title(f"Predicted vs True {label.upper()} Direction")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, f"pred_vs_true_direction_{label}.png"))
        plt.close()


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

def get_event_numbers_in_chunks(db_path, truth_table, chunk_size=100000):
    """
    Fetches event numbers in chunks to reduce memory usage.
    """
    connection = sqlite3.connect(db_path)
    query = f"SELECT DISTINCT event_no FROM {truth_table}"
    event_numbers = []
    
    for chunk in pd.read_sql(query, connection, chunksize=chunk_size):
        event_numbers.extend(chunk["event_no"].values.flatten().tolist())
    
    connection.close()
    return event_numbers


def main():
    args = parse_arguments()

    # Paths
    db_path = "/groups/icecube/simon/GNN/workspace/Scribts/filtered_all.db"
    pulsemap = "SplitInIcePulsesSRT"
    truth_table = "truth"
    output_dir = "/groups/icecube/simon/GNN/workspace/Plots"
    best_model_path = "/groups/icecube/simon/GNN/workspace/Models/best_model-v6.ckpt"
    train_model_path = "/groups/icecube/simon/GNN/workspace/Models/model_state.pth"

    # Ensure index on event_no for faster queries
    preprocess_truth_table(db_path, truth_table)
    # preprocess_truth_table_with_directions(db_path, truth_table)
    ensure_index_on_event_no(db_path)

    # Initialize WandbLogger
    wandb.init(project="GNN_Muon_Regression", name="DynEdge_Position_Prediction_part2")
    wandb_logger = WandbLogger(
        project="GNN_Muon_Regression",
        name="DynEdge_Position_Prediction_part2",
        log_model=True,
        save_dir="/groups/icecube/simon/GNN/workspace/Logs",
    )

    # Event selection
    connection = sqlite3.connect(db_path)
    events = pd.read_sql(f"SELECT DISTINCT event_no FROM {truth_table};", connection)["event_no"].tolist()
    train_events, val_events, test_events = events[:500000], events[500000:548658], events[548658:]
    connection.close()
    
    features = ["dom_x", "dom_y", "dom_z", "charge", "dom_time", "rde", "pmt_area","pmt_dir_x", "pmt_dir_y", "pmt_dir_z"]
    truth_labels = [
        "position_x", "position_y", "position_z",  # Stopping positions
        "start_position_x", "start_position_y", "start_position_z",  # Starting positions
        "direction_x", "direction_y", "direction_z",  # Direction vectors
    ]


    graph_definition = KNNGraph(
        detector=IceCubeUpgrade(),
        nb_nearest_neighbours=8,
        node_definition=NodesAsPulses(),
        input_feature_names=features,
    )

    train_dataset = BatchedSQLiteDataset(db_path, pulsemap, features, truth_labels, train_events, truth_table)
    val_dataset = BatchedSQLiteDataset(db_path, pulsemap, features, truth_labels, val_events, truth_table)
    test_dataset = BatchedSQLiteDataset(db_path, pulsemap, features, truth_labels, test_events, truth_table)


    # Wrap datasets in DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True, num_workers=8)
    val_loader = DataLoader(dataset=val_dataset, batch_size=512, shuffle=False, num_workers=8)
    test_loader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False, num_workers=8)




    gnn = DynEdge(nb_inputs=graph_definition.nb_outputs, global_pooling_schemes=["min", "max", "mean", "sum"])

    direction_task = DirectionReconstruction(
        hidden_size=gnn.nb_outputs,
        target_labels=["direction_x", "direction_y", "direction_z"],
        loss_function=LogCoshLoss(),  # You could use EuclideanDistanceLoss as well
    )
    stop_task = StopPositionReconstruction(
        hidden_size=gnn.nb_outputs,
        target_labels=["position_x", "position_y", "position_z"],
        loss_function=LogCoshLoss(),
    )
    
    # task = PositionReconstruction_start_and_stop(
    #     hidden_size=gnn.nb_outputs,
    #     target_labels=["position_x", "position_y", "position_z", "start_position_x", "start_position_y", "start_position_z"],
    #     loss_function=StraightLineLoss(position_weight=1.0, direction_weight=0.5),
    # )
    
    model = StandardModel(
        graph_definition=graph_definition,
        backbone=gnn,
        tasks=[direction_task, stop_task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=OneCycleLR,
        scheduler_kwargs={
            "max_lr": 1e-3,
            "steps_per_epoch": len(train_loader),
            "epochs": 100,
            "anneal_strategy": "cos",
            "div_factor": 25.0,
        },
        scheduler_config={"interval": "step"},
    )



    callbacks = [
        EarlyStopping(monitor="val_loss", patience=40, verbose=True, mode="min"),
        ModelCheckpoint(dirpath="/groups/icecube/simon/GNN/workspace/Models", filename="best_model", monitor="val_loss", save_top_k=1, mode="min"),
    ]
    trainer = Trainer(max_epochs=200, accelerator="gpu", devices=1, log_every_n_steps=10, callbacks=callbacks, logger=wandb_logger)
    if args.mode == "train":
        # Load best weights if specified
        if args.weights == "best_weights":
            if os.path.exists(best_model_path):
                print(f"Loading best weights from {best_model_path}...")
                checkpoint = torch.load(best_model_path)
                model.load_state_dict(checkpoint["state_dict"], strict=False)
            else:
                print(f"Best weights not found at {best_model_path}. Starting from scratch.")
        elif args.weights == "last_weights":
            print("Starting training with the last weights (if resuming).")
        else:
            print("No weights loaded. Starting training from scratch.")

        # Train the model
        trainer.fit(model, train_loader, val_loader)


        
    elif args.mode == "predict":
        # Load weights for prediction
        if args.weights == "best_weights" and os.path.exists(best_model_path):
            print(f"Loading best weights from {best_model_path}...")
            checkpoint = torch.load(best_model_path)
            gnn_module.model.load_state_dict(checkpoint["state_dict"], strict=False)
        elif args.weights == "last_weights":
            print("Using last weights for prediction (if available).")
        else:
            print("No weights provided. Starting with the untrained model for predictions.")

        # Generate predictions
        predictions = gnn_module.predict_as_dataframe(
            test_loader,
            prediction_columns=[
                "direction_x_pred", "direction_y_pred", "direction_z_pred",
                "stop_position_x_pred", "stop_position_y_pred", "stop_position_z_pred",
            ],
            additional_attributes=[
                "direction_x", "direction_y", "direction_z",
                "position_x", "position_y", "position_z",
            ],
        )

        # Extract true and predicted values for stopping points
        true_stopping = predictions[["position_x", "position_y", "position_z"]].values
        pred_stopping = predictions[[f"{label}_pred" for label in ["stop_position_x", "stop_position_y", "stop_position_z"]]].values

        # Extract true and predicted values for direction vectors
        true_directions = predictions[["direction_x", "direction_y", "direction_z"]].values
        pred_directions = predictions[["direction_x_pred", "direction_y_pred", "direction_z_pred"]].values

        # Plot stopping points
        plot_stopping_predictions(true_stopping, pred_stopping, output_dir)

        # Plot direction vectors
        plot_direction_predictions(true_directions, pred_directions, output_dir)


if __name__ == "__main__":
    main()
