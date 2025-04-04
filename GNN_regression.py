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
from graphnet.models.task.reconstruction import PositionReconstruction, StopPositionReconstruction, StartPositionReconstruction
from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import SQLiteDataset
from graphnet.training.loss_functions import LogCoshLoss, EuclideanDistanceLoss
from graphnet.training.callbacks import PiecewiseLinearLR

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch
from torch.optim.adam import Adam
import torch.nn.functional as F
from torch import set_float32_matmul_precision
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

set_float32_matmul_precision('medium')  # 'medium' for performance vs 'high' for precision


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



def preprocess_truth_table(db_path, truth_table):
    """
    Adds start position columns (start_position_x, start_position_y, start_position_z) to the truth table 
    if they are missing, based on the stop position, direction, and track length.
    """
    connection = sqlite3.connect(db_path)

    # Load the truth table into a DataFrame
    df_truth = pd.read_sql(f"SELECT * FROM {truth_table};", connection)

    # Check if the necessary columns exist in the table
    required_columns = ["position_x", "position_y", "position_z", "zenith", "azimuth", "track_length"]
    for column in required_columns:
        if column not in df_truth.columns:
            raise ValueError(f"Column '{column}' is missing from the '{truth_table}' table.")

    # Add new columns for the start position if they don't already exist
    if "start_position_x" not in df_truth.columns or \
       "start_position_y" not in df_truth.columns or \
       "start_position_z" not in df_truth.columns:
        print("Calculating start positions...")
        start_points = df_truth.apply(calculate_start_point, axis=1)
        df_truth[["start_position_x", "start_position_y", "start_position_z"]] = pd.DataFrame(
            start_points.tolist(), index=df_truth.index
        )

    # Replace the table in the database with the updated DataFrame
    df_truth.to_sql(truth_table, connection, if_exists="replace", index=False)
    print(f"`{truth_table}` table updated with start position columns.")
    connection.close()





def plot_predictions(true_positions, predicted_positions, output_dir, position_type = "Stopping"):
    """
    Plots scatter plots of predicted vs true positions for x, y, and z coordinates.
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
        plt.xlabel(f"True {label.upper()} {position_type} Position")
        plt.ylabel(f"Predicted {label.upper()} {position_type} Position")
        plt.title(f"Predicted vs True {label.upper()} {position_type} Position")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, f"pred_vs_true_{position_type.lower()}_{label}.png"))
        plt.close()

def main():
    # Paths
    db_path = "/groups/icecube/simon/GNN/workspace/filtered.db"
    #data_dir = "/lustre/hpc/project/icecube/upgrade_sim/level4_queso/converted_sql/"
    #db_path = f"{data_dir}/140029_cc.db"
    pulsemap = "SplitInIcePulsesSRT"
    truth_table = "truth"
    output_dir = "/groups/icecube/simon/GNN/workspace/Plots"
    trained_model_path = "/groups/icecube/simon/GNN/workspace/Models/model_state.pth"
    best_model_path = "/groups/icecube/simon/GNN/workspace/Models/best_model.ckpt"
    connection = sqlite3.connect(db_path)
    try:
        df_truth = pd.read_sql(f"SELECT * FROM {truth_table} LIMIT 5;", connection).copy()
        print("Truth table already exists.")
    except:
        print("Truth table does not exist. Creating it...")
    # Ensure index on event_no for faster queries
    preprocess_truth_table(db_path, truth_table)
    ensure_index_on_event_no(db_path)
    connection.close()
    
    wandb.init(project="GNN_Muon_Regression", name="DynEdge_Position_Prediction")

    # Initialize WandbLogger
    wandb_logger = WandbLogger(
        project="GNN_Muon_Regression",
        name="DynEdge_Position_Prediction",
        log_model=True,
        save_dir="/groups/icecube/simon/GNN/workspace/Logs",
    )

    # Split dataset into training and validation events
    connection = sqlite3.connect(db_path)
    event_numbers = pd.read_sql("SELECT DISTINCT event_no FROM truth", connection)
    train_events = event_numbers.iloc[:100000].values.flatten().tolist()
    val_events = event_numbers.iloc[100000:140000].values.flatten().tolist()
    test_events = event_numbers.iloc[140000:150000].values.flatten().tolist()
    connection.close()

    print(f"Training on {len(train_events)} events and validating on {len(val_events)} events.")

    # Define graph representation
    features = ["dom_x", "dom_y", "dom_z", "charge", "dom_time", "rde", "pmt_area"]
    truth_labels = ["position_x", "position_y", "position_z", "start_position_x", "start_position_y", "start_position_z"]

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
        selection=train_events,
        truth_table=truth_table,
        graph_definition=graph_definition,
    )

    val_dataset = SQLiteDataset(
        path=db_path,
        pulsemaps=pulsemap,
        features=features,
        truth=truth_labels,
        selection=val_events,
        truth_table=truth_table,
        graph_definition=graph_definition,
    )
    
    test_dataset = SQLiteDataset(
        path=db_path,
        pulsemaps=pulsemap,
        features=features,
        truth=truth_labels,
        selection=test_events,
        truth_table=truth_table,
        graph_definition=graph_definition,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=8,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=8,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=8,
    )
    # Define the model
    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )

    # task = PositionReconstruction(
    #     hidden_size=gnn.nb_outputs,
    #     target_labels=truth_labels,
    #     loss_function=LogCoshLoss(),
    # )
    start_task = StartPositionReconstruction(
        hidden_size=gnn.nb_outputs,
        target_labels=["start_position_x", "start_position_y", "start_position_z"],
        loss_function=EuclideanDistanceLoss(),
    )
    stop_task = StopPositionReconstruction(
        hidden_size=gnn.nb_outputs,
        target_labels=["position_x", "position_y", "position_z"],
        loss_function=EuclideanDistanceLoss(),
    )
    model = StandardModel(
        graph_definition=graph_definition,
        backbone=gnn,
        tasks=[start_task, stop_task],
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
    # Load weights from best model (if available)
    if os.path.exists(best_model_path):
        print(f"Loading weights from {best_model_path}...")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)  # Load only the model weights
    else:
        print("No pre-trained best model found. Starting from scratch.")
        
    # Training setup with EarlyStopping and ModelCheckpoint
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=40, verbose=True, mode="min"),
        ModelCheckpoint(
            dirpath="/groups/icecube/simon/GNN/workspace/Models",
            filename="best_model",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        ),
    ]
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")
    trainer = Trainer(
        max_epochs=200,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=10,
        callbacks=callbacks,
        logger=wandb_logger,
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    print("Training completed.")

    # Save final model
    os.makedirs("/groups/icecube/simon/GNN/workspace/Models", exist_ok=True)
    final_model_path = "/groups/icecube/simon/GNN/workspace/Models/final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    # Generate predictions
    
    predictions = model.predict_as_dataframe(
        test_loader,
        prediction_columns=[
            "start_position_x_pred", "start_position_y_pred", "start_position_z_pred",
            "stop_position_x_pred", "stop_position_y_pred", "stop_position_z_pred"
        ],  # Match model output
        additional_attributes=[
            "start_position_x", "start_position_y", "start_position_z",
            "position_x", "position_y", "position_z"
        ],  # Include true positions for comparison
    )

    # Save predictions and log to W&B
    wandb_logger.experiment.log({"validation_results": wandb.Table(dataframe=predictions)})

    true_stopping = predictions[["position_x", "position_y", "position_z"]].values
    pred_stopping = predictions[[f"{label}_pred" for label in ["stop_position_x", "stop_position_y", "stop_position_z"]]].values
    plot_predictions(true_stopping, pred_stopping, output_dir, position_type="Stopping")

    # Plot predictions for starting positions
    true_starting = predictions[["start_position_x", "start_position_y", "start_position_z"]].values
    pred_starting = predictions[[f"{label}_pred" for label in ["start_position_x", "start_position_y", "start_position_z"]]].values
    plot_predictions(true_starting, pred_starting, output_dir, position_type="Starting")

    # Finish W&B logging
    wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()