import argparse
import sqlite3
import pandas as pd
import os
import torch
import gc
import numpy as np
import multiprocessing as mp
from collections import OrderedDict
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeUpgrade
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.task.reconstruction import StopPositionReconstruction, DirectionReconstructionWithUncertainty
from graphnet.training.loss_functions import EuclideanDistanceLoss, DirectionalCosineLossWithKappa
from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import SQLiteDataset, worker_init_fn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch import set_float32_matmul_precision
from tqdm import tqdm

set_float32_matmul_precision('medium')
mp.set_start_method("spawn", force=True)
torch.multiprocessing.set_sharing_strategy("file_system")

def ensure_index_on_event_no(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_event_no ON SplitInIcePulsesSRT(event_no);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_event_no_truth ON truth(event_no);")
    print("Index on 'event_no' ensured.")

def safe_worker_init_fn(worker_id):
    worker_init_fn(worker_id)  # existing GraphNet setup
    if torch.cuda.is_available():
        torch.cuda.memory._set_allocator_settings('expandable_segments:False')

def write_predictions_to_db(df, db_path):
    print(f"\n🔄 Preparing to merge {len(df)} predictions into database...")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        df.to_sql("prediction_tmp", conn, index=False, if_exists="replace")

        # Ensure main table exists
        cursor.execute("CREATE TABLE IF NOT EXISTS prediction (event_no INTEGER PRIMARY KEY)")

        tmp_cols = df.columns.tolist()
        if "event_no" in tmp_cols:
            tmp_cols.remove("event_no")

        existing_cols = pd.read_sql("PRAGMA table_info(prediction);", conn)["name"].tolist()
        for col in tmp_cols:
            if col not in existing_cols:
                cursor.execute(f"ALTER TABLE prediction ADD COLUMN {col} REAL")

        for col in tmp_cols:
            cursor.execute(
                f"""UPDATE prediction
                    SET {col} = (
                        SELECT {col} FROM prediction_tmp
                        WHERE prediction_tmp.event_no = prediction.event_no
                    )
                    WHERE event_no IN (SELECT event_no FROM prediction_tmp)"""
            )

        # Insert new rows
        select_cols = ", ".join(["event_no"] + tmp_cols)
        cursor.execute(
            f"""INSERT INTO prediction ({select_cols})
                SELECT {select_cols} FROM prediction_tmp
                WHERE event_no NOT IN (SELECT event_no FROM prediction)"""
        )

        cursor.execute("DROP TABLE prediction_tmp")
        conn.commit()

        # Show preview
        print("\n📋 First 5 rows from prediction table:")
        preview = pd.read_sql("SELECT * FROM prediction LIMIT 5", conn)
        print(preview)


def run_prediction(task):
    input_db = f"{os.environ['SCRATCH']}/filtered_all_big_data.db"
    output_db = f"/groups/icecube/simon/GNN/workspace/data/Stopped_muons/predictions_combined.db"
    pulsemap = "SplitInIcePulsesSRT"
    truth_table = "truth"
    threshold = 30
    batch_size = 64
    chunk_size = 100000

    model_path = {
        "position": "/groups/icecube/simon/GNN/workspace/Models/best_model-v31.ckpt",
        "direction": "/groups/icecube/simon/GNN/workspace/Models/best_model-v23.ckpt",
    }[task]

    prediction_columns = {
        "position": ["stop_position_x_pred", "stop_position_y_pred", "stop_position_z_pred", "x_pred_kappa", "y_pred_kappa", "z_pred_kappa"],
        "direction": ["zenith_pred", "azimuth_pred", "zenith_kappa", "azimuth_kappa"],
    }[task]

    truth_labels = {
        "position": ["position_x", "position_y", "position_z"],
        "direction": ["zenith", "azimuth"],
    }[task]

    features = [
        "dom_x", "dom_y", "dom_z", "charge", "dom_time", "rde",
        "pmt_area", "pmt_dir_x", "pmt_dir_y", "pmt_dir_z",
        "hlc", "string", "pmt_number", "dom_number", "dom_type"
    ]

    ensure_index_on_event_no(input_db)

    with sqlite3.connect(input_db) as conn:
        df_pulses = pd.read_sql(f"SELECT event_no, COUNT(*) AS n_pulses FROM {pulsemap} GROUP BY event_no", conn)
    allowed_event_nos = df_pulses[df_pulses["n_pulses"] >= threshold]["event_no"].astype(int).tolist()

    graph_definition = KNNGraph(
        detector=IceCubeUpgrade(),
        nb_nearest_neighbours=8,
        node_definition=NodesAsPulses(),
        input_feature_names=features,
    )

    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )

    task_instance = StopPositionReconstruction(
        hidden_size=gnn.nb_outputs,
        target_labels=truth_labels,
        loss_function=EuclideanDistanceLoss(),
    ) if task == "position" else DirectionReconstructionWithUncertainty(
        hidden_size=gnn.nb_outputs,
        target_labels=truth_labels,
        loss_function=DirectionalCosineLossWithKappa(),
    )

    model = StandardModel(
        graph_definition=graph_definition,
        gnn=gnn,
        tasks=[task_instance],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=OneCycleLR,
        scheduler_kwargs={
            "max_lr": 5e-5,
            "steps_per_epoch": 1,
            "epochs": 1,
            "anneal_strategy": "cos",
            "pct_start": 0.25,
            "div_factor": 100.0,
            "final_div_factor": 100.0,
        },
        scheduler_config={"interval": "step"},
    )
    model.load_state_dict(torch.load(model_path)["state_dict"], strict=False)

    for i in range(0, len(allowed_event_nos), chunk_size):
        chunk_selection = allowed_event_nos[i:i+chunk_size]
        dataset = SQLiteDataset(
            path=input_db,
            pulsemaps=pulsemap,
            features=features,
            truth=truth_labels,
            selection=chunk_selection,
            truth_table=truth_table,
            graph_definition=graph_definition,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=16,
            worker_init_fn=worker_init_fn,
        )
        print(f"📤 Running prediction on {len(dataset)} events (chunk {i})...")
        df = model.predict_as_dataframe(
            dataloader,
            gpus=1,
            prediction_columns=prediction_columns,
            additional_attributes=["event_no"] + truth_labels,
        )
        print(f"✅ Prediction for chunk {i} complete. Writing to DB...")
        if task == "position":
            df[prediction_columns] = df[prediction_columns] * 500.0

        write_predictions_to_db(df, output_db)
        torch.cuda.empty_cache()
        gc.collect()
    if task == "position":
        print("\nℹ️ Copying filtered pulses to prediction DB...")
        with sqlite3.connect(input_db) as conn_in, sqlite3.connect(output_db) as conn_out:
            event_no_list = tuple(allowed_event_nos)
            query = f"""
                SELECT event_no, dom_x, dom_y, dom_z, charge, dom_time
                FROM {pulsemap}
                WHERE event_no IN {event_no_list}
            """
            df_pulses_filtered = pd.read_sql(query, conn_in)
            df_pulses_filtered.to_sql(pulsemap, conn_out, index=False, if_exists="replace")

    print(f"\n✅ {task.capitalize()} prediction complete and merged into {output_db}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["position", "direction"], required=True, help="Which prediction task to run.")
    args = parser.parse_args()
    run_prediction(args.task)
