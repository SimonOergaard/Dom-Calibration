import os
import sqlite3
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam
from torch.nn.functional import one_hot, softmax
from graphnet.data.dataset import SQLiteDataset
from graphnet.models import Model
from graphnet.utilities.config import ModelConfig
from graphnet.training.loss_functions import CrossEntropyLoss,BinaryCrossEntropyLoss, LogCoshLoss
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models.detector import IceCube86
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.task.classification import BinaryClassificationTask
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.utils import (
    get_predictions,
    make_train_validation_dataloader,
    make_dataloader,
    save_results,
)
from graphnet.data.dataloader import DataLoader
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
from torch import set_float32_matmul_precision
set_float32_matmul_precision('medium')  # 'medium' for performance vs 'high' for precision

# Constants
features = [
        "dom_x",
        "dom_y",
        "dom_z",
        "dom_time",
        "charge",
        "rde",
        "pmt_area",
    ]
truth = ["stopped_muon"]

def ensure_index_on_event_no(db_path: str):
    """
    Adds an index on event_no for all tables that contain this column.
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        # Fetch all table names
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cur.fetchall()]

        for table in tables:
            try:
                # Check if 'event_no' is a column in the table
                cur.execute(f"PRAGMA table_info({table});")
                columns = [col[1] for col in cur.fetchall()]
                if "event_no" in columns:
                    index_name = f"idx_{table}_event_no"
                    cur.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}(event_no);")
                    print(f"Index on event_no created for table: {table}")
            except Exception as e:
                print(f"Error while indexing table {table}: {e}")

        conn.commit()

# Main function definition
def main(
    input_path: str,
    output_path: str,
    model_path: str,
):
    #MC_selection = pd.read_parquet('/groups/icecube/ptzatzag/work/workspace/training_models/ClassificationStoppedThrough/McSelection_Model2_1Mil.parquet').reset_index(drop = True)['event_no'].ravel().tolist()

    db_path = input_path
    ensure_index_on_event_no(db_path)
    connection = sqlite3.connect(db_path)
    event_numbers = pd.read_sql("SELECT DISTINCT event_no FROM truth", connection)
    val_events = event_numbers.sample(frac=0.5, random_state=42).reset_index(drop=True)
    val_events = val_events['event_no'].to_numpy().tolist()

    print(f"Length of val_events: {len(val_events)}")
    connection.close()
    
    # Configuration
    config = {
        "db": input_path,
        "pulsemap": "SRTInIcePulses",
        "truth_table": "truth",
        "batch_size": 512,
        "num_workers": 12,
        "accelerator": "gpu",
        "devices": 1,
        "target": "stopped_muon",   
        "n_epochs": 1,
        "patience": 1,
    }

    # Define graph representation
    graph_definition = KNNGraph(detector = IceCube86(),
                                nb_nearest_neighbours = 8,
                                node_definition=NodesAsPulses(),   # nearest neighbors and node definition was added
                                input_feature_names=features,
                                )   
    
    prediction_dataloader_RD = make_dataloader(
        db=config["db"],
        pulsemaps=config["pulsemap"],
        graph_definition=graph_definition,
        features=features,
        truth=truth,
        selection=val_events,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    # print("Starting to load the dataset...")
    # val_dataset = SQLiteDataset(
    #     path=config["db"],
    #     pulsemaps=config["pulsemap"],
    #     features=features,
    #     truth=truth_label,
    #     selection=val_events,
    #    truth_table=config["truth_table"],
    #     graph_definition=graph_definition,
    # )
    # print("Dataset loaded successfully.")
    # val_loader = DataLoader(dataset=val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    # import time

    # for i, batch in tqdm(enumerate(val_loader), total=1000):
    #     start = time.time()
        
    #     simulate processing (or measure real processing if needed)
    #     _ = batch  # access to trigger loading
        
    #     print(f"Batch {i}: {time.time() - start:.4f} s")
        
    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )

    task = BinaryClassificationTask(
        hidden_size=gnn.nb_outputs,
        target_labels=config["target"],
        loss_function=BinaryCrossEntropyLoss(),
        )

    model = StandardModel(
        graph_definition=graph_definition,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=PiecewiseLinearLR,
        scheduler_config={
            "interval": "step",
        },
    )

    # Training model
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["patience"],
        ),
        ProgressBar(),
    ]

    trainer = Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        max_epochs=config["n_epochs"],
        callbacks=callbacks,
        log_every_n_steps=1,
        #logger=wandb_logger,
    )
#################################### CHECK AGAIN HOW TO LOAD THE MODEL ####################################
    # Load model
    #model_config = ModelConfig.load("model_config.yml")
    #model = Model.from_config(model_config)  # With randomly initialised weights.
    model.load_state_dict(torch.load(model_path))       # Now with trained weight.
#################################### CHECK AGAIN HOW TO LOAD THE MODEL ####################################
    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint, strict=False)
   # model.to("cuda" if torch.cuda.is_available() else "cpu")
   # model.eval()
    predictions = model.predict_as_dataframe(
        prediction_dataloader_RD,
        gpus=1,
        prediction_columns=[config["target"] + "_pred"],
        additional_attributes=["event_no", "stopped_muon"],
    )
    predictions.to_csv(
        os.path.join(output_path, f"{config['target']}_predictions.csv"),
        index=False,
    )
    
    # # Saving predictions to file
    # resultsRD = get_predictions(
    #     trainer,
    #     model,
    #     prediction_dataloader_RD,
    #     [config["target"] + "_pred"],
    #     additional_attributes=["event_no","stopped_muon"],
    # )
    # output_db_path = os.path.join(output_folder, f"{config['target']}__MC_1Mil_db02.db")
    # conn_out = sqlite3.connect(output_db_path)

    # resultsRD.to_sql(
    #     name="results",  # or any table name you want
    #     con=conn_out,
    #     index=False,
    #     if_exists="replace",
    # )

    # conn_out.close()

# Main function call
if __name__ == "__main__":

    #input_db      = "/groups/icecube/petersen/GraphNetDatabaseRepository/dev_lvl3_genie_burnsample/dev_lvl3_genie_burnsample_v5.db"
    input_db      = f"{os.environ['SCRATCH']}/muon_sorted.db"#f"{os.environ['SCRATCH']}/filtered_all_big_data.db" ##  
    model_path    = "/groups/icecube/ptzatzag/work/workspace/storage/Training/stopped_through_classification/train_model_without_configs/osc_next_level3_v2/dynedge_stopped_muon_example/state_dict.pth"
    #input_db = "/tmp/filtered_all_no_upgrade.db"
    #model_path = "/tmp/state_dict.pth"

    output_folder = "/groups/icecube/simon/GNN/workspace/storage/Training/stopped_through_classification"
    
    main(input_db, output_folder, model_path)
