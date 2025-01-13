"""Example of training Model."""

import os
from typing import Any, Dict, List, Optional

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from  pytorch_lightning import Trainer
import torch
from torch.utils.data import DataLoader
from torch.optim.adam import Adam

from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.task.classification import BinaryClassificationTask
from graphnet.training.loss_functions import BinaryCrossEntropyLoss
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.loss_functions import LogCoshLoss
from graphnet.training.utils import make_train_validation_dataloader
from graphnet.training.utils import make_dataloader, get_predictions, save_results
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger
import multiprocessing

import numpy as np

print("All imported!")
# Source Panagiotis Tzatzagos
# Construct Logger
logger = Logger()

# Constants
features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86


training_data_path  = '/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_Peter_and_Morten/merged_database/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_130000_888003_retro.db'
event_no_path = '/groups/icecube/debes/work/analyses/Stopped_muon_labeling/1M_stopped_and_through_muons_mixed_event_nos.txt'   # select events to run our classifier
OUTPUT_DIR = '/groups/icecube/simon/GNN/workspace/storage/Training/stopped_through_classification/'


# Make sure W&B output directory exists
#WANDB_DIR = "./wandb/"
#os.makedirs(WANDB_DIR, exist_ok=True)

# Event numbers (must be integers in list)
event_nos = np.loadtxt(event_no_path).tolist()
N_train = int(len(event_nos)*0.8) #800k
train_events = [int(x) for x in event_nos[:10000]]#[:1000]
val_events = [int(x) for x in event_nos[10000:12000]]#[1000:1200]

def main(
    path: str,
    pulsemap: str,
    target: str,
    truth_table: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    #wandb: bool = True,
) -> None:
    """Run example."""
  

    # Initialise Weights & Biases (W&B) run
    # wandb_logger = WandbLogger(
    #     project="ptzatzag",
    #     #entity="graphnet-team",
    #     name="Panos_test_run",
    #     save_dir=WANDB_DIR,
    #     log_model=True,
    # )

    logger.info(f"features: {features}")
    logger.info(f"truth: {truth}")

    # Configuration
    config: Dict[str, Any] = {
        "path": path,
        "pulsemap": pulsemap,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "target": target,
        "early_stopping_patience": early_stopping_patience,
        "fit": {
            "gpus": [0],
            "max_epochs": max_epochs,
            },
    }

    archive = os.path.join(OUTPUT_DIR, "train_model_without_configs")
    run_name = "dynedge_{}_example".format(config["target"])

    # Log configuration to W&B
    #wandb_logger.experiment.config.update(config)

    # Define graph representation
    graph_definition = KNNGraph(detector = IceCube86(),
                                nb_nearest_neighbours = 8,
                                node_definition=NodesAsPulses(),
                                input_feature_names= features# nearest neighbors and node definition was added
                                )   

   
   
    training_dataloader = make_dataloader(
        db=config["path"],
        graph_definition=graph_definition,
        pulsemaps=config["pulsemap"],
        features=features,
        truth=truth,
        batch_size=config["batch_size"],
        shuffle=False,
        selection=train_events,
        num_workers=config["num_workers"],
        truth_table=truth_table,
    )

    validation_dataloader = make_dataloader(
        db=config["path"],
        pulsemaps=config["pulsemap"],
        graph_definition=graph_definition,
        features=features,
        truth=truth,
        batch_size=config["batch_size"],
        shuffle=False,
        selection=val_events,
        num_workers=config["num_workers"],
        truth_table=truth_table,
    )

    # Models architecture
    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,   
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )
    task = BinaryClassificationTask(
        hidden_size=gnn.nb_outputs,   # dimensions that the gnn layer produces 
        target_labels=config["target"],
        loss_function=BinaryCrossEntropyLoss(),
        )
    
    # Construct the Model
    model = StandardModel(
        graph_definition=graph_definition,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [                # List of step indices. Must be increasing ?
                0,
                len(training_dataloader) / 2,
                len(training_dataloader) * config["fit"]["max_epochs"],
            ],
            "factors": [1e-2, 1, 1e-02],   # List of multiplicative factors. Must be same length as milestones?
        },
        scheduler_config={
            "interval": "step",
        },
    )
    
    # Training model
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
        ),
        ProgressBar(),
    ]
    
    # Configure Trainer ########## CHECK THIS ###########
    trainer = Trainer(
        #accelerator=config["accelerator"],
        #devices = "gpu",
        max_epochs=config["fit"]["max_epochs"],
        callbacks=callbacks,
        log_every_n_steps=1,
        #logger=wandb_logger,
        )

    # Train model
    trainer.fit(
        model,
        training_dataloader,
        validation_dataloader,
    )
     
    # Get predictions
    prediction_columns = [config["target"] + "_pred"]
    additional_attributes = [config["target"]]
    
    additional_attributes = model.target_labels    ### CHECK THIS
    assert isinstance(additional_attributes, list)  # mypy

    # Predict on test set and return as pandas.DataFrame
    results = model.predict_as_dataframe(
        validation_dataloader,
        prediction_columns=prediction_columns,
        additional_attributes=additional_attributes + ["event_no"],
    )

    # Save predictions and model to file
    ##db_name = path.split("/")[-1].split(".")[0]
    ##path = os.path.join(archive, db_name, run_name)
    ##logger.info(f"Writing results to {path}")
    ##os.makedirs(path, exist_ok=True)
    
    # Save results as .csv
    ##results.to_csv(f"{path}/results.csv")

    # Save full model (including weights) to .pth file - not version safe
    # Note: Models saved as .pth files in one version of graphnet
    #       may not be compatible with a different version of graphnet.
    ##model.save(f"{path}/model.pth")

    # Save model config and state dict - Version safe save method.
    # This method of saving models is the safest way.
    ##model.save_state_dict(f"{path}/state_dict.pth")
    #model.save_config(f"{path}/model_config.yml")
    
    
    # Define paths
    db_name = path.split("/")[-1].split(".")[0]
    result_path = os.path.join(archive, db_name, run_name)

    # Log information
    logger.info(f"Writing results to {result_path}")

    # Ensure the directory exists
    os.makedirs(result_path, exist_ok=True)

    # Save results as .csv
    results.to_csv(f"{result_path}/results.csv")

    # Save model state dictionary
    model.save_state_dict(f"{result_path}/state_dict.pth")

    # Save model configuration as a YAML file
    model.save_config(f"{result_path}/model_config.yml")


if __name__ == "__main__":
    
    torch.multiprocessing.set_sharing_strategy("file_system") 
    # Parse command-line arguments
    parser = ArgumentParser(
        description=
        """
        Train GNN model without the use of config files.
        """
        )
    
    # add arguments for the model: 
    # Give the path to the directory here:
    parser.add_argument(
        "--path",
        help="Path to dataset file (default: %(default)s)",
        default=f"{training_data_path}",
    )

    parser.add_argument(
        "--pulsemap",
        help="Name of pulsemap to use (default: %(default)s)",
        default="SplitInIcePulses",
    )

    parser.add_argument(
        "--target",
        help=(
            "Name of feature to use as classification target (default: "
            "%(default)s)"
        ),
        default="stopped_muon",   # change the task to get the through muons
        #default="through_muon",
    )

    parser.add_argument(
        "--truth-table",
        help="Name of truth table to be used (default: %(default)s)",
        default="truth",
    )

    # Give the name of the file you wanna create
    parser.add_argument(
    "-r",
    "--run_name",
    dest="run_name",
    type=str,
    help="<required> the name for the model. [str]",
    #New name
    default="Panos TestRun"#last_one_lvl3MC_SplitInIcePulses_21.5_mill_equal_frac"
    # required=True,
    )
    
    parser.with_standard_arguments(
        "gpus",
        ("max-epochs", 50),
        "early-stopping-patience",    # probably I should pass a value here  
        ("batch-size", 512),
        "num-workers", 
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="If True, Weights & Biases are used to track the experiment.",
    )

    #args, unknown = parser.parse_known_args()
    args = parser.parse_args()
    print("Argparse done")


    main(
        args.path,
        args.pulsemap,
        args.target,
        args.truth_table,
        args.gpus,
        args.max_epochs,
        args.early_stopping_patience,
        args.batch_size,
        args.num_workers,
    )
