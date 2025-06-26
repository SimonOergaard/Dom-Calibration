"""Running the trained model on cleaned pulses. Continued from Peters work."""

# Packages
import os
import sqlite3
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam
from torch.nn.functional import one_hot, softmax


from graphnet.training.loss_functions import CrossEntropyLoss
from graphnet.data.constants import FEATURES, TRUTH
#from graphnet.data.sqlite.sqlite_selection import (
#    get_desired_event_numbers,
#    get_equal_proportion_neutrino_indices,
#)

from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
#from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.graphs import KNNGraph
#from graphnet.models.task.reconstruction import MulticlassClassificationTask
from graphnet.models.task.classification import MulticlassClassificationTask
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.utils import (
    get_predictions,
    make_train_validation_dataloader,
    make_dataloader,
    save_results,
)
#from graphnet.utilities.logging import get_logger

import numpy as np
import pandas as pd
import csv

from torch import set_float32_matmul_precision
set_float32_matmul_precision('medium')  # 'medium' for performance vs 'high' for precision

# Configurations
torch.multiprocessing.set_sharing_strategy("file_system")

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
truth = TRUTH.DEEPCORE[:-1]

# Main function definition
def main(
    input_path: str,
    output_path: str,
    model_path: str,
):
    # Burnsample rigtig data og mc data (begge ligger i burnsample). Dvs. den data du vil bruge din model på
    #RD_selection = pd.read_csv('/groups/icecube/peter/workspace/analyses/multi_classification_on_stop_and_track_muons/plotting/Comparison_RD_MC/dev_lvl3_genie_burnsample_RD_event_numbers.csv').reset_index(drop = True)['event_no'].ravel().tolist()
    #MC_selection = pd.read_csv('/groups/icecube/peter/workspace/analyses/multi_classification_on_stop_and_track_muons/plotting/Comparison_RD_MC/dev_lvl3_genie_burnsample_MC_event_numbers.csv').reset_index(drop = True)['event_no'].ravel().tolist()

    db_path = input_path
    connection = sqlite3.connect(db_path)
    event_numbers = pd.read_sql("SELECT DISTINCT event_no FROM SRTInIcePulses", connection)
    val_events = event_numbers.sample(frac=0.5, random_state=42).reset_index(drop=True)
    val_events = val_events['event_no'].ravel().tolist()
    print(f"Length of val_events: {len(val_events)}")
    connection.close()


    # Configuration
    config = {
        "db": input_path,
        "pulsemap": "SRTInIcePulses",
        "batch_size": 512,
        #"num_workers": 10, #Think this is 15 in training
        "num_workers": 15,
        "accelerator": "gpu",
        #Important!!! Needs to be changed to whatever GPU is free 
        "devices": 1,
        "target": "pid",
        "n_epochs": 1,
        "patience": 1,
    }
    archive = output_path
    # Run name
    run_name = "dynedge_trained_on_peter_test_{}_predict_40_days_first_no_5".format(
        config["target"]
    )

    # Log configuration to W&B
    # wandb_logger.experiment.config.update(config)
    # train_selection, _ = get_equal_proportion_neutrino_indices(config["db"])
    # train_selection = train_selection[0:50000]

#    prediction_dataloader_MC = make_dataloader(
    #    config["db"],
    #    config["pulsemap"],
    #    features,
    #    truth,
    #    selection = MC_selection,
    #    batch_size=config["batch_size"],
    #    shuffle=False,
    #    num_workers=config["num_workers"],
#    )


    # Building model
    #detector = IceCubeDeepCore(
    #    graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    #)

    graph_definition = KNNGraph(
        detector = IceCubeDeepCore(),
        nb_nearest_neighbours=8,
        node_definition=NodesAsPulses(),   # nearest neighbors and node definition was added
        input_feature_names=features
        
    )

    prediction_dataloader = make_dataloader(
        db = config["db"],
        pulsemaps = config["pulsemap"],
        features = features,
        truth = truth,
        selection = val_events,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        graph_definition = graph_definition
    )

    


    #gnn = DynEdge(
    #    nb_inputs=detector.nb_outputs,
    #    global_pooling_schemes=["min", "max", "mean", "sum"],
    #)

    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )

    task = MulticlassClassificationTask(
        #nb_inputs=3,
        nb_outputs = 3,
        hidden_size=gnn.nb_outputs,
        target_labels=config["target"],
        loss_function=CrossEntropyLoss(options={1: 0, -1: 0, 13: 1, -13: 1, 12: 2, -12: 2, 14: 2, -14: 2, 16: 2, -16: 2}),
        #loss_function=CrossEntropyLoss(),
        transform_inference=lambda x: softmax(x,dim=-1),
    )
    model = StandardModel(
        #detector=detector,
        graph_definition=graph_definition,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-04, "eps": 1e-03},
        scheduler_class=PiecewiseLinearLR,
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
        logger = None,
    )

    # Load model
    model.load_state_dict(model_path)


    # Saving predictions to file
    resultsRD = get_predictions(
        trainer = trainer,
        model = model,
        dataloader = prediction_dataloader,
        # Hvad der skal predictes (navngivning)
        prediction_columns = [config["target"] + "_noise_pred", config["target"] + "_muon_pred", config["target"]+ "_neutrino_pred"],
        additional_attributes=[config["target"], "event_no"],# "EventID", "SubEventID", "RunID", "SubrunID"],
    )


    # save_results(config["db"], run_name, results, archive, model)
    # Giv  navn ovenpå outputfolderen
    resultsRD.to_csv(
        output_folder + "/{}_deployment_model_prediction_burnsample_MC_data.csv".format(config["target"])
    )

    
if __name__ == "__main__":
    # Input databasen
    input_db      = f"{os.environ['SCRATCH']}/merged.db"
    # Output folderen
    output_folder = "/groups/icecube/simon/GNN/workspace/storage/Training/stopped_through_classification"
    
    #Hvor ligger modellen
    #model_path = "/groups/icecube/cjb924/workspace/work/peter_continued/predicted_runed_training/osc_next_level3_v2/dynedge_pid_Frederik_test_multiclass_peter_måned_1_validation_set/dynedge_pid_Frederik_test_multiclass_peter_måned_1_validation_set_state_dict.pth"
    #model_path = "/groups/icecube/cjb924/workspace/work/peter_continued/predicted_runed_training/osc_next_level3_v2/dynedge_pid_Frederik_test_multiclass_peter_måned_1_no_2_validation_set/dynedge_pid_Frederik_test_multiclass_peter_måned_1_no_2_validation_set_state_dict.pth"
    #model_path = "/groups/icecube/cjb924/workspace/work/peter_continued/predicted_runed_training/osc_next_level3_v2/dynedge_pid_Frederik_test_multiclass_peter_måned_1_no_4_validation_set/dynedge_pid_Frederik_test_multiclass_peter_måned_1_no_4_validation_set_state_dict.pth"
    #model_path = "/groups/icecube/cjb924/workspace/work/peter_continued/predicted_runed_training/osc_next_level3_v2/dynedge_pid_Frederik_test_multiclass_peter_måned_1_no_5_validation_set/dynedge_pid_Frederik_test_multiclass_peter_måned_1_no_5_validation_set_state_dict.pth"
    model_path = "/groups/icecube/cjb924/workspace/work/peter_continued/predicted_runed_training/osc_next_level3_v2/deployment_model/osc_next_level3_v2/dynedge_pid_Frederik_multiclass_deployment_model_validation_set/dynedge_pid_Frederik_multiclass_deployment_model_validation_set_state_dict.pth"

    main(input_db, output_folder, model_path)

