"""Example of converting I3-files to SQLite and Parquet."""

import glob, os
import numpy as np
from graphnet.constants import EXAMPLE_OUTPUT_DIR, TEST_DATA_DIR
from graphnet.data.extractors.icecube import (
    I3FeatureExtractorIceCubeUpgrade,
    I3FeatureExtractorIceCube86,
    I3RetroExtractor,
    I3TruthExtractor,
    I3PISAExtractor,
    I3LineFitExtractor
)
from graphnet.data.dataconverter import DataConverter
from graphnet.data.parquet import ParquetDataConverter
from graphnet.data.sqlite import SQLiteDataConverter
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import Logger
from icecube import icetray, dataclasses, simclasses, StartingTrackVeto, recclasses

ERROR_MESSAGE_MISSING_ICETRAY = (
    "This example requires IceTray to be installed, which doesn't seem to be "
    "the case. Please install IceTray; run this example in the GraphNeT "
    "Docker container which comes with IceTray installed; or run an example "
    "script in one of the other folders:"
    "\n * examples/02_data/"
    "\n * examples/03_weights/"
    "\n * examples/04_training/"
    "\n * examples/05_pisa/"
    "\nExiting."
)

CONVERTER_CLASS = {
    "sqlite": SQLiteDataConverter,
    "parquet": ParquetDataConverter,
}


def main_icecube86(backend: str) -> None:
    """Convert IceCube-86 I3 files to intermediate `backend` format."""
    # Check(s)
    assert backend in CONVERTER_CLASS

    inputs = "/lustre/hpc/project/icecube/MuonGun_upgrade_full_detector_generation_volume_no_kde/130028/"
    outdir = "/lustre/hpc/project/icecube/MuonGun_upgrade_full_detector_generation_volume_no_kde/130028/"
    gcd_rescue = (
        "/lustre/hpc/project/icecube/MuonGun_upgrade_full_detector_generation_volume_no_kde/130028/GCD/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V1.i3.bz2"
    )

    converter = CONVERTER_CLASS[backend](
        extractors=[
            #I3FeatureExtractorIceCube86("SRTInIcePulses"),
            I3TruthExtractor(),
        ],
        outdir=outdir,
        gcd_rescue=gcd_rescue,
        workers=1,
    )
    converter(inputs)
    if backend == "sqlite":
        converter.merge_files()


def main_icecube_upgrade(backend: str) -> None:
    """Convert IceCube-Upgrade I3 files to intermediate `backend` format."""
    # Check(s)
    assert backend in CONVERTER_CLASS

    inputs =  "/groups/icecube/simon/GNN/workspace/data/I3_files/132028_part2/"
    outdir =  "/groups/icecube/simon/GNN/workspace/data/Converted_I3_file/merged"#"/lustre/hpc/project/icecube/MuonGun_upgrade_full_detector_generation_volume_no_kde/130028/"
    gcd_rescue = (
        "/groups/icecube/simon/GNN/workspace/data/GCD_files/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V1.i3.bz2"
    )
    workers = 20
    converter: DataConverter = CONVERTER_CLASS[backend](
        extractors=[    
            I3TruthExtractor(),
            #I3PISAExtractor('I3MMCTrackList'),
            #I3RetroExtractor(),
            #I3FeatureExtractorIceCubeUpgrade("SplitInIcePulsesSRT"),
            I3FeatureExtractorIceCubeUpgrade("SplitInIcePulses"),
            # I3LineFitExtractor(
            #     name="linefit_finitereco_ic",
            #     inputs = inputs_folder,
            #     input_pulses="SplitInIcePulsesSRT",
            #     linefitKey="linefit_improved",
            #     gcd_rescue= gcd_rescue
            # ),
            #I3ParticleExtractor("I3MCTree"),
            #I3FeatureExtractorIceCube86("SplitInIcePulsesSRT"),
            #I3FeatureExtractorIceCubeUpgrade("I3MCTree"),
            #I3FeatureExtractorIceCubeUpgrade("MCCTrackList"),
        ], 
        outdir=outdir,
        workers=workers,
        gcd_rescue=gcd_rescue,
    )
    converted_files = converter(inputs)
    if backend == "sqlite":
        converter.merge_files(converted_files)


if __name__ == "__main__":

    if not has_icecube_package():
        Logger(log_folder=None).error(ERROR_MESSAGE_MISSING_ICETRAY)
    else:
        # Parse command-line arguments
        parser = ArgumentParser(
            description="""
    Convert I3 files to an intermediate format.
    """
        )

        parser.add_argument("backend", choices=["sqlite", "parquet"])
        parser.add_argument(
            "detector", choices=["icecube-86", "icecube-upgrade"]
        )

        args, unknown = parser.parse_known_args()

        # Run example script
        if args.detector == "icecube-86":
            main_icecube86(args.backend)
        else:
            main_icecube_upgrade(args.backend)
