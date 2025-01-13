import os
import argparse
from collections import OrderedDict
import numpy as np
from icecube import dataio, dataclasses, icetray
from icecube.icetray import I3Tray
import sqlite3



from graphnet.constants import EXAMPLE_OUTPUT_DIR, TEST_DATA_DIR
from graphnet.data.extractors.icecube import (
    I3FeatureExtractorIceCubeUpgrade,
    I3FeatureExtractorIceCube86,
    I3RetroExtractor,
    I3TruthExtractor,
)
from graphnet.data.dataconverter import DataConverter
from graphnet.data.parquet import ParquetDataConverter
from graphnet.data.sqlite import SQLiteDataConverter
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import Logger

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

def main_icecube_upgrade(backend: str) -> None:
    """Convert IceCube-Upgrade I3 files to intermediate `backend` format."""
    # Check(s)
    assert backend in CONVERTER_CLASS

    inputs = "/groups/icecube/simon/GNN/workspace/data/I3_files/"
    outdir = "/groups/icecube/simon/GNN/workspace/data/Converted_I3_file/"
    gcd_rescue = (
        "/groups/icecube/simon//GNN/workspace/data/GCD_files/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V1.i3.bz2"
    )
    workers = 1

    converter: DataConverter = CONVERTER_CLASS[backend](
        extractors=[
            I3TruthExtractor(),
            I3FeatureExtractorIceCubeUpgrade("SplitInIcePulsesSRT"),
        ],
        outdir=outdir,
        workers=workers,
        gcd_rescue=gcd_rescue,
    )
    converter(inputs)
    if backend == "sqlite":
        converter.merge_files(inputs)


def append_data_or_nan(data_container, key, value):
    """
    Appends a value to the data container; if the value is missing or not found, appends NaN.
    """
    if value is not None:  # Check if value exists and is not None
        try:
            data_container[key].append(np.asarray(value))
        except Exception as e:  # Handles cases where value extraction fails
            print(f"Error appending value for {key}: {e}")
            data_container[key].append(np.NaN)
    else:
        data_container[key].append(np.NaN)

def harvest_quantities(frame, data_container=None):
    # Ensure data_container is a valid type
    assert isinstance(data_container, (dict, OrderedDict))
    
    current_event_id = frame['I3EventHeader'].event_id if frame.Has('I3EventHeader') else None
    data_container['EventID'].append(current_event_id)
    
    primary_extracted = False
    secondary_extracted = False
    #tertiary_extracted = False
    if frame.Has('I3MCTree'):
        primary = frame['I3MCTree'].get_primaries()[0] if frame['I3MCTree'].get_primaries() else None
        
        if primary:
            # Append primary information or NaN if missing
            append_data_or_nan(data_container, 'X_position_Prim', primary.pos.x)
            append_data_or_nan(data_container, 'Y_position_Prim', primary.pos.y)
            append_data_or_nan(data_container, 'Z_position_Prim', primary.pos.z)
            append_data_or_nan(data_container, 'Azimuth_Prim', primary.dir.azimuth)
            append_data_or_nan(data_container, 'Zenith_Prim', primary.dir.zenith)
            append_data_or_nan(data_container, 'Energy_Prim', primary.energy)
            append_data_or_nan(data_container, 'Time_Prim', primary.time)
            # Explicitly set Length_Prim to 300 if not available
            data_container['Length_Prim'].append(300)
            primary_extracted = True
        else:
            # Append NaN to all primary fields if primary is not found
            for key in ['X_position_Prim', 'Y_position_Prim', 'Z_position_Prim', 
                        'Azimuth_Prim', 'Zenith_Prim', 'Energy_Prim', 'Length_Prim', 'Time_Prim']:
                append_data_or_nan(data_container, key, None)
                
        # Process secondary particles only if primary is neutrino (pdg_encoding 14)
        if primary and np.abs(primary.pdg_encoding) == 14:
            
            secondaries = frame["I3MCTree"].get_daughters(primary)
            mu_minus, mu_plus = None, None

            # Separate handling for MuMinus and MuPlus to avoid double counting
            for secondary in secondaries:
                if secondary.pdg_encoding == 13 and mu_minus is None:
                    mu_minus = secondary
                elif secondary.pdg_encoding == -13 and mu_plus is None:
                    mu_plus = secondary

            # Handle MuMinus if present
            if mu_minus:
                append_data_or_nan(data_container, 'X_position_MuMinus', mu_minus.pos.x)
                append_data_or_nan(data_container, 'Y_position_MuMinus', mu_minus.pos.y)
                append_data_or_nan(data_container, 'Z_position_MuMinus', mu_minus.pos.z)
                append_data_or_nan(data_container, 'Azimuth_MuMinus', mu_minus.dir.azimuth)
                append_data_or_nan(data_container, 'Zenith_MuMinus', mu_minus.dir.zenith)
                append_data_or_nan(data_container, 'length_MuMinus', mu_minus.length)
                append_data_or_nan(data_container, 'Energy_MuMinus', mu_minus.energy)
                append_data_or_nan(data_container, 'Time_MuMinus', mu_minus.time)
                secondary_extracted = True
            else:
                for key in ['X_position_MuMinus', 'Y_position_MuMinus', 'Z_position_MuMinus', 
                            'Azimuth_MuMinus', 'Zenith_MuMinus', 'length_MuMinus', 'Energy_MuMinus', 'Time_MuMinus']:
                    append_data_or_nan(data_container, key, None)

            # Handle MuPlus if present
            if mu_plus:
                append_data_or_nan(data_container, 'X_position_MuPlus', mu_plus.pos.x)
                append_data_or_nan(data_container, 'Y_position_MuPlus', mu_plus.pos.y)
                append_data_or_nan(data_container, 'Z_position_MuPlus', mu_plus.pos.z)
                append_data_or_nan(data_container, 'Azimuth_MuPlus', mu_plus.dir.azimuth)
                append_data_or_nan(data_container, 'Zenith_MuPlus', mu_plus.dir.zenith)
                append_data_or_nan(data_container, 'length_MuPlus', mu_plus.length)
                append_data_or_nan(data_container, 'Energy_MuPlus', mu_plus.energy)
                append_data_or_nan(data_container, 'Time_MuPlus', mu_plus.time)
                secondary_extracted = True
            else:
                for key in ['X_position_MuPlus', 'Y_position_MuPlus', 'Z_position_MuPlus', 
                            'Azimuth_MuPlus', 'Zenith_MuPlus', 'length_MuPlus', 'Energy_MuPlus', 'Time_MuPlus']:
                    append_data_or_nan(data_container, key, None)
                    
            if not secondary_extracted:
                for key in ['X_position_MuMinus', 'Y_position_MuMinus', 'Z_position_MuMinus', 
                            'Azimuth_MuMinus', 'Zenith_MuMinus', 'length_MuMinus', 'Energy_MuMinus', 'Time_MuMinus',
                            'X_position_MuPlus', 'Y_position_MuPlus', 'Z_position_MuPlus', 
                            'Azimuth_MuPlus', 'Zenith_MuPlus', 'length_MuPlus', 'Energy_MuPlus', 'Time_MuPlus']:
                    append_data_or_nan(data_container, key, None)
        else:
            # Append NaN if primary isn't neutrino or extraction fails
            for key in ['X_position_MuMinus', 'Y_position_MuMinus', 'Z_position_MuMinus', 
                        'Azimuth_MuMinus', 'Zenith_MuMinus', 'length_MuMinus', 'Energy_MuMinus', 'Time_MuMinus',
                        'X_position_MuPlus', 'Y_position_MuPlus', 'Z_position_MuPlus', 
                        'Azimuth_MuPlus', 'Zenith_MuPlus', 'length_MuPlus', 'Energy_MuPlus', 'Time_MuPlus']:
                append_data_or_nan(data_container, key, None)

        # Process tertiary particles from the daughters of MuMinus and MuPlus
        # tertiary_particles = []
        # if mu_minus:
        #     tertiary_particles += frame["I3MCTree"].get_daughters(mu_minus)
        # if mu_plus:
        #     tertiary_particles += frame["I3MCTree"].get_daughters(mu_plus)

        # particle_mappings = {
        #     'MuMinus': 13, 'MuPlus': -13, 'EPlus': -11, 'EMinus': 11, 
        #     'Gamma': 22, 'NuE': 12, 'NuEBar': -12, 'NuMu': 14, 'NuMuBar': -14
        # }

        # # Extract tertiary particles based on mapping and append data for each occurrence
        # for particle, pdg in particle_mappings.items():
        #     x_pos_list, y_pos_list, z_pos_list = [], [], []
        #     azimuth_list, zenith_list, energy_list = [], [], []
        #     length_list = []  # Handle lengths if applicable

        #     for tertiary in tertiary_particles:
        #         if tertiary.pdg_encoding == pdg:
        #             x_pos_list.append(tertiary.pos.x)
        #             y_pos_list.append(tertiary.pos.y)
        #             z_pos_list.append(tertiary.pos.z)
        #             azimuth_list.append(tertiary.dir.azimuth)
        #             zenith_list.append(tertiary.dir.zenith)
        #             energy_list.append(tertiary.energy)
        #             length_list.append(getattr(tertiary, 'length', float('nan')))
        #             tertiary_extracted = True

        #     # Append all found occurrences or NaNs if none found
        #     if x_pos_list:
        #         for x, y, z, az, zen, en, ln in zip(x_pos_list, y_pos_list, z_pos_list, azimuth_list, zenith_list, energy_list, length_list):
        #             append_data_or_nan(data_container, f'X_position_{particle}', x)
        #             append_data_or_nan(data_container, f'Y_position_{particle}', y)
        #             append_data_or_nan(data_container, f'Z_position_{particle}', z)
        #             append_data_or_nan(data_container, f'Azimuth_{particle}', az)
        #             append_data_or_nan(data_container, f'Zenith_{particle}', zen)
        #             append_data_or_nan(data_container, f'Energy_{particle}', en)
        #             if particle in ['MuMinus', 'MuPlus']:
        #                 append_data_or_nan(data_container, f'length_{particle}', ln)
        #     else:
        #         # Append NaNs for particles not found
        #         for key in [f'X_position_{particle}', f'Y_position_{particle}', f'Z_position_{particle}',
        #                     f'Azimuth_{particle}', f'Zenith_{particle}', f'Energy_{particle}']:
        #             append_data_or_nan(data_container, key, None)
        #         if particle in ['MuMinus', 'MuPlus']:
        #             append_data_or_nan(data_container, f'length_{particle}', None)

        # # Append NaNs if no tertiary particles were extracted
        # if not tertiary_extracted:
        #     for particle in particle_mappings.keys():
        #         for key in [f'X_position_{particle}', f'Y_position_{particle}', f'Z_position_{particle}',
        #                     f'Azimuth_{particle}', f'Zenith_{particle}', f'Energy_{particle}']:
        #             append_data_or_nan(data_container, key, None)
        #         if particle in ['MuMinus', 'MuPlus']:
        #             append_data_or_nan(data_container, f'length_{particle}', None)


                        
def check_container_lengths(data_container):
    """Ensures all lists in the data container have the same length by filling shorter lists with NaN."""
    max_length = max(len(lst) for lst in data_container.values())
    for key, lst in data_container.items(): 
        if len(lst) < max_length:
            # Fill the list to match max length with NaN
            lst.extend([float('nan')] * (max_length - len(lst)))
            #print(f"Extended {key} to match length {max_length}")


        
def Convert_to_Sqlite(data_container, db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Create table primary table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS primary_particle
                    (X_position_Prim REAL, Y_position_Prim REAL, Z_position_Prim REAL, Azimuth_Prim REAL, Zenith_Prim REAL, Energy_Prim REAL, Length_Prim REAL, Time_Prim REAL, Prim_EventID)''')
    # Create table secondary table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS secondary_particle
                    (X_position_MuMinus REAL, Y_position_MuMinus REAL, Z_position_MuMinus REAL, Azimuth_MuMinus REAL, Zenith_MuMinus REAL, Energy_MuMinus REAL, Length_MuMinus REAL, Time_MuMinus REAL,
                    X_position_MuPlus REAL, Y_position_MuPlus REAL, Z_position_MuPlus REAL, Azimuth_MuPlus REAL, Zenith_MuPlus REAL, Energy_MuPlus REAL, Length_MuPlus REAL, Time_MuPlus REAL, EventID
                    )''')
    # Create a tertiary table if not exists for other particles
    # c.execute('''
    #     CREATE TABLE IF NOT EXISTS tertiary_particle (
    #         X_position_EPlus REAL, Y_position_EPlus REAL, Z_position_EPlus REAL, 
    #         Azimuth_EPlus REAL, Zenith_EPlus REAL, Energy_EPlus REAL,
    #         X_position_EMinus REAL, Y_position_EMinus REAL, Z_position_EMinus REAL, 
    #         Azimuth_EMinus REAL, Zenith_EMinus REAL, Energy_EMinus REAL,
    #         X_position_Gamma REAL, Y_position_Gamma REAL, Z_position_Gamma REAL, 
    #         Azimuth_Gamma REAL, Zenith_Gamma REAL, Energy_Gamma REAL,
    #         X_position_NuE REAL, Y_position_NuE REAL, Z_position_NuE REAL, 
    #         Azimuth_NuE REAL, Zenith_NuE REAL, Energy_NuE REAL,
    #         X_position_NuEBar REAL, Y_position_NuEBar REAL, Z_position_NuEBar REAL, 
    #         Azimuth_NuEBar REAL, Zenith_NuEBar REAL, Energy_NuEBar REAL,
    #         X_position_NuMu REAL, Y_position_NuMu REAL, Z_position_NuMu REAL, 
    #         Azimuth_NuMu REAL, Zenith_NuMu REAL, Energy_NuMu REAL,
    #         X_position_NuMuBar REAL, Y_position_NuMuBar REAL, Z_position_NuMuBar REAL, 
    #         Azimuth_NuMuBar REAL, Zenith_NuMuBar REAL, Energy_NuMuBar REAL,
    #         X_position_MuMinus REAL, Y_position_MuMinus REAL, Z_position_MuMinus REAL, 
    #         Azimuth_MuMinus REAL, Zenith_MuMinus REAL, Energy_MuMinus REAL, Length_MuMinus REAL,
    #         X_position_MuPlus REAL, Y_position_MuPlus REAL, Z_position_MuPlus REAL, 
    #         Azimuth_MuPlus REAL, Zenith_MuPlus REAL, Energy_MuPlus REAL, Length_MuPlus REAL, EventID
    #     )
    # ''')
    def safe_float_conversion(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float('nan')
    # Insert data into primary table
    for i in range(len(data_container['X_position_Prim'])):
        
        c.execute('''
            INSERT INTO primary_particle VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            safe_float_conversion(data_container['X_position_Prim'][i]), safe_float_conversion(data_container['Y_position_Prim'][i]),
            safe_float_conversion(data_container['Z_position_Prim'][i]), safe_float_conversion(data_container['Azimuth_Prim'][i]),
            safe_float_conversion(data_container['Zenith_Prim'][i]), safe_float_conversion(data_container['Energy_Prim'][i]),
            safe_float_conversion(data_container['Length_Prim'][i]), safe_float_conversion(data_container['Time_Prim'][i]) ,safe_float_conversion(data_container['EventID'][i])
        ))

    # Insert data into secondary table
    for i in range(len(data_container['X_position_MuMinus'])):
        #print(data_container['X_position_MuMinus'][i])
        c.execute('''
            INSERT INTO secondary_particle VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            safe_float_conversion(data_container['X_position_MuMinus'][i]), safe_float_conversion(data_container['Y_position_MuMinus'][i]),
            safe_float_conversion(data_container['Z_position_MuMinus'][i]), safe_float_conversion(data_container['Azimuth_MuMinus'][i]),
            safe_float_conversion(data_container['Zenith_MuMinus'][i]), safe_float_conversion(data_container['Energy_MuMinus'][i]),
            safe_float_conversion(data_container['length_MuMinus'][i]),safe_float_conversion(data_container['Time_MuMinus'][i]), safe_float_conversion(data_container['X_position_MuPlus'][i]),
            safe_float_conversion(data_container['Y_position_MuPlus'][i]), safe_float_conversion(data_container['Z_position_MuPlus'][i]),
            safe_float_conversion(data_container['Azimuth_MuPlus'][i]), safe_float_conversion(data_container['Zenith_MuPlus'][i]),
            safe_float_conversion(data_container['Energy_MuPlus'][i]), safe_float_conversion(data_container['length_MuPlus'][i]), safe_float_conversion(data_container['Time_MuPlus'][i]), safe_float_conversion(data_container['EventID'][i])
        ))

    # Insert data into tertiary table
    # for i in range(len(data_container['X_position_EPlus'])):
        
    #     c.execute('''
    #         INSERT INTO tertiary_particle VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? ,?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    #     ''', (
    #         safe_float_conversion(data_container['X_position_EPlus'][i]), safe_float_conversion(data_container['Y_position_EPlus'][i]), safe_float_conversion(data_container['Z_position_EPlus'][i]),
    #         safe_float_conversion(data_container['Azimuth_EPlus'][i]), safe_float_conversion(data_container['Zenith_EPlus'][i]), safe_float_conversion(data_container['Energy_EPlus'][i]),
    #         safe_float_conversion(data_container['X_position_EMinus'][i]), safe_float_conversion(data_container['Y_position_EMinus'][i]), safe_float_conversion(data_container['Z_position_EMinus'][i]),
    #         safe_float_conversion(data_container['Azimuth_EMinus'][i]), safe_float_conversion(data_container['Zenith_EMinus'][i]), safe_float_conversion(data_container['Energy_EMinus'][i]),
    #         safe_float_conversion(data_container['X_position_Gamma'][i]), safe_float_conversion(data_container['Y_position_Gamma'][i]), safe_float_conversion(data_container['Z_position_Gamma'][i]),
    #         safe_float_conversion(data_container['Azimuth_Gamma'][i]), safe_float_conversion(data_container['Zenith_Gamma'][i]), safe_float_conversion(data_container['Energy_Gamma'][i]),
    #         safe_float_conversion(data_container['X_position_NuE'][i]), safe_float_conversion(data_container['Y_position_NuE'][i]), safe_float_conversion(data_container['Z_position_NuE'][i]),
    #         safe_float_conversion(data_container['Azimuth_NuE'][i]), safe_float_conversion(data_container['Zenith_NuE'][i]), safe_float_conversion(data_container['Energy_NuE'][i]),
    #         safe_float_conversion(data_container['X_position_NuEBar'][i]), safe_float_conversion(data_container['Y_position_NuEBar'][i]), safe_float_conversion(data_container['Z_position_NuEBar'][i]),
    #         safe_float_conversion(data_container['Azimuth_NuEBar'][i]), safe_float_conversion(data_container['Zenith_NuEBar'][i]), safe_float_conversion(data_container['Energy_NuEBar'][i]),
    #         safe_float_conversion(data_container['X_position_NuMu'][i]), safe_float_conversion(data_container['Y_position_NuMu'][i]), safe_float_conversion(data_container['Z_position_NuMu'][i]),
    #         safe_float_conversion(data_container['Azimuth_NuMu'][i]), safe_float_conversion(data_container['Zenith_NuMu'][i]), safe_float_conversion(data_container['Energy_NuMu'][i]),
    #         safe_float_conversion(data_container['X_position_NuMuBar'][i]), safe_float_conversion(data_container['Y_position_NuMuBar'][i]), safe_float_conversion(data_container['Z_position_NuMuBar'][i]),
    #         safe_float_conversion(data_container['Azimuth_NuMuBar'][i]), safe_float_conversion(data_container['Zenith_NuMuBar'][i]), safe_float_conversion(data_container['Energy_NuMuBar'][i]),
    #         safe_float_conversion(data_container['X_position_MuMinus'][i]), safe_float_conversion(data_container['Y_position_MuMinus'][i]), safe_float_conversion(data_container['Z_position_MuMinus'][i]),
    #         safe_float_conversion(data_container['Azimuth_MuMinus'][i]), safe_float_conversion(data_container['Zenith_MuMinus'][i]), safe_float_conversion(data_container['Energy_MuMinus'][i]),
    #         safe_float_conversion(data_container['length_MuMinus'][i]), safe_float_conversion(data_container['X_position_MuPlus'][i]), safe_float_conversion(data_container['Y_position_MuPlus'][i]),
    #         safe_float_conversion(data_container['Z_position_MuPlus'][i]), safe_float_conversion(data_container['Azimuth_MuPlus'][i]), safe_float_conversion(data_container['Zenith_MuPlus'][i]),
    #         safe_float_conversion(data_container['Energy_MuPlus'][i]), safe_float_conversion(data_container['length_MuPlus'][i]), safe_float_conversion(data_container['EventID'][i])
    #     ))

    conn.commit()
    conn.close()
    
            
        
        
        
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
            "detector", choices=[ "icecube-upgrade"]
        )
        parser.add_argument('-i', '--input', help='Input I3 file', type=str, default='/groups/icecube/simon/GNN/workspace/data/I3_files/upgrade_genie_level4_queso_141029_000002.i3.zst')
        parser.add_argument('-o', '--output', help='Name of the output SQLite file', default='/groups/icecube/simon/GNN/workspace/data/Converted_I3_file/upgrade_genie_level4_queso_141029_000002.db')
        
        args = parser.parse_args()
        if args.detector == "icecube-upgrade":
            main_icecube_upgrade(args.backend)
            
            
        container = OrderedDict()
        container['X_position_Prim'] = []
        container['Y_position_Prim'] = []
        container['Z_position_Prim'] = []
        container['Azimuth_Prim'] = []
        container['Zenith_Prim'] = []
        container['Energy_Prim'] = []
        container['Length_Prim'] = []
        container['Time_Prim'] = []
        

        for key in ['MuMinus', 'MuPlus', 'EPlus', 'EMinus', 'Gamma', 'NuE', 'NuEBar', 'NuMu', 'NuMuBar']:
            for subkey in ['X_position', 'Y_position', 'Z_position', 'Azimuth', 'Zenith', 'Energy', 'length', 'Time']:
                container[f'{subkey}_{key}'] = []
            if key in ['MuMinus', 'MuPlus']:
                container[f'length_{key}'] = []
        container['EventID'] = []
                
        tray = I3Tray()
        
        tray.AddModule('I3Reader', 'read_stuff', FilenameList=[args.input])
        tray.AddModule(harvest_quantities, 'harvest_quantities', data_container=container, Streams = [icetray.I3Frame.Physics])
        tray.Execute()
        # Ensure consistent list lengths in the data container
        check_container_lengths(container)
        
        Convert_to_Sqlite(container, args.output)
        