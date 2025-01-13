import os,sys
import numpy as np
import sqlite3
import pandas as pd
pd.set_option('display.max_columns', None)

#file_path = "/groups/icecube/simon/GNN/workspace/data/Converted_I3_file/upgrade_genie_level4_queso_141029_000002.db"
file_path = "/lustre/hpc/project/icecube/MuonGun_upgrade_full_detector_generation_volume_no_kde/130028/merged/merged.db"
#gcd_file_path = "/lustre/hpc/project/icecube/MuonGun_upgrade_full_detector_generation_volume_no_kde/130028/GCD/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V1.i3.bz2"
con = sqlite3.connect(file_path)

cur = con.cursor()

table_list = [a for a in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]

print(table_list)

# Load data from CSV
df_truth = pd.read_sql_query("SELECT * FROM truth", con)
df_splits = pd.read_sql_query("SELECT * FROM SplitInIcePulsesSRT", con)


#Filtert truth data for SubEvent == 0 and for stopped_muon = 1
df_truth = df_truth[df_truth['stopped_muon']==1]
df_truth = df_truth[df_truth['SubEventID']==0]
# Apply the same mask to the split data
df_splits = df_splits[df_splits['event_no'].isin(df_truth['event_no'])]

# Generate a new sqlite database with the filtered data called 'filtered.db'
con_filtered = sqlite3.connect('filtered.db')
df_truth.to_sql('truth', con_filtered)
df_splits.to_sql('SplitInIcePulsesSRT', con_filtered)
con_filtered.commit()
con_filtered.close()
