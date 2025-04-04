import os,sys
import numpy as np
import sqlite3
import pandas as pd
pd.set_option('display.max_columns', None)

#file_path = "/groups/icecube/simon/GNN/workspace/data/Converted_I3_file/upgrade_genie_level4_queso_141029_000002.db"
file_path = "/groups/icecube/simon/GNN/workspace/data/Converted_I3_file/merged/merged/merged.db"
output_db_path = "/groups/icecube/simon/GNN/workspace/data/Converted_I3_file/filtered_all_non_clean.db"

#gcd_file_path = "/lustre/hpc/project/icecube/MuonGun_upgrade_full_detector_generation_volume_no_kde/130028/GCD/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V1.i3.bz2"
if os.path.exists(output_db_path):
    os.remove(output_db_path)

con_in = sqlite3.connect(file_path)
con_out = sqlite3.connect(output_db_path)
#cur = con.cursor()

#table_list = [a for a in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]

#print(table_list)

# Load data from CSV
#df_truth = pd.read_sql_query("SELECT * FROM truth", con)
#df_splits = pd.read_sql_query("SELECT * FROM SplitInIcePulsesSRT", con)


#Filtert truth data for SubEvent == 0 and for stopped_muon = 1
#df_truth = df_truth[df_truth['stopped_muon']==1]
#df_truth = df_truth[df_truth['SubEventID']==0]
# Mask the data to only accept events in which position_x, position_y, and position_z are within 
border_xy = np.array(
    [
        (-256.1400146484375, -521.0800170898438),
        (-132.8000030517578, -501.45001220703125),
        (-9.13000011444092, -481.739990234375),
        (114.38999938964844, -461.989990234375),
        (237.77999877929688, -442.4200134277344),
        (361.0, -422.8299865722656),
        (405.8299865722656, -306.3800048828125),
        (443.6000061035156, -194.16000366210938),
        (500.42999267578125, -58.45000076293945),
        (544.0700073242188, 55.88999938964844),
        (576.3699951171875, 170.9199981689453),
        (505.2699890136719, 257.8800048828125),
        (429.760009765625, 351.0199890136719),
        (338.44000244140625, 463.7200012207031),
        (224.5800018310547, 432.3500061035156),
        (101.04000091552734, 412.7900085449219),
        (22.11000061035156, 509.5),
        (-101.05999755859375, 490.2200012207031),
        (-224.08999633789062, 470.8599853515625),
        (-347.8800048828125, 451.5199890136719),
        (-392.3800048828125, 334.239990234375),
        (-437.0400085449219, 217.8000030517578),
        (-481.6000061035156, 101.38999938964844),
        (-526.6300048828125, -15.60000038146973),
        (-570.9000244140625, -125.13999938964844),
        (-492.42999267578125, -230.16000366210938),
        (-413.4599914550781, -327.2699890136719),
        (-334.79998779296875, -424.5),
    ]
)
border_z = np.array([-500, 524.56])
chunksize = 100000

# --- Process the truth table ---
# Create the new truth table in the filtered DB by processing chunks.
print("Processing truth table...")
for chunk in pd.read_sql_query("SELECT * FROM truth", con_in, chunksize=chunksize):
    # Optionally, also filter by SubEventID if desired:
    chunk = chunk[chunk['SubEventID'] == 0]
    # Apply spatial mask:
    chunk_filtered = chunk[
        (chunk["position_x"] >= border_xy[:, 0].min()) &
        (chunk["position_x"] <= border_xy[:, 0].max()) &
        (chunk["position_y"] >= border_xy[:, 1].min()) &
        (chunk["position_y"] <= border_xy[:, 1].max()) &
        (chunk["position_z"] >= border_z.min()) &
        (chunk["position_z"] <= border_z.max())
    ]
    # Write the filtered chunk to the output DB (append mode creates the table if it doesn't exist)
    chunk_filtered.to_sql('truth', con_out, if_exists='append', index=False)
con_out.commit()

# --- Process the SplitInIcePulsesSRT table ---
print("Processing SplitInIcePulsesSRT table...")
# First, read the filtered event numbers from the newly created truth table.
filtered_events = pd.read_sql_query("SELECT DISTINCT event_no FROM truth", con_out)
filtered_event_nos = filtered_events['event_no'].tolist()

# Process the pulses table in chunks and only keep rows for events in the filtered truth table.
for chunk in pd.read_sql_query("SELECT * FROM SplitInIcePulses", con_in, chunksize=chunksize):
    chunk_filtered = chunk[chunk['event_no'].isin(filtered_event_nos)]
    chunk_filtered.to_sql('SplitInIcePulses', con_out, if_exists='append', index=False)
con_out.commit()

# Close the connections.
con_in.close()
con_out.close()

print(f"Filtered database created at: {output_db_path}")