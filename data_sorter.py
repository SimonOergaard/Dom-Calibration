import os
import numpy as np
import sqlite3
import pandas as pd
from collections import defaultdict

pd.set_option('display.max_columns', None)

# Input and output database paths
file_path = "/groups/icecube/simon/GNN/workspace/data/final_data/merged/merged.db"
output_db_path = "/groups/icecube/simon/GNN/workspace/data/Converted_I3_file/2mill_stopping_muons.db"

# Remove existing output DB if it exists
if os.path.exists(output_db_path):
    os.remove(output_db_path)

# Connect to databases
con_in = sqlite3.connect(file_path)
con_out = sqlite3.connect(output_db_path)

# Define detector boundary (XY polygon and Z range)
border_xy = np.array([
    (-256.14, -521.08), (-132.80, -501.45), (-9.13, -481.74),
    (114.39, -461.99), (237.78, -442.42), (361.0, -422.83),
    (405.83, -306.38), (443.60, -194.16), (500.43, -58.45),
    (544.07, 55.89), (576.37, 170.92), (505.27, 257.88),
    (429.76, 351.02), (338.44, 463.72), (224.58, 432.35),
    (101.04, 412.79), (22.11, 509.5), (-101.06, 490.22),
    (-224.09, 470.86), (-347.88, 451.52), (-392.38, 334.24),
    (-437.04, 217.80), (-481.60, 101.39), (-526.63, -15.60),
    (-570.90, -125.14), (-492.43, -230.16), (-413.46, -327.27),
    (-334.80, -424.5),
])
border_z = np.array([-500, 524.56])
chunksize = 100000

# --- Step 1: Count events with ≥3 pulses on strings ≤ 86 ---
print("Scanning pulses to find valid events (≥3 pulses on strings ≤ 86)...")
event_pulse_counts = defaultdict(int)

for chunk in pd.read_sql_query("SELECT event_no, string FROM SplitInIcePulsesSRT", con_in, chunksize=chunksize):
    chunk_valid = chunk[chunk['string'] <= 93]
    counts = chunk_valid['event_no'].value_counts()
    for event_no, count in counts.items():
        event_pulse_counts[event_no] += count

valid_event_nos = [event_no for event_no, count in event_pulse_counts.items() if count >= 3]
print(f"Found {len(valid_event_nos)} valid events with ≥3 pulses on strings ≤ 86.")

# --- Step 2: Filter and save the truth table ---
print("Filtering and writing the truth table...")
for chunk in pd.read_sql_query("SELECT * FROM truth", con_in, chunksize=chunksize):
    chunk = chunk[chunk['SubEventID'] == 0]
    chunk = chunk[chunk['event_no'].isin(valid_event_nos)]
    chunk_filtered = chunk[
        (chunk["position_x"] >= border_xy[:, 0].min()) &
        (chunk["position_x"] <= border_xy[:, 0].max()) &
        (chunk["position_y"] >= border_xy[:, 1].min()) &
        (chunk["position_y"] <= border_xy[:, 1].max()) &
        (chunk["position_z"] >= border_z.min()) &
        (chunk["position_z"] <= border_z.max())
    ]
    chunk_filtered.to_sql('truth', con_out, if_exists='append', index=False)
con_out.commit()

# --- Step 3: Filter and save the pulses table ---
print("Filtering and writing the SplitInIcePulsesSRT table...")
for chunk in pd.read_sql_query("SELECT * FROM SplitInIcePulsesSRT", con_in, chunksize=chunksize):
    chunk_filtered = chunk[
        (chunk['event_no'].isin(valid_event_nos)) &
        (chunk['string'] <= 93)
    ]
    chunk_filtered.to_sql('SplitInIcePulsesSRT', con_out, if_exists='append', index=False)
con_out.commit()

# --- Finalization ---

print("Creating indexes for faster querying...")
cur = con_out.cursor()

# Index on event_no in truth
cur.execute("CREATE INDEX IF NOT EXISTS idx_truth_event_no ON truth (event_no);")

# Index on event_no in pulses
cur.execute("CREATE INDEX IF NOT EXISTS idx_pulses_event_no ON SplitInIcePulsesSRT (event_no);")

# Optional: Add more if needed (e.g., dom_time or string)
cur.execute("CREATE INDEX IF NOT EXISTS idx_pulses_dom_time ON SplitInIcePulsesSRT (dom_time);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_pulses_string ON SplitInIcePulsesSRT (string);")

con_out.commit()

con_in.close()
con_out.close()
print(f"✅ Filtered database created at: {output_db_path}")
