import os
import sqlite3
import pandas as pd
from tqdm import tqdm

# --- CONFIG ---
input_db_path = "/groups/icecube/simon/GNN/workspace/data/Converted_I3_file/merged/merged/merged.db"
csv_filter_path = "/groups/icecube/simon/GNN/workspace/storage/Training/stopped_through_classification/pid_deployment_model_prediction_burnsample_data_RD.csv"
output_db_path = "/groups/icecube/simon/GNN/workspace/storage/Training/stopped_through_classification/train_model_without_configs/muon_sorted.db"
threshold = 0.8
chunksize = 100_000

# --- CLEANUP ---
if os.path.exists(output_db_path):
    os.remove(output_db_path)

# --- LOAD EVENT FILTER ---
print("Loading filtered event list from CSV...")
predictions = pd.read_csv(csv_filter_path)
filtered_event_nos = predictions.query("pid_muon_pred > @threshold")["event_no"].unique().tolist()
print(f"Number of events passing threshold: {len(filtered_event_nos)}")

# --- SETUP DATABASE CONNECTIONS ---
con_in = sqlite3.connect(input_db_path)
con_out = sqlite3.connect(output_db_path)

# --- COPY TABLES ---
print("Copying filtered tables to new DB...")

def copy_filtered_table(table_name, event_column="event_no"):
    for chunk in pd.read_sql_query(f"SELECT * FROM {table_name}", con_in, chunksize=chunksize):
        chunk_filtered = chunk[chunk[event_column].isin(filtered_event_nos)]
        if not chunk_filtered.empty:
            chunk_filtered.to_sql(table_name, con_out, if_exists="append", index=False)

# Get all table names
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", con_in)['name'].tolist()

# Filter each table that has an event_no column
for table in tqdm(tables, desc="Filtering tables"):
    try:
        # Check if event_no is in the table
        df_head = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 1", con_in)
        if "event_no" in df_head.columns:
            copy_filtered_table(table)
        else:
            # Copy full table if no event_no (e.g., geometry or metadata)
            full_table = pd.read_sql_query(f"SELECT * FROM {table}", con_in)
            full_table.to_sql(table, con_out, if_exists="replace", index=False)
    except Exception as e:
        print(f"Skipping table {table} due to error: {e}")

# --- ADD INDEXES ---
print("Creating indexes...")
cur = con_out.cursor()
index_tables = ["SRTInIcePulses"]
for table in index_tables:
    try:
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_event_no ON {table} (event_no);")
    except Exception as e:
        print(f"Could not index table {table}: {e}")
con_out.commit()

# --- CLEANUP ---
con_in.close()
con_out.close()
print(f"✅ Filtered database written to: {output_db_path}")
