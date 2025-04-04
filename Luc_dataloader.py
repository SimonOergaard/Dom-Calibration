import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
def feature_preprocessing_pyarrow(table: pa.Table) -> pa.Table:
    """
    Preprocess the input features when creating the dataset.
    Args:
    - table: Input features as a PyArrow Table.
    Returns:
    - Processed features as a PyArrow Table with the same schema as the input.
    """
    def normalize_column(col, scale, shift=0):
        return pc.divide(pc.subtract(col, shift), scale)
    # Initialize empty output table
    updated_columns = []
    for col_name in table.column_names:
        if col_name in ['dom_x', 'dom_y', 'dom_z', 'dom_x_rel', 'dom_y_rel', 'dom_z_rel']:
            updated_columns.append(normalize_column(table[col_name], 500))
        elif col_name in ['rde']:
            updated_columns.append(normalize_column(table[col_name], 0.25, shift=1.25))
        elif col_name in ['pmt_area']:
            updated_columns.append(normalize_column(table[col_name], 0.05))
        elif col_name in ['q1', 'q2', 'q3', 'q4', 'q5', 'Q25', 'Q75', 'Qtotal']:
            mask = pc.greater(table[col_name], 0)
            updated_columns.append(pc.if_else(mask, pc.log10(table[col_name]), table[col_name]))
        elif col_name in ['t1', 't2', 't3','t4', 't5']:
            mask = pc.greater(table[col_name], 0)
            updated_columns.append(pc.if_else(mask, normalize_column(table[col_name], 3.0e04, shift=1.0e04), table[col_name]))
        elif col_name in ['T10', 'T50', 'sigmaT']:
            mask = pc.greater(table[col_name], 0)
            updated_columns.append(pc.if_else(mask, normalize_column(table[col_name], 1.0e04), table[col_name]))
        else:
            updated_columns.append(table[col_name])
    return pa.table(updated_columns, schema=table.schema)
class PMTfiedDatasetPyArrow(Dataset):
    def __init__(
            self,
            truth_paths,
            selection=None,
            transform=feature_preprocessing_pyarrow,
    ):
        '''
        Args:
        - truth_paths: List of paths to the truth files
        - selection: List of event numbers to select from the corresponding truth files
        - transform: Function to apply to the features as preprocessing
        '''
        self.truth_paths = truth_paths
        self.selection = selection
        self.transform = transform
        # Metadata variables
        self.event_counts = []
        self.cumulative_event_counts = []
        self.current_file_idx = None
        self.current_truth = None
        self.current_feature_path = None
        self.current_features = None
        total_events = 0
        # Scan the truth files to get the event counts
        for path in self.truth_paths:
            truth = pq.read_table(path)
            if self.selection is not None:
                mask = pc.is_in(truth['event_no'], value_set=pa.array(self.selection))
                truth = truth.filter(mask)
            n_events = len(truth)
            self.event_counts.append(n_events)
            total_events += n_events
            self.cumulative_event_counts.append(total_events)
        self.total_events = total_events
    def __len__(self):
        return self.total_events
    def __getitem__(self, idx):
        # Find the corresponding file index
        file_idx = np.searchsorted(self.cumulative_event_counts, idx, side='right')
        # Define the truth paths
        truth_path = self.truth_paths[file_idx]
        # Define the local event index
        local_idx = idx if file_idx == 0 else idx - self.cumulative_event_counts[file_idx - 1]
        # Load the truth and apply selection
        if file_idx != self.current_file_idx:
            self.current_file_idx = file_idx
            truth = pq.read_table(truth_path)
            #print("Loaded truth table")
            if self.selection is not None:
                mask = pc.is_in(truth['event_no'], value_set=pa.array(self.selection))
                self.current_truth = truth.filter(mask)
            else:
                self.current_truth = truth
        truth = self.current_truth
        # Get the event details
        event_no = torch.tensor(int(truth.column('event_no')[local_idx].as_py()), dtype=torch.long)
        energy = torch.tensor(truth.column('energy')[local_idx].as_py(), dtype=torch.float32)
        azimuth = torch.tensor(truth.column('azimuth')[local_idx].as_py(), dtype=torch.float32)
        zenith = torch.tensor(truth.column('zenith')[local_idx].as_py(), dtype=torch.float32)
        pid = torch.tensor(truth.column('pid')[local_idx].as_py(), dtype=torch.float32)
        # Calculate a 3D unit-vector from the zenith and azimuth angles
        x_dir = torch.sin(zenith) * torch.cos(azimuth)
        y_dir = torch.sin(zenith) * torch.sin(azimuth)
        z_dir = torch.cos(zenith)
        # Stack to dir3vec tensor
        dir3vec = torch.stack([x_dir, y_dir, z_dir], dim=-1)
        offset = int(truth.column('offset')[local_idx].as_py())
        n_doms = int(truth.column('N_doms')[local_idx].as_py())
        part_no = int(truth.column('part_no')[local_idx].as_py())
        shard_no = int(truth.column('shard_no')[local_idx].as_py())
        # Define the feature path based on the truth path
        feature_path = truth_path.replace('truth_{}.parquet'.format(part_no), '' + str(part_no) + '/PMTfied_{}.parquet'.format(shard_no))
        # x from rows (offset-n_doms) to offset
        start_row = offset - n_doms
        end_row = offset
        # Load the features and apply preprocessing
        if feature_path != self.current_feature_path:
            self.current_feature_path = feature_path
            if self.transform is not None:
                self.current_features = self.transform(pq.read_table(feature_path))
            else:
                self.current_features = pq.read_table(feature_path)
            #print("Loaded feature table")
        features = self.current_features
        # Select the rows corresponding to the event and convert to numpy array
        rows = pa.array(range(start_row, end_row))
        x = features.take(rows).to_pandas().to_numpy(dtype=np.float32)
        # Remove the first two column with event numbers
        x = x[:, 2:]
        x = torch.tensor(x, dtype=torch.float32)
        return Data(x=x, n_doms=n_doms, event_no=event_no, feature_path=feature_path, energy=energy, azimuth=azimuth, zenith=zenith, dir3vec=dir3vec, pid=pid)
    
    
    
import torch
import json
with open('config.json') as f:
    config = json.load(f)
    reconstruction_target = config['reconstruction_target']
def pad_or_truncate(event, max_seq_length=256, total_charge_index=int(16)):
    """
    Pad or truncate an event to the given max sequence length, and create an attention mask.
    Args:
    - event: Tensor of shape (seq_length, feature_dim) where seq_length can vary.
    - max_seq_length: Maximum sequence length to pad/truncate to.
    Returns:
    - Padded or truncated event of shape (max_seq_length, feature_dim).
    - Attention mask of shape (max_seq_length) where 1 indicates a valid token and 0 indicates padding.
    """
    seq_length = event.size(0)
    # Truncate if the sequence is too long
    if seq_length > max_seq_length:
        # sort the event by total charge
        event = event[event[:, total_charge_index].argsort(descending=True)]
        truncated_event = event[:max_seq_length]
        return truncated_event, max_seq_length
    # Pad if the sequence is too short
    elif seq_length < max_seq_length:
        padding = torch.zeros((max_seq_length - seq_length, event.size(1)))
        padded_event = torch.cat([event, padding], dim=0)
        return padded_event,  seq_length
    # No need to pad or truncate if it's already the correct length
    return event, seq_length
def custom_collate_fn(batch, max_seq_length=config['seq_dim']):
    """
    Custom collate function to pad or truncate each event in the batch.
    Args:
    - batch: List of (event, label) tuples where event has a variable length [seq_length, num_features].
    - max_seq_length: The fixed length to pad/truncate each event to (default is 512).
    Returns:
    - A batch of padded/truncated events with shape [batch_size, max_seq_length, num_features].
    - Corresponding labels.
    """
    # Separate events and labels
    events = [item.x for item in batch]  # Each event has shape [seq_length, 7]
    # Pad or truncate each event
    padded_events, event_lengths = zip(*[pad_or_truncate(event, max_seq_length) for event in events])
    # Stack the padded events and masks to form the batch
    batch_events = torch.stack(padded_events)
    event_lengths = torch.tensor(event_lengths)
    # Extract labels and convert to tensors
    label_name = reconstruction_target
    # Extract labels and convert to tensors (3D vectors)
    vectors = [item[label_name] for item in batch]
    # Stack labels in case of multi-dimensional output
    labels = torch.stack(vectors)
    # set to float32
    batch_events = batch_events.float()
    labels = labels.float()
    return batch_events, labels, event_lengths