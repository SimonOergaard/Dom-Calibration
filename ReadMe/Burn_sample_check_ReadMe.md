# 📊 DOM Charge Analysis & Comparison

This script performs **DOM-level charge aggregation and comparison** between multiple IceCube datasets. It supports internal dataset splitting, MC-vs-RD comparisons, and RDE-based marker classification. The output includes visualizations such as charge ratios, double ratios, and per-DOM scatter plots.

---

## 🧠 What the Script Does

- Loads DOM pulse data from multiple SQLite `.db` files (MC, RD1, RD2).
- Aggregates total charge per DOM (`dom_x`, `dom_y`, `dom_z`).
- Classifies DOMs by string using coordinates.
- Computes and visualizes:
  - **Total charge vs depth**
  - **MC / RD charge ratio per DOM**
  - **Double ratios**: e.g. `(MC_a / MC_b) / (RD_a / RD_b)`
  - **Internal split ratios**: using random splits within a single RD dataset
  - **Marker categories** based on RDE combinations (NQE/HQE)

---

## 📥 Inputs

### SQLite Databases:
- `SplitInIcePulsesSRT`: MC pulsemap table
- `SRTInIcePulses` and `SplitInIcePulses`: RD pulsemap tables
- `truth`: table for event-level info (e.g., zenith, energy)

### Pulsemap Schema:
The script assumes the following columns exist:
- `dom_x`, `dom_y`, `dom_z`
- `charge`, `event_no`, `rde`

---

## 🔧 Key Functions

### Charge Aggregation:
- `process_chunk`, `process_chunk_1`: standard groupby-based aggregation
- `aggregate_charge_without_groupby`: NumPy-based aggregation for speed

### DOM Classification:
- `classify_dom_strings(dom_x, dom_y)`: uses DOM (x, y) coordinates to assign string labels

---

## 📈 Plotting Functions

### `plot_scatter_metrics`
Visualizes total charge vs. DOM depth, grouped by reconstructed string and marked by RDE (>1.0 = `^`, else `o`).

### `plot_predictions`
Plots:
- Histogram of position_z
- 2D histogram of zenith vs energy

### `plot_charge_ratio`
Plots `MC / RD` charge ratio per DOM vs depth. Points are color-coded by string and marked by RDE.

### `plot_raw_data_ratio`
Compares two RD datasets: 
(RD1_a / RD1_b) / (RD2_a / RD2_b)

Markers show RDE combinations:
- `o`: NQE–NQE
- `s`: NQE–HQE
- `^`: HQE–NQE
- `D`: HQE–HQE

### `plot_internal_split_charge_ratio`
Same logic as `plot_raw_data_ratio` but splits a single RD dataset randomly (e.g., 50/50) to check internal consistency.

### `plot_double_charge_ratio`
Calculates:
(MC_a / MC_b) / (RD_a / RD_b)
and plots the double ratio vs DOM number with marker classification.

---

## ⚙️ Configuration

### String Combinations
Inside `main()`:
combinations = [(79, 80), (81, 86), (79, 85), (80, 83), (83, 85), (82, 86), (84, 85)]
file_path = "/path/to/MC.db"
file_path_Raw = "/path/to/RD1.db"
file_path_panos = "/path/to/RD2.db"


