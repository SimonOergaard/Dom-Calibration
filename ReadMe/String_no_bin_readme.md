# 📊 DOM Hit & Charge Analysis Script

This script processes pulse and track-level data from an IceCube SQLite database to analyze DOM activity relative to muon track distance. It produces a suite of heatmaps and line plots showing DOM-wise hit counts and total charge over distance bins.

---

## 🧠 What the Script Does

- Loads data from a `.db` file:
  - Track data from the `truth` table
  - DOM-level pulse data from the `SplitInIcePulsesSRT` table
- Computes the minimum distance between each DOM and the muon track
- Aggregates charge and hit counts for each DOM within defined distance bins
- Separates DOMs into two categories:
  - **Monitor string** (default: string 80)
  - **Comparison string(s)** (e.g. string 79)
- Supports multiprocessing for scalable performance
- Generates various scatter plots and histograms for varification of data and method.

---

## 📥 Inputs

### SQLite database

Must contain:
- A `truth` table with event-level track parameters:
  - `zenith`, `azimuth`, `position_x/y/z`, `event_no`
- A `SplitInIcePulsesSRT` table with pulse-level data:
  - `dom_x/y/z`, `charge`, `string`, `dom_number`, `event_no`, `rde`,`dom_time` etc.

Modify the database path at the top of the script:
```python
file_path = '/groups/icecube/simon/GNN/workspace/data/Converted_I3_file/2mill_stopping_muons.db'
