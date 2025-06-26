# 📊 DOM Charge and RIDE Ratio — Non-Cumulative Binning

This script performs DOM-level aggregation of charge and hit counts across non-overlapping distance bins (from muon track to DOM), using DeepCore/IceCube pulsemap data. It compares per-DOM RIDE values (charge per hit) to a monitor DOM on string 80 and generates detailed heatmaps for charge, hits, and RIDE ratios.

---

## 🧠 What the Script Does

- Loads pulse and truth data from an SQLite `.db` file.
- Filters DOMs based on string ID and DOM number.
- Computes the minimum distance from each DOM to the event muon track.
- Bins DOMs into distance ranges (non-overlapping).
- Aggregates per-DOM total charge and hit count in each bin.
- Computes and visualizes:
  - Charge heatmap
  - Hit heatmap
  - RIDE ratio heatmap vs monitor DOM
  - Per-DOM RIDE relative to monitor DOM (string 80)

---

## 📥 Inputs

- **SQLite database file**  
  Required tables:
  - `truth` with columns: `event_no`, `zenith`, `azimuth`, `position_x/y/z`, `track_length`
  - `SplitInIcePulsesSRT` with columns: `event_no`, `dom_x/y/z`, `string`, `dom_number`, `charge`, `dom_time`, `rde`

---

## ⚙️ Configuration

### DOM Filtering

In `main()`:
pulses_df = pulses_df[
    pulses_df['string'].isin([80, 81, 82, 83, 84, 85]) & 
    (pulses_df['dom_number'] > 10)
]

Distance Binning
Non-cumulative binning:
distance_lower, upper_bounds = define_non_overlapping_bins(0, 160, 10)
Creates bins like: [0–10), [10–20), ..., [150–160)

All plots are saved to the output_dir.
plot_dom_charge_heatmap_non_cumulative
Heatmap of total charge collected per DOM and distance bin.

plot_dom_hit_heatmap_non_cumulative
Heatmap of total hit count per DOM and distance bin.

plot_dom_ride_ratio_heatmap
RIDE = charge / hits (per DOM).
Ratio = (DOM RIDE / string 80 RIDE for same DOM number).
Excludes string 80 DOMs from the final plot.

plot_dom_RIDE_non_cumulative
Same as above but includes monitor string DOMs in the plot for full comparison.

