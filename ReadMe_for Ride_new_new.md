# 📊 DOM Calibration — Plotting Functions

This README explains the plotting functions used in the DOM calibration analysis pipeline. These functions generate heatmaps and line plots to help visualize charge, hit counts, and RIDE ratios across distance bins and DOM numbers.

---

## 🔧 How to Change the Monitor String (Default: String 80)

The script compares DOMs on other strings to those on a **monitor string**, typically String 80.

To **change the monitor string**, edit the following line inside the `process_event_chunk` function:

monitor_id = 80  # Default is String 80.

In the main function it possible to define datasets and filter them according to pulsemaps.
i.e. 
    # pulses_df = pulses_df[
    # pulses_df['string'].isin([79, 81, 82, 83, 84, 85, 86]) & 
    # (pulses_df['dom_number'] > 10)
    # ]
where pulses_df is a pulsemap and string & dom_number are columns in the pulsemap.

In the script there are various plotting functions.  
Of particular note are:

- `plot_ride_vs_distance_for_selected_doms_separate_errors
- `plot_recomputed_ride_ratio

---

### `plot_ride_vs_distance_for_selected_doms_separate_errors(grouped_df, output_dir, selected_doms)`

**Description**:  
This function creates **two separate line plots** showing how the median RIDE ratio (String X / monitor string) changes with increasing distance upper bounds for a list of selected DOM numbers.

The two error types plotted are:

1. **Propagated error** — from uncertainties in RIDE calculation per DOM
2. **Sample variance error** — from spread across repeated DOMs

**Inputs**:
- grouped_df: A DataFrame returned by compute_grouped_statistics_with_sample_variance
- output_dir: Folder to save the plots
- selected_doms: List of DOM numbers (integers) to visualize

**Outputs**:
- ride_vs_distance_propagated_errors_79.png: One line per DOM with propagated errors
- ride_vs_distance_sample_variance_errors.png: One line per DOM with sample variance errors

Each line represents a single DOM across all distance bins. Error bars help assess statistical and systematic uncertainty.

---

### `plot_recomputed_ride_ratio(hits_79_by_bin, chg_79_by_bin, hits_80_by_bin, chg_80_by_bin, output_dir, filename="dom_ratio_recomputed_79_80.png")`

**Description**:  
This function **recomputes the RIDE ratio (charge/hit)** for each DOM and distance bin directly from the aggregated data, **bypassing the original `ratio_all` dictionary**.

It is useful for validating or double-checking the primary ratio calculation logic using only raw counts and charges.

**Inputs**:
- hits_79_by_bin: Dictionary of hit counts per DOM on String X by distance bin
- chg_79_by_bin: Corresponding charge values
- hits_80_by_bin: Hit counts for the monitor string (e.g., String 80)
- chg_80_by_bin: Corresponding charges for monitor string
- output_dir: Output folder
- filename: Optional custom filename for the heatmap

**Output**:
- A heatmap image showing recalculated RIDE ratios per DOM per distance bin
- Default filename: dom_ratio_recomputed_79_80.png

Each cell in the heatmap shows (charge/hit)_X / (charge/hit)_monitor. This allows for cross-checking the consistency of the ratio calculations.

---
