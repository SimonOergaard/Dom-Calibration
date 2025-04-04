import pandas as pd
import os

def merge_new_predictions(existing_csv_path, new_predictions_df, output_csv_path):
    """
    Merges new prediction results into an existing CSV file based on the 'event_no' key.
    
    Both the existing CSV and new_predictions_df must contain the 'event_no' column.
    If any prediction columns exist in both, the new values will overwrite the old ones.
    
    Args:
        existing_csv_path (str): Path to the existing CSV file.
        new_predictions_df (pd.DataFrame): DataFrame with new prediction columns.
            Must include 'event_no' as one of its columns.
        output_csv_path (str): Path where the updated CSV will be saved.
    """
    # Load the existing CSV
    df_existing = pd.read_csv(existing_csv_path)
    
    # Merge on event_no using a left join so that all original rows are preserved.
    df_merged = pd.merge(df_existing, new_predictions_df, on="event_no", how="left", suffixes=("", "_new"))
    
    # For any columns coming from the new predictions, update the existing ones.
    for col in new_predictions_df.columns:
        if col == "event_no":
            continue
        new_col = col + "_new"
        if new_col in df_merged.columns:
            # Overwrite the original column with the new values.
            df_merged[col] = df_merged[new_col]
            df_merged.drop(columns=[new_col], inplace=True)
    
    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Save the merged DataFrame to the output CSV.
    df_merged.to_csv(output_csv_path, index=False)
    print(f"Updated CSV file saved to: {output_csv_path}")


def main():
    # Define the paths to the existing CSV and the new predictions.
    existing_csv_path = "data/predictions.csv"
    new_predictions_path = "data/new_predictions.csv"
    
    # Load the new predictions DataFrame.
    new_predictions_df = pd.read_csv(new_predictions_path)
    
    # Define the path for the updated CSV output.
    output_csv_path = "data/updated_predictions.csv"
    
    # Merge the new predictions into the existing CSV.
    merge_new_predictions(
    existing_csv_path=f"{output_dir}/predictions.csv",
    new_predictions_df=predictions_direction,
    output_csv_path=f"{output_dir}/predictions_updated.csv"
    )

if __name__ == "__main__":
    main()