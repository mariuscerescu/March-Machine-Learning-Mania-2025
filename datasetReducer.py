import pandas as pd
import numpy as np
import os
from typing import List
import random

def reduce_dataset_size(input_file: str, output_file: str, max_rows: int = 10000, 
                       random_seed: int = 42) -> None:
    """
    Reduces the size of a dataset by taking a sample of rows.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the reduced CSV file
        max_rows: Maximum number of rows in the output file
        random_seed: Random seed for reproducibility
    """
    print(f"Processing {input_file}...")
    
    # Check file size
    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    print(f"Original file size: {file_size_mb:.2f} MB")
    
    # Load the dataset
    try:
        df = pd.read_csv(input_file)
        original_rows = len(df)
        print(f"Original row count: {original_rows}")
        
        # If fewer rows than max_rows, keep all rows
        if original_rows <= max_rows:
            print(f"File already has {original_rows} rows, which is below the {max_rows} limit.")
            if input_file != output_file:
                df.to_csv(output_file, index=False)
                print(f"Copied to {output_file}")
            return
        
        # Sample rows
        random.seed(random_seed)
        if 'Season' in df.columns:
            # Stratified sampling by season to preserve recent data
            seasons = df['Season'].unique()
            print(f"Found {len(seasons)} unique seasons")
            
            # Calculate how many rows to take per season
            rows_per_season = max_rows // len(seasons)
            remaining_rows = max_rows % len(seasons)
            
            # Allocate more rows to recent seasons
            sorted_seasons = sorted(seasons)
            allocation = {season: rows_per_season for season in sorted_seasons}
            
            # Distribute remaining rows to most recent seasons
            for season in sorted_seasons[-remaining_rows:]:
                allocation[season] += 1
            
            # Take samples for each season
            sampled_dfs = []
            for season, row_count in allocation.items():
                season_df = df[df['Season'] == season]
                if len(season_df) > row_count:
                    sampled_dfs.append(season_df.sample(row_count, random_state=random_seed))
                else:
                    sampled_dfs.append(season_df)  # Take all rows if fewer than allocated
            
            # Combine samples
            reduced_df = pd.concat(sampled_dfs)
            
        else:
            # Simple random sampling if no Season column
            reduced_df = df.sample(max_rows, random_state=random_seed)
        
        # Save reduced dataset
        reduced_df.to_csv(output_file, index=False)
        
        # Report results
        reduced_rows = len(reduced_df)
        print(f"Reduced to {reduced_rows} rows ({reduced_rows/original_rows:.2%} of original)")
        
        reduced_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"Reduced file size: {reduced_size_mb:.2f} MB ({reduced_size_mb/file_size_mb:.2%} of original)")
        print(f"Saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")

def main():
    # Files to reduce (based on the image)
    large_files = [
        "MMasseyOrdinals.csv",  # ~119 MB
        "MRegularSeasonDetailedResults.csv",  # ~11.5 MB
        "SeedBenchmarkStage1.csv",  # ~10.4 MB
        "SampleSubmissionStage1.csv",  # ~9.9 MB
        "WRegularSeasonDetailedResults.csv",  # ~7.9 MB
        "MRegularSeasonCompactResults.csv",  # ~5.5 MB
        "WRegularSeasonCompactResults.csv",  # ~3.9 MB
        "MGameCities.csv",  # ~2.7 MB
        "WGameCities.csv",  # ~2.6 MB
        "SampleSubmissionStage2.csv"  # ~2.5 MB
    ]
    
    data_dir = "data"
    reduced_dir = "reducedDataset"
    
    # Create reduced directory if it doesn't exist
    if not os.path.exists(reduced_dir):
        os.makedirs(reduced_dir)
        print(f"Created directory: {reduced_dir}")
    
    # Process each file
    for filename in large_files:
        input_path = os.path.join(data_dir, filename)
        output_path = os.path.join(reduced_dir, filename)
        
        # Special handling for massive files
        if filename == "MMasseyOrdinals.csv":
            max_rows = 10000  # Take fewer rows for this very large file
        else:
            max_rows = 10000
            
        reduce_dataset_size(input_path, output_path, max_rows)
        print("-" * 50)
    
    print("Dataset reduction complete!")

if __name__ == "__main__":
    main()
    