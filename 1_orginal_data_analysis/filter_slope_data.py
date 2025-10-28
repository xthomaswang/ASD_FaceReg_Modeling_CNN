#!/usr/bin/env python3

import pandas as pd
import os
import re
from pathlib import Path

def parse_slope_tuple(slope_str):
    """Parse slope tuple string like '(0.005, 0, 0)' and return the second value"""
    try:
        # Remove parentheses and split by comma
        slope_str = slope_str.strip('()')
        values = [float(x.strip()) for x in slope_str.split(',')]
        return values[1] if len(values) > 1 else None
    except:
        return None

def filter_csv_files():
    """Filter CSV files where slope column second value is 0"""
    
    # Define source and target directories
    source_dir = Path("EIBResult/csv_1000_negslope")
    target_dir = Path("EIBResult/csv_1000_categorical")
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all CSV files in source directory
    csv_files = sorted([f for f in source_dir.glob("trainedACC_1000_*.csv")])
    
    print(f"Found {len(csv_files)} CSV files to process...")
    
    processed_count = 0
    
    for csv_file in csv_files:
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Check if slope column exists
            if 'slope' not in df.columns:
                print(f"Warning: No 'slope' column found in {csv_file.name}")
                continue
            
            # Filter rows where second value in slope tuple is 0
            filtered_rows = []
            for idx, row in df.iterrows():
                slope_value = str(row['slope'])
                second_value = parse_slope_tuple(slope_value)
                
                if second_value == 0.0:
                    filtered_rows.append(row)
            
            # Create filtered dataframe
            if filtered_rows:
                filtered_df = pd.DataFrame(filtered_rows)
                
                # Generate output filename
                output_filename = f"trainedACC_1000_{csv_file.stem.split('_')[-1]}.csv"
                output_path = target_dir / output_filename
                
                # Save filtered data
                filtered_df.to_csv(output_path, index=False)
                
                print(f"Processed {csv_file.name}: {len(filtered_rows)} rows saved to {output_filename}")
                processed_count += 1
            else:
                print(f"No matching rows found in {csv_file.name}")
                
        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")
    
    print(f"\nProcessing complete! {processed_count} files processed.")
    print(f"Output directory: {target_dir.absolute()}")

if __name__ == "__main__":
    # Change to the correct directory
    os.chdir("1_orginal_data_analysis")
    
    # Run the filtering
    filter_csv_files() 