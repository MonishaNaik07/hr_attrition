import os
import pandas as pd
from generate_and_train import generate_ibm_hr_dataset

def create_synthetic_datasets():
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Generate 3 datasets of 2000 records each
    seeds = [101, 202, 303]
    filenames = []
    
    for i, seed in enumerate(seeds):
        print(f"Generating Synthetic Dataset {i+1} (n=2000, seed={seed})...")
        df = generate_ibm_hr_dataset(n=2000, seed=seed)
        
        # Save to data directory
        filename = f'data/synthetic_hr_data_{i+1}.csv'
        df.to_csv(filename, index=False)
        filenames.append(filename)
        
        # Quick stats
        rate = (df['Attrition'] == 'Yes').mean()
        print(f"  Saved to {filename}")
        print(f"  Attrition Rate: {rate:.1%}")

    print("\nAll 3 synthetic datasets created successfully!")
    print("These files are ready to be uploaded to the AttritionIQ Dashboard.")

if __name__ == '__main__':
    create_synthetic_datasets()
