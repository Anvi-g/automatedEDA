# generate_data.py
import pandas as pd
import numpy as np
import random

def create_messy_data(num_rows=200):
    np.random.seed(42)
    
    # 1. Base Data Generation
    data = {
        'Square_Feet': np.random.normal(1500, 500, num_rows).astype(int),
        'Bedrooms': np.random.randint(1, 6, num_rows),
        'Age_of_House': np.random.randint(0, 50, num_rows),
        'Neighborhood': np.random.choice(['Downtown', 'Suburb', 'Rural', 'West End'], num_rows),
        'Price': np.random.normal(300000, 100000, num_rows) # Target Variable
    }
    
    df = pd.DataFrame(data)
    
    # 2. Injecting "Messiness" (The problems your Agent must fix)
    
    # A. Missing Values (NaNs)
    # Randomly set 10% of Square_Feet and Bedrooms to NaN
    df.loc[df.sample(frac=0.1).index, 'Square_Feet'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'Bedrooms'] = np.nan
    
    # B. Outliers 
    # Add a massive mansion (extreme outlier)
    df.loc[0, 'Square_Feet'] = 50000 
    df.loc[0, 'Price'] = 10000000
    
    # C. Bad Data Types / Noise
    # Make one row have a string in a numeric column (if not handled by NaN)
    # (Optional: usually pandas forces object type, kept simple here for basic regression testing)
    
    # D. Duplicates
    # Duplicate the first 5 rows
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    
    # 3. Save to CSV
    filename = 'housing_data.csv'
    df.to_csv(filename, index=False)
    print(f"✅ '{filename}' created with {len(df)} rows.")
    print("   - Includes NaNs, Outliers, Categorical strings, and Duplicates.")

if __name__ == "__main__":
    create_messy_data()