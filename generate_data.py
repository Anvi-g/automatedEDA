# generate_data.py
import pandas as pd
import numpy as np
import random

def create_messy_data(num_rows=200):
    np.random.seed(42)
    
    # 1. Base Data Generation with more variation
    square_feet = np.random.normal(1500, 800, num_rows).astype(int)  # Increased std dev
    bedrooms = np.random.randint(1, 7, num_rows)  # More bedroom variation
    age_of_house = np.random.randint(0, 60, num_rows)  # More age variation
    neighborhoods = np.random.choice(['Downtown', 'Suburb', 'Rural', 'West End'], num_rows)
    
    # 2. Realistic Price Calculation
    # Base price formula: Price depends on features with realistic relationships
    neighborhood_premiums = {
        'Downtown': 50000,
        'Suburb': 20000,
        'West End': 35000,
        'Rural': 0
    }
    
    # Create price with target correlations using multi-linear approach
    # Start with base price
    price = np.full(num_rows, 200000.0)  # $200k base (float)
    
    # Add feature contributions with appropriate weights for target correlations
    price += square_feet * 30.0          # Modest square feet contribution
    price += bedrooms * 15000.0          # Bedroom contribution  
    price += age_of_house * -1500.0      # Age contribution (negative)
    price += np.vectorize(neighborhood_premiums.get)(neighborhoods).astype(float)
    
    # Add substantial random noise to reduce correlations to realistic levels
    random_component = np.random.normal(0, 100000, num_rows)  # ±$100k random variation
    price += random_component
    
    # Ensure minimum price and round to reasonable values
    price = np.maximum(price, 50000)
    price = price.astype(int)
    
    data = {
        'Square_Feet': square_feet,
        'Bedrooms': bedrooms,
        'Age_of_House': age_of_house,
        'Neighborhood': neighborhoods,
        'Price': price.astype(int)
    }
    
    df = pd.DataFrame(data)
    
    # 2. Injecting "Messiness" (The problems your Agent must fix)
    
    # A. Missing Values (NaNs)
    # Randomly set 10% of Square_Feet and Bedrooms to NaN
    df.loc[df.sample(frac=0.1).index, 'Square_Feet'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'Bedrooms'] = np.nan
    
    # B. Outliers 
    # Add a massive mansion (extreme outlier) - maintain realistic price relationship
    df.loc[0, 'Square_Feet'] = 50000 
    df.loc[0, 'Price'] = df.loc[0, 'Square_Feet'] * 150 + df.loc[0, 'Bedrooms'] * 25000 + 1000000  # Realistic mansion price
    
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
    print("   - Features now have realistic correlations with Price:")
    
    # Show correlations for verification (only numeric columns)
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['Price'].drop('Price')
    for feature, corr in correlations.items():
        print(f"     {feature} vs Price: {corr:.2f}")

if __name__ == "__main__":
    create_messy_data()