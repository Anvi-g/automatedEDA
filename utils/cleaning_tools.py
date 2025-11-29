import pandas as pd
import numpy as np
def standard_cleaning_tool(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Standard cleaning for DataFrames (Memory-based).
    Returns: (cleaned_df, report_string)
    """
    try:
        df = df.copy() # Prevent SettingWithCopy warnings
        report = []

        # 1. Drop Constant Columns
        for col in df.columns:
            if df[col].nunique() <= 1:
                df.drop(columns=[col], inplace=True)
                report.append(f"Dropped constant column '{col}'.")
        
        # 2. Duplicates
        init_len = len(df)
        df = df.drop_duplicates()
        if len(df) < init_len: 
            report.append(f"Dropped {init_len-len(df)} duplicates.")
    
        # 3. NaNs
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if np.issubdtype(df[col].dtype, np.number):
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    report.append(f"Filled NaNs in {col} (Median).")
                else:
                    mode_s = df[col].mode()
                    if not mode_s.empty:
                        mode_val = mode_s[0]
                        df[col] = df[col].fillna(mode_val)
                        report.append(f"Filled NaNs in {col} (Mode).")
        
        # 4. Outliers (Z-score > 3) - Only for Numeric
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].std() > 0: 
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > 3
                if outliers.sum() > 0:
                    df = df[~outliers]
                    report.append(f"Removed {outliers.sum()} outliers from '{col}'.")
        
        report_str = "\n".join(report) if report else "No standard issues found."
        return df, report_str

    except Exception as e:
        return df, f"Error in cleaning: {e}"
def load_training_data() -> pd.DataFrame:
    """
    Robustly loads training data from 'data/processed/'.
    Checks for 'train.csv' AND 'logistic_ready_train.csv'.
    """
    try:
        # Priority 1: Generic Name (New Standard)
        path1 = 'data/processed/train.csv'
        if os.path.exists(path1):
            print(f"Loading data from: {path1}")
            return pd.read_csv(path1)
            
        # Priority 2: Legacy Name (Fallback)
        path2 = 'data/processed/logistic_ready_train.csv'
        if os.path.exists(path2):
            print(f"Loading data from: {path2}")
            return pd.read_csv(path2)
            
        # Failure - Create empty DF to prevent crash, but log error
        print("ERROR: Could not find 'train.csv' OR 'logistic_ready_train.csv'.")
        return pd.DataFrame() 
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()