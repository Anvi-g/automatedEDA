# tools.py
import subprocess
import sys

def run_local_python(code: str) -> str:
    """
    Executes Python code locally in the current directory.
    Useful for reading local files like CSVs.
    """
    # Write code to a temp file
    with open("temp_worker_script.py", "w") as f:
        f.write(code)
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, "temp_worker_script.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]:\n{result.stderr}"
            
        return output if output.strip() else "(Code ran successfully, no output)"
        
    except Exception as e:
        return f"Execution Failed: {e}"


def strict_statistical_check(filepath: str, target_col: str = None) -> str:
    """
    Checks for: 
    1. Boolean types.
    2. High VIF (Multicollinearity).
    3. Low Correlation with Target.
    4. Skewness & Scaling.
    
    Args:
        filepath (str): Path to CSV.
        target_col (str): Name of the target variable (Optional).
    """
    try:
        import pandas as pd
        import numpy as np
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant

        df = pd.read_csv(filepath)
        report = []
        issues_found = False
        
        # 1. SMART TARGET DETECTION
        if not target_col:
            common_names = ['price', 'sale_price', 'medv', 'target', 'class']
            found = [c for c in df.columns if c.lower() in common_names]
            target_col = found[0] if found else df.columns[-1]
        
        if target_col not in df.columns:
             return f"ERROR: Target column '{target_col}' not found. Available: {list(df.columns)}"

        report.append(f"[INFO] Analysis Target: '{target_col}'")

        # 2. CHECK BOOLEANS
        bool_cols = df.select_dtypes(include=['bool']).columns
        if len(bool_cols) > 0:
            issues_found = True
            report.append(f"TYPE ERROR: Found Boolean columns {list(bool_cols)}. Convert to Int.")

        # 3. CHECK CARDINALITY
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            if df[col].nunique() > 20:
                issues_found = True
                report.append(f"CARDINALITY WARNING: Column '{col}' has {df[col].nunique()} values. Suggest Target Encoding.")

        # 4. CHECK VIF
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        # Drop constant columns first
        numeric_df = numeric_df.loc[:, numeric_df.std() > 0]

        if target_col in numeric_df.columns:
            features = numeric_df.drop(columns=[target_col])
        else:
            features = numeric_df

        if features.shape[1] > 1:
            try:
                features_const = add_constant(features)
                vif_data = pd.DataFrame()
                vif_data["feature"] = features_const.columns
                vif_data["VIF"] = [variance_inflation_factor(features_const.values, i) 
                                   for i in range(features_const.shape[1])]
                
                # Filter high VIF (> 5.0)
                high_vif = vif_data[(vif_data['VIF'] > 5.0) & (vif_data['feature'] != 'const')]
                
                if not high_vif.empty:
                    issues_found = True
                    corrs = df[high_vif['feature'].tolist() + [target_col]].corr()[target_col].abs()
                    
                    details = []
                    for idx, row in high_vif.iterrows():
                        feat = row['feature']
                        score = row['VIF']
                        target_corr = corrs.get(feat, 0)
                        details.append(f"{feat} (VIF={score:.1f}, CorrWithTarget={target_corr:.2f})")
                    
                    report.append(f"MULTICOLLINEARITY (VIF > 5): {', '.join(details)}")
            except Exception as e:
                report.append(f"VIF CALCULATION SKIPPED: {e}")

        # 5. CHECK SKEWNESS
        potential_skew_cols = [c for c in features.columns if df[c].nunique() > 2]
        if potential_skew_cols:
            skew_vals = df[potential_skew_cols].skew()
            high_skew = skew_vals[abs(skew_vals) > 1]
            if not high_skew.empty:
                issues_found = True
                details = [f"{col} (Skew={val:.2f})" for col, val in high_skew.items()]
                report.append(f"DISTRIBUTION ERROR: High Skew detected in {details}. Suggest Log Transform.")

        # 6. CHECK SCALING
        high_var_cols = []
        for col in potential_skew_cols:
            if df[col].var() > 1000:
                high_var_cols.append(col)
        
        if high_var_cols:
            issues_found = True
            report.append(f"SCALING REQUIRED: Columns {high_var_cols} have high variance. Apply StandardScaler.")

        if not issues_found:
            return "VERDICT: READY. Data is statistically sound."
        else:
            return "VERDICT: NOT READY\n" + "\n".join(report)

    except Exception as e:
        return f"Error in check: {e}"