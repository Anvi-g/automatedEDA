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


def strict_statistical_check(filepath: str) -> str:
    """
    Checks for: 
    1. Boolean types.
    2. High Cardinality.
    3. High VIF (Multicollinearity) - NOW ROBUST TO CONSTANT COLUMNS.
    4. Skewness & Scaling.
    """
    try:
        import pandas as pd
        import numpy as np
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant

        df = pd.read_csv(filepath)
        report = []
        issues_found = False
        
        # Define Target
        if not target_col:
            common_names = ['price', 'sale_price', 'medv', 'target', 'class']
            found = [c for c in df.columns if c.lower() in common_names]
            target_col = found[0] if found else df.columns[-1]

        # --- NEW: TARGET LEAKAGE CHECK ---
        # Features with > 95% correlation to the target are suspicious.
        # They are usually derived from the target (e.g., Tax_Rate = Tax/Price).
        if target_col in df.columns:
            numeric_df = df.select_dtypes(include=[np.number])
            corrs = numeric_df.corr()[target_col].abs().drop(target_col, errors='ignore')
            
            # Threshold: 0.95 (Suspiciously High)
            leakage_cols = corrs[corrs > 0.95].index.tolist()
            
            if leakage_cols:
                issues_found = True
                report.append(f"CRITICAL LEAKAGE ALERT: Features {leakage_cols} have >95% correlation with target. They likely contain the answer. DROP THEM.")

        # 1. CHECK BOOLEANS
        bool_cols = df.select_dtypes(include=['bool']).columns
        if len(bool_cols) > 0:
            issues_found = True
            report.append(f"TYPE ERROR: Found Boolean columns {list(bool_cols)}. Convert to Int.")

        # 2. CHECK PAIRWISE CORRELATION (Threshold 0.80)
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        if target_col in numeric_df.columns:
            features = numeric_df.drop(columns=[target_col])
            target_series = numeric_df[target_col]
        else:
            features = numeric_df
            target_series = None

        if not features.empty:
            corr_matrix = features.corr().abs()
            # Select upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find Index & Column pairs with correlation > 0.80
            high_corr_pairs = []
            cols_to_check = [c for c in upper.columns if any(upper[c] > 0.80)]
            
            for col in cols_to_check:
                # Get rows with high corr
                rows = upper.index[upper[col] > 0.80]
                for row in rows:
                    val = upper.loc[row, col]
                    # Get Target Corr for decision making
                    t_corr1 = df[row].corr(df[target_col]) if target_col in df else 0
                    t_corr2 = df[col].corr(df[target_col]) if target_col in df else 0
                    
                    pair_info = f"{row}({t_corr1:.2f}) vs {col}({t_corr2:.2f}) [Corr={val:.2f}]"
                    high_corr_pairs.append(pair_info)

            if high_corr_pairs:
                issues_found = True
                report.append(f"HIGH CORRELATION (>0.80): Found redundant pairs.\n   Details: {', '.join(high_corr_pairs)}\n   ACTION: Drop feature with lower target correlation.")

        # 3. CHECK CARDINALITY
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            if df[col].nunique() > 50:
                issues_found = True
                report.append(f"CARDINALITY WARNING: Column '{col}' has {df[col].nunique()} values. Suggest Target Encoding.")

        # 4. CHECK VIF (Updated for Stability)
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        # --- FIX: Drop Constant Columns (Zero Variance) before VIF ---
        # Constant columns cause division by zero in VIF calculation
        non_constant_cols = [c for c in numeric_df.columns if numeric_df[c].std() > 0]
        numeric_df = numeric_df[non_constant_cols]

        if target_col in numeric_df.columns:
            features = numeric_df.drop(columns=[target_col])
        else:
            features = numeric_df

        if features.shape[1] > 1:
            try:
                # Add constant for intercept
                features_const = add_constant(features)
                
                vif_data = pd.DataFrame()
                vif_data["feature"] = features_const.columns
                
                # Calculate VIF safely
                vif_vals = []
                for i in range(features_const.shape[1]):
                    try:
                        val = variance_inflation_factor(features_const.values, i)
                        vif_vals.append(val)
                    except Exception:
                        vif_vals.append(float('inf')) # Handle infinite VIF

                vif_data["VIF"] = vif_vals
                
                # Filter high VIF (> 10), ignore constant intercept
                high_vif = vif_data[(vif_data['VIF'] > 5) & (vif_data['feature'] != 'const')]
                
                if not high_vif.empty:
                    issues_found = True
                    # Calculate correlation with target
                    corrs = df[high_vif['feature'].tolist() + [target_col]].corr()[target_col].abs()
                    
                    details = []
                    for idx, row in high_vif.iterrows():
                        feat = row['feature']
                        score = row['VIF']
                        target_corr = corrs.get(feat, 0)
                        score_str = "INF" if score == float('inf') else f"{score:.1f}"
                        details.append(f"{feat} (VIF={score_str}, CorrWithTarget={target_corr:.2f})")
                    
                    report.append(f"MULTICOLLINEARITY (High VIF): Found features with VIF > 10.\n   Details: {', '.join(details)}")
            except Exception as e:
                report.append(f"VIF CALCULATION SKIPPED: {e}")

        # 4. CHECK SKEWNESS
        potential_skew_cols = [c for c in features.columns if df[c].nunique() > 2]
        if potential_skew_cols:
            skew_vals = df[potential_skew_cols].skew()
            high_skew = skew_vals[abs(skew_vals) > 1]
            if not high_skew.empty:
                issues_found = True
                details = [f"{col} (Skew={val:.2f})" for col, val in high_skew.items()]
                report.append(f"DISTRIBUTION ERROR: High Skew detected in {details}. Suggest Log Transform.")

        # 5. CHECK SCALING
        high_var_cols = []
        for col in potential_skew_cols:
            if df[col].var() > 1000:
                high_var_cols.append(col)
        
        if high_var_cols:
            issues_found = True
            report.append(f"SCALING REQUIRED: Columns {high_var_cols} have high variance. Apply StandardScaler.")

        if not issues_found:
            return "VERDICT: READY. VIF low, types correct, distribution ok, cardinality safe."
        else:
            return "VERDICT: NOT READY\n" + "\n".join(report)

    except Exception as e:
        return f"Error in check: {e}"