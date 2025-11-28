import pandas as pd
import numpy as np
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool, FunctionTool
from tools import run_local_python, strict_statistical_check

# ---------------------------------------------------------
# 1. TOOL: Standard Deterministic Cleaner (GENERIC)
# ---------------------------------------------------------
def standard_cleaning_tool(filepath: str) -> str:
    """Standard cleaning: Drop duplicates, Median fill, Mode fill, Z-score Outliers."""
    try:
        import pandas as pd
        import numpy as np

        df = pd.read_csv(filepath)
        report = []
        
        # 1. Duplicates
        init_len = len(df)
        df = df.drop_duplicates()
        if len(df) < init_len: 
            report.append(f"Dropped {init_len-len(df)} duplicates.")

        # 2. NaNs
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if np.issubdtype(df[col].dtype, np.number):
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    report.append(f"Filled NaNs in {col} (Median: {median_val}).")
                else:
                    mode_s = df[col].mode()
                    if not mode_s.empty:
                        mode_val = mode_s[0]
                        df[col] = df[col].fillna(mode_val)
                        report.append(f"Filled NaNs in {col} (Mode: {mode_val}).")

        # Drop Constant Columns (Zero Variance)
        for col in df.columns:
            if df[col].nunique() <= 1:
                df.drop(columns=[col], inplace=True)
                report.append(f"Dropped constant column '{col}' (Single unique value).")


        # 3. Outliers (Z-score > 3)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].std() > 0: 
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > 3
                num_outliers = outliers.sum()
                if num_outliers > 0:
                    df = df[~outliers]
                    report.append(f"Removed {num_outliers} outliers from '{col}' (Z-score > 3).")
        
        df.to_csv("cleaned_data.csv", index=False)
        report.append("Saved to 'cleaned_data.csv'.")
        return "\n".join(report)
    except Exception as e: 
        return f"Error: {e}"

# ---------------------------------------------------------
# 2. AGENT: The Strict Critic (Reasoning Engine)
# ---------------------------------------------------------
critic_agent = LlmAgent(
    name="critic_agent",
    model="gemini-2.0-flash", # Valid Model ID
    instruction="""
 
    You are a Data Quality Auditor & Feature Strategist.
    
     
    TASK: Generate a "Regression Readiness & Engineering Report" for 'cleaned_data.csv'.
    
    WORKFLOW:
    1. Load data via 'run_local_python'.
    2. RUN 'strict_statistical_check' tool (for stats).
    3. PERFORM SEMANTIC CHECKS (Your reasoning):
       - Look at column names. Do columns like 'Price', 'Area', 'Age', 'Distance', 'Count' have negative values? (Impossible).
       - Interactions: Do you see 'Length' & 'Width' (Suggest Area)? 'Start' & 'End' (Suggest Duration)?
       - Do columns imply a ratio? (e.g. 5000 sqft with 1 bedroom is suspicious).
       - Cardinality: Do categorical columns have > 50 unique values? (Suggest Target Encoding).

    REPORT FORMAT:
    --- AUDIT REPORT ---
    ISSUES:
    1. [Issue] (COPY EXACT NUMBERS from the tool. Do not summarize. If tool says 'VIF=8.2, Corr=0.4', write exactly that.)
    2. [Logic Issue] (e.g. "Column 'Square_Feet' has negatives")
    3. [Stat Issue] (e.g. "Dummy Trap in Neighborhood")
    4. [Engineering Opp] (e.g. "Create 'Area' from Length*Width")
    ...
    VERDICT: [READY / NOT READY]
    """,
    tools=[
        FunctionTool(run_local_python),
        FunctionTool(strict_statistical_check)
    ]
)

# ---------------------------------------------------------
# 3. AGENT: The Orchestrator (Manager + Coder)
# ---------------------------------------------------------
orchestrator_agent = LlmAgent(
    name="orchestrator",
    model="gemini-2.5-flash", 
    instruction="""
    You are a Senior Data Scientist.
    
    GOAL: Produce a dataset optimized for Linear Regression.
    
    WORKFLOW:
    1. FAST CLEAN: Run 'standard_cleaning_tool'.
    2. AUDIT LOOP: Run 'critic_agent'. Read Report.
    3. ACTION: Fix issues using the "Smart Strategies".
    4. FINALIZE (MANDATORY):
       - Whether "VERDICT: READY" or you are stopping due to "Anti-Looping":
       
       - **ACTION 1: SAVE REPORT TO DISK**
         You MUST write and execute a Python script to save the file:
         ```python
         report_content = \"\"\"
         # Cleaning Report
         [...Your Detailed Summary of Changes...]
         \"\"\"
         with open('cleaning_report.md', 'w') as f:
             f.write(report_content)
         print("SUCCESS: cleaning_report.md saved.")
         ```
       
       - **ACTION 2: TERMINATE**
         - IF "VERDICT: READY": Output "DATA CLEANING COMPLETE (Perfect)".
         - IF stopping due to Anti-Looping: Output "DATA CLEANING COMPLETE (Best Effort)".
    
    SMART STRATEGIES:
    
    1. LOGIC & STATS:
       - Negative Values: Drop rows. `df = df[df['col'] >= 0]`
       - Boolean: `df[col] = df[col].astype(int)`
       
    2. MULTICOLLINEARITY (VIF > 5):
       - **DECISION RULE:** The report gives you "CorrWithTarget" for the high-VIF features.
       - **ACTION:** Keep the feature with HIGHER correlation. DROP the feature with LOWER correlation.
       - Code: `df.drop(columns=['lower_corr_col'], inplace=True)`
    
    3. TRANSFORMATION (Skewness Handling):
       - **Right Skew (> 1):** Use Log Transform.
         Code: `df['col'] = np.log1p(df['col'])`
       - **Left Skew (< -1):** Use Square Transform (pushes mass right).
         Code: `df['col'] = df['col']**2`
       - **Persistence:** If you have already transformed a column once, DO NOT transform it again. Accept the remaining skew as "Best Effort".
    
    4. SCALING:
       - Range Diff: `scaler.fit_transform(df[['col']])`

    5. ENCODING (The "Tax" Fix):
       - **Low Cardinality (< 20):** One-Hot (`pd.get_dummies`).
       - **High Cardinality (> 20):** Target Encoding.
         Code: 
         ```python
         mean_target = df['price'].mean()
         agg = df.groupby('col')['price'].mean()
         df['col_encoded'] = df['col'].map(agg).fillna(mean_target)
         df.drop(columns=['col'], inplace=True)
         ```

    CRITICAL BATCHING RULES:
    - Aggregate ALL fixes into ONE Python script execution.
    - **Anti-Looping:** If the Critic complains about the SAME issue (e.g. "Skew in Age") for the second time, IGNORE IT. Do not re-apply the same fix.
    - Always load 'cleaned_data.csv', modify it, and save it back.
    - On finish: Write 'cleaning_report.md' explaining your decisions.
    """,
    tools=[
        FunctionTool(standard_cleaning_tool), 
        FunctionTool(run_local_python),       
        AgentTool(agent=critic_agent)         
    ]
)