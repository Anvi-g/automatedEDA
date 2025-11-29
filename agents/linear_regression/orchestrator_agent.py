import pandas as pd
import numpy as np
from google.adk.agents import Agent, LlmAgent
from google.adk.tools import AgentTool, FunctionTool
import sys
import os
import traceback
import io

# --- 1. SETUP PATHS & IMPORTS ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.shared_environment import SHARED_GLOBALS
from config.agent_config import config
from tools import run_local_python, strict_statistical_check
from utils.cleaning_tools import standard_cleaning_tool, load_training_data

# --- 2. LOCAL HELPER: Run Code in Shared Memory ---
def run_linear_code(code: str) -> str:
    """Executes Python code in SHARED_GLOBALS environment."""
    print(f"\n[🤖 Linear Agent Code]:\n{code}\n")
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    try:
        # Pass the shared memory dict to exec
        exec_globals = SHARED_GLOBALS
        exec_globals["SHARED_GLOBALS"] = SHARED_GLOBALS 
        
        exec(code, exec_globals)
        
        # Sync back updates
        SHARED_GLOBALS.update(exec_globals)
        
        sys.stdout = old_stdout
        return f"Execution Success. Output:\n{redirected_output.getvalue()}"
    except Exception as e:
        sys.stdout = old_stdout
        return f"Execution Error:\n{traceback.format_exc()}"

# --- 3. HELPER TOOLS: Sync Memory <-> Disk ---
def sync_memory_to_file():
    """Dump SHARED_GLOBALS['train_data'] to 'linear_working.csv'."""
    try:
        import pandas as pd
        if 'train_data' in SHARED_GLOBALS and SHARED_GLOBALS['train_data'] is not None:
            SHARED_GLOBALS['train_data'].to_csv('linear_working.csv', index=False)
            return "Synced memory to 'linear_working.csv'."
        else:
            return "WARNING: 'train_data' is None. You must LOAD it first."
    except Exception as e: return f"Sync Error: {e}"

def sync_file_to_memory():
    """Load 'linear_working.csv' back to SHARED_GLOBALS['train_data']."""
    try:
        import pandas as pd
        if os.path.exists('linear_working.csv'):
            SHARED_GLOBALS['train_data'] = pd.read_csv('linear_working.csv')
            SHARED_GLOBALS['readiness_score'] = 1.0 
            return "Synced 'linear_working.csv' back to memory."
        return "File not found."
    except Exception as e: return f"Sync Error: {e}"

def create_linear_critic():
    return LlmAgent(
        name="Linear_Critic_Agent",
        model="gemini-2.0-flash", 
        instruction="""
        You are a Strict Data Quality Auditor & Feature Strategist.
        
        TASK: Generate a "Regression Readiness & Engineering Report" for 'linear_working.csv'.
        
        WORKFLOW:
        
        1. **INSPECT & IDENTIFY TARGET:**
           - Run `run_local_python` with: `print(pd.read_csv('linear_working.csv').columns.tolist())`
           - **IDENTIFY THE TARGET:** Look for 'price', 'medv', 'target', 'cost', 'amount'.
           - **IDENTIFY IDs:** Look for 'ID', 'id', 'index', 'serial'.
        
        2. **STATISTICAL AUDIT:**
           **MANDATORY CHECK:** Call `strict_statistical_check('linear_working.csv')`.
           - This tool checks Logic, VIF, Skewness, and Scaling *all at once*.
        
        3. **SEMANTIC REASONING:**
           - **Logic:** Check if positive-only features (Age, Area, Distance) have negative values.
           - **Interaction:** Suggest 'Area' if you see 'Length'/'Width'.
           - **Cardinality:** If tool reports > 20 unique values in a text column, suggest Target Encoding.
        
        4. **COMPREHENSIVE REPORT:** - Do not report just one issue. List **EVERY** failure found by the tool.
           - If the tool finds VIF > 5 AND Skew > 1 AND Logic Errors, report ALL of them.
            REPORT FORMAT (Strict Syntax):
            --- AUDIT REPORT ---
            TARGET: [Name of Target Column]
            ISSUES:
            1. [Stat Issue] (e.g. "VIF=8.2, CorrWithTarget=0.4 for 'tax'.") <- MUST INCLUDE NUMBERS.
            2. [Logic Issue] (e.g. "Column 'crim' has negative values.")
            3. [Identifier] (e.g. "Column 'ID' should be dropped.")
            4. [Engineering Opp] (e.g. "Create interaction 'room_size' from 'rm' * 'tax'")
            ...
        VERDICT: [READY / NOT READY]
        """,
        tools=[
            FunctionTool(run_local_python),
            FunctionTool(strict_statistical_check)
        ]
    )

def create_linear_orchestrator() -> Agent:
    linear_critic = create_linear_critic()
    return Agent(
            name="Linear_Orchestrator_Agent",
            model=config.MODEL_NAME, 
            tools=[
                FunctionTool(run_linear_code),      
                FunctionTool(run_local_python),     
                FunctionTool(sync_memory_to_file),  
                FunctionTool(sync_file_to_memory),  
                AgentTool(agent=linear_critic)
            ],
            description="Orchestrates Linear Regression specific cleaning",
            instruction="""

            You are a Senior Data Scientist.
        
            GOAL: Optimize 'train_data' (in memory) for Linear Regression.
            
            WORKFLOW:
        
            1. INITIALIZE:
            - Call `sync_memory_to_file()`.
            - **IF it says "No train_data"**: 
                - You must LOAD the data from the processed folder using `run_linear_code`.
                - The previous agent saved it as **'data/processed/train.csv'**.
                - Code: `train_data = pd.read_csv('data/processed/train.csv'); SHARED_GLOBALS['train_data'] = train_data`
                - Then call `sync_memory_to_file()` again.
            
            - Once synced, Call `run_linear_code` to run:
                ```python
                # The tool is available in the shared environment
                df = train_data
                df, report = standard_cleaning_tool(df)
                print(report)
                train_data = df
                ```
                    
            2. AUDIT LOOP: - Call 'Linear_Critic_Agent' to audit 'linear_working.csv' and read the report.
            3. ACTION: 
            - **CRITICAL:** If the Critic reports multiple issues (e.g. "VIF High" AND "Skew High"), do not fix them one by one.
            - **WRITE ONE SINGLE PYTHON SCRIPT** that applies ALL fixes in sequence:
                1. Drop IDs/Constant Cols.
                2. Fix Logic (Drop Negatives).
                3. Fix Multicollinearity (Drop High VIF).
                4. Fix Skew (Log Transform).
                5. Fix Scaling.
            - Execute this script using `run_linear_code`.
            - Fix issues using the "Smart Strategies".
            - **IMPORTANT**: Any changes you make to `train_data` in memory MUST be synced to disk for the Critic to see.
            - After changing `train_data`, Call `sync_memory_to_file()`.
            4. FINALIZE (MANDATORY):
            - Whether "VERDICT: READY" or you are stopping due to "Anti-Looping":
            
                - Call `sync_file_to_memory()` to finalize.
                    - **Save the Final Dataset:**
                    ```python
                    train_data.to_csv('data/processed/linear_ready_train.csv', index=False)
                    print("Saved final linear dataset.")
                    ```
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
            
            1.LOGIC & STATS:
            - **Identifiers:** Drop 'ID', 'Id', 'index' columns immediately.
            - **Constant Columns:** Drop columns with only 1 unique value (e.g. 'chas' if all 0).
            - **Negative Values:** Drop rows. `df = df[df['col'] >= 0]`
            - **Boolean:** `df[col] = df[col].astype(int)`
            
            1. MULTICOLLINEARITY (VIF > 5 or Corr > 0.8) - EXTREME AGGRESSION:
           - **DECISION RULE:** The report provides "CorrWithTarget" for each feature.
           - **CRITICAL MATH:** Compare the **ABSOLUTE VALUES** (`abs()`) of the correlations.
             - Example: `FeatureA (-0.9)` is BETTER than `FeatureB (0.4)`.
           - **ACTION:** Drop the feature with the LOWER absolute correlation.
           - *Note:* If multiple pairs exist, drop the weaker one from EACH pair in the same script.
        
            
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
            6. TARGET LEAKAGE (CRITICAL):
            - If the report says "LEAKAGE ALERT", these columns are cheating.
            - ACTION: DROP them immediately.
            - Code: `df.drop(columns=['leaky_col'], inplace=True)`

            7. CONSTANT/ZERO VARIANCE (General Solution):
            - If a column has only 1 unique value (e.g. all 0s), it provides no information.
            - **ACTION:** Drop the column immediately.
            - Code: `df.drop(columns=['col'], inplace=True)`

            CRITICAL BATCHING RULES:
            - Aggregate ALL fixes into ONE Python script execution.
            - **Anti-Looping:** If the Critic complains about the SAME issue (e.g. "Skew in Age") for the second time, IGNORE IT. Do not re-apply the same fix.
            - Always load 'linear_working.csv', modify it, and save it back.
            - On finish: Write 'cleaning_report.md' explaining your decisions.
            """
        )