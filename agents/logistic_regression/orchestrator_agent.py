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
from utils.cleaning_tools import standard_cleaning_tool, load_training_data, run_logistic_code

# --- 2. LOCAL HELPER: Run Code in Shared Memory ---
def run_logistic_code(code: str) -> str:
    """Executes Python code in SHARED_GLOBALS environment."""
    print(f"\n[🤖 Logistic Agent Code]:\n{code}\n")
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
    """Dump SHARED_GLOBALS['train_data'] to 'logistic_working.csv'."""
    try:
        import pandas as pd
        if 'train_data' in SHARED_GLOBALS and SHARED_GLOBALS['train_data'] is not None:
            SHARED_GLOBALS['train_data'].to_csv('logistic_working.csv', index=False)
            return "Synced memory to 'logistic_working.csv'."
        else:
            return "WARNING: 'train_data' is None. You must LOAD it first."
    except Exception as e: return f"Sync Error: {e}"

def sync_file_to_memory():
    """Load 'logistic_working.csv' back to SHARED_GLOBALS['train_data']."""
    try:
        import pandas as pd
        if os.path.exists('logistic_working.csv'):
            SHARED_GLOBALS['train_data'] = pd.read_csv('logistic_working.csv')
            SHARED_GLOBALS['readiness_score'] = 1.0 
            return "Synced 'logistic_working.csv' back to memory."
        return "File not found."
    except Exception as e: return f"Sync Error: {e}"

# --- 4. THE LOGISTIC CRITIC ---
def create_logistic_critic():
    return LlmAgent(
        name="Logistic_Critic_Agent",
        model="gemini-2.0-flash", 
        instruction="""
        Rewritten critic instructions to always audit the current CSV statelessly
        
        You are a Classification Data Auditor.

        DATA SOURCE:
        - You must ALWAYS audit the CURRENT version of 'logistic_working.csv'.
        - Treat EVERY call as FRESH: do NOT rely on previous messages or memories.
        - You do NOT read SHARED_GLOBALS directly; you only use the tools provided.

        MANDATORY WORKFLOW (EVERY TIME YOU ARE CALLED):
        1. Call `strict_statistical_check('logistic_working.csv')` with NO target_col argument.
           - Let the tool infer the target.
        2. Base your entire analysis ONLY on the latest tool output.
        3. Extract and report all issues the tool mentions:
           - Target classes (multiclass vs binary)
           - Class imbalance
           - Logic issues (e.g., invalid values)
           - Multicollinearity (VIF > 5)
           - Skewness and scaling needs
        4. If the tool returns a "VERDICT: READY" line, you MUST also output VERDICT: READY.
           Otherwise, output VERDICT: NOT READY and list the concrete issues.

        OUTPUT FORMAT:
        - Start with a short explanation of key issues.
        - End with a line: VERDICT: READY  or  VERDICT: NOT READY
        """,  # CHANGED: removed df-based instructions and explicit target_col param usage
        tools=[
            FunctionTool(run_local_python),
            FunctionTool(strict_statistical_check)
        ]
    )

# --- 5. THE LOGISTIC ORCHESTRATOR ---
def create_logistic_orchestrator() -> Agent:
    logistic_critic = create_logistic_critic()

    return Agent(
        name="Logistic_Orchestrator_Agent",
        model=config.MODEL_NAME, 
        tools=[
            FunctionTool(run_logistic_code),      
            FunctionTool(run_local_python),     
            FunctionTool(sync_memory_to_file),  
            FunctionTool(sync_file_to_memory),  
            AgentTool(agent=logistic_critic)
        ],
        description="Orchestrates Logistic Regression cleaning",
        instruction="""
        You are a Classification Expert.
        GOAL: Optimize 'train_data' for Logistic Regression.
        
        IMPORTANT CONTEXT (SHARED STATE)  # CHANGED: clarified access to SHARED_GLOBALS
        - All data and config live in SHARED_GLOBALS.
        - The tool `run_logistic_code` executes Python INSIDE this shared memory.
        - This means you CAN directly read and write:
          - SHARED_GLOBALS['train_data'], SHARED_GLOBALS['test_data'],
          - SHARED_GLOBALS['target_column'], SHARED_GLOBALS['chosen_regression'], etc.

        WORKFLOW:
        1. INITIALIZE (Robust Loading):  # CHANGED: clarified initialization logic using SHARED_GLOBALS
           - First, check if 'train_data' exists and is non-empty via `run_logistic_code`:
             ```python
             train_data = SHARED_GLOBALS.get('train_data')
             if train_data is None or len(train_data) == 0:
                 # Fallback: load from disk using the shared utility
                 train_data = load_training_data()
                 SHARED_GLOBALS['train_data'] = train_data
             ```
           - Then call `sync_memory_to_file()` so 'logistic_working.csv' is in sync.

           - Run Standard Cleaner (in memory):
             - You MUST use the `run_logistic_code` tool for this step, as it provides access to SHARED_GLOBALS['train_data'].
             - Tool Call: `run_logistic_code`
             - Code:
               ```python
               train_data = SHARED_GLOBALS['train_data']
               df = train_data
               df, report = standard_cleaning_tool(df)
               print(report)
               train_data = df
               SHARED_GLOBALS['train_data'] = train_data  # Ensure memory update
               ```

        2. AUDIT LOOP:
           - Call 'Logistic_Critic_Agent'. It will ALWAYS recompute from 'logistic_working.csv'.
           - READ the Verdict from its response.
           - CRITICAL STOPPING CONDITION:
             - IF VERDICT IS "READY": SKIP Step 3 and GO DIRECTLY to Step 4 (FINALIZE).
             - IF VERDICT IS "NOT READY": Proceed to Step 3 (ACTION).

        3. ACTION: Apply "Smart Strategies".  # CHANGED: emphasised batched fixes + sync after changes
           - Use `run_logistic_code` to modify SHARED_GLOBALS['train_data'].
           - Apply ALL necessary fixes in as few scripts as possible (batch fixes together).
           - After each set of changes, always call `sync_memory_to_file()` so the critic sees updated data.
           - If the Critic complains about the SAME issue twice (e.g. skew in the same column), you may IGNORE it to avoid infinite loops.

        4. FINALIZE (Mandatory):  # CHANGED: ensured final save uses SHARED_GLOBALS and sets readiness_score
           - USE `run_logistic_code` FOR THIS ENTIRE STEP.
             Do NOT use `run_local_python` (it cannot access SHARED_GLOBALS data).
           - Code Block:
             ```python
             # Save Dataset
             train_data = SHARED_GLOBALS['train_data']
             train_data.to_csv('data/processed/logistic_ready_train.csv', index=False)

             # Mark pipeline as fully ready
             SHARED_GLOBALS['readiness_score'] = 1.0

             # Save Report (you can build a richer report based on your actions)
             report_text = "DATA CLEANING REPORT:\\nData cleaning and preprocessing steps were performed to prepare the data for Logistic Regression.\\n"
             with open('classification_report.md', 'w') as f:
                 f.write(report_text)

             print("DATA CLEANING COMPLETE")
             ```
           - Output "DATA CLEANING COMPLETE" in your final answer.

        SMART STRATEGIES (Logistic Specific):

        1. TARGET HANDLING (CRITICAL):
           - Retrieve Target Name: `target_col = SHARED_GLOBALS.get('target_column')` (Do NOT default to 'target').
           - Check Multiclass: If target has values {0, 1, 2, 3...}, BINARIZE IT.
             Code:
             ```python
             df = SHARED_GLOBALS['train_data']
             df[target_col] = (df[target_col] > 0).astype(int)
             SHARED_GLOBALS['train_data'] = df
             ```
           - If target is boolean (True/False), convert to int (1/0).

        2. LOGIC & SAFETY:
           - Drop 'ID', 'id', 'index' columns.
           - Drop Negative values in positive-only columns (e.g. Age, Blood Pressure).
           - Replace Infinite values with NaN, then impute using the median for numeric columns.
             Example:
             ```python
             df = SHARED_GLOBALS['train_data']
             df = df.replace([np.inf, -np.inf], np.nan)
             num_cols = df.select_dtypes(include=[np.number]).columns
             df[num_cols] = df[num_cols].fillna(df[num_cols].median())
             SHARED_GLOBALS['train_data'] = df
             ```

        3. MULTICOLLINEARITY (VIF > 5):
           - Retrieve Target Name: `target_col = SHARED_GLOBALS.get('target_column')`
           - The Critic report lists high VIF features.
           - Calculate correlation of each candidate feature with `target_col`.
           - Drop the feature with the LOWER absolute correlation with the target.
           - Use actual column names (strings), not integer indices.

        4. ENCODING:
           - Low Cardinality (< 20 unique): One-Hot Encoding (`pd.get_dummies` with drop_first=True).
           - High Cardinality (> 20 unique): Target Encoding (using the target column from SHARED_GLOBALS).

        5. SCALING:
           - Apply `StandardScaler` to all numeric features.
           - BEFORE scaling:
             - Ensure no NaNs and no infinities:
               ```python
               df = SHARED_GLOBALS['train_data']
               num_cols = df.select_dtypes(include=[np.number]).columns
               df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
               df[num_cols] = df[num_cols].fillna(df[num_cols].median())
               SHARED_GLOBALS['train_data'] = df
               ```
           - THEN scale all numeric columns safely.

        CRITICAL:
        - NEVER hardcode 'target'. Always use `SHARED_GLOBALS.get('target_column')`.
        - Anti-Looping: If Critic complains about the SAME issue twice for the SAME column, you may stop trying to fix it again and proceed to FINALIZE.
        """,
    )