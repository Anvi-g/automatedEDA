import pandas as pd
from google.adk.agents import Agent
from config.agent_config import config
import traceback
import sys
import io
import os

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.shared_environment import SHARED_GLOBALS

def run_python_code(code: str) -> str:
    """Executes Python code in the SHARED_GLOBALS environment."""
    print(f"\n[🤖 Basic EDA Agent Code]:\n{code}\n")
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    try:
        # Inject self-reference for robustness
        exec_globals = SHARED_GLOBALS
        exec_globals["SHARED_GLOBALS"] = SHARED_GLOBALS 
        
        exec(code, exec_globals)
        sys.stdout = old_stdout
        return f"Execution Success. Output:\n{redirected_output.getvalue()}"
    except Exception as e:
        sys.stdout = old_stdout
        return f"Execution Error:\n{traceback.format_exc()}"

def create_basic_eda_agent() -> Agent:
    return Agent(
        name="BasicEDA_Agent",
        model=config.MODEL_NAME,
        tools=[run_python_code],
        description="Performs initial data loading, splitting (80/20), and routing.",
        # FIX: Removed the f-string syntax that confused the template engine
        instruction="""
        <system-reminder>
        BASIC EDA AGENT STARTING
        </system-reminder>
        
        You are an Intelligent Data Preprocessing Specialist. 
        Your goal is to prepare the raw data for the downstream Linear/Logistic pipelines.
        
        TASK:
        1. LOAD: Load 'data/raw_dataset.csv' into `df`.
        2. INSPECT: Print info, describe, and shape.
        3. TARGET: Identify the target column. Store name in `SHARED_GLOBALS["target_column"]`.
        
        4. SPLIT (80/20):
           - Perform a Train/Test split (80% Train, 20% Test).
           - You MUST use a try-except block to handle stratification errors.
           - Code Pattern:
             ```python
             try:
                 # Try stratified split
                 X_train, X_test, y_train, y_test = train_test_split(..., stratify=bins)
                 print("Success: Stratified Split")
             except ValueError:
                 print("Stratification failed. Defaulting to Random Split.")
                 X_train, X_test, y_train, y_test = train_test_split(..., stratify=None)
             ```
           - Store results:
             `SHARED_GLOBALS["train_data"] = train_df`
             `SHARED_GLOBALS["test_data"] = test_df`
        
        5. **CRITICAL SAVE STEP (Use GENERIC Names):**
           - You MUST run this specific code to save the files to disk so the next agent can find them:
           ```python
           import os
           os.makedirs('data/processed', exist_ok=True)
           # Save as generic 'train.csv' and 'test.csv'
           SHARED_GLOBALS["train_data"].to_csv('data/processed/train.csv', index=False)
           SHARED_GLOBALS["test_data"].to_csv('data/processed/test.csv', index=False)
           print("SUCCESS: Saved data/processed/train.csv and test.csv")
           ```
        
        6. ROUTE:
           - Check target column type.
           - If Continuous -> Set `SHARED_GLOBALS["chosen_regression"] = "linear"`.
           - If Categorical -> Set `SHARED_GLOBALS["chosen_regression"] = "logistic"`.
           - PRINT: "Detected regression type: [TYPE]"
        
        **CRITICAL CODING RULES:**
        - **NEVER** write `SHARED_GLOBALS = {}`. It is ALREADY defined.
        - **NEVER** re-import `SHARED_GLOBALS`.
        - **AVOID** using `\n` inside print statements. Use multiple print() calls instead.
        - Use `run_python_code`.
        """
    )

def create_user_choice_agent() -> Agent:
    # Minimal placeholder as BasicEDA handles routing now
    return Agent(
        name="UserChoice_Agent",
        model=config.MODEL_NAME,
        tools=[run_python_code],
        description="Legacy placeholder for user interaction",
        instruction="Orchestrate next steps."
    )