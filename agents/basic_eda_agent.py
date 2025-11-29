import pandas as pd
from google.adk.agents import Agent
from config.agent_config import config
import traceback
import sys
import io
import os

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from google.adk.tools import FunctionTool
from utils.shared_environment import SHARED_GLOBALS
from utils.hitl_tools import confirm_experiment_setup

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

        keys_to_ignore = {'__builtins__', 'SHARED_GLOBALS'}
        for k, v in exec_globals.items():
            if k not in keys_to_ignore:
                SHARED_GLOBALS[k] = v

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
      
        instruction="""
        <system-reminder>
        BASIC EDA AGENT STARTING
        </system-reminder>
        
        You are an Intelligent Data Preprocessing Specialist. 
        Your goal is to LOAD and INSPECT the raw data.
        
        TASK:
        1. LOAD: Load 'data/raw_dataset.csv' into `df`.
           - Code: `df = pd.read_csv('data/raw_dataset.csv'); SHARED_GLOBALS['raw_data'] = df`
        
        2. INSPECT & GUESS (CRITICAL):
            - Write ONE Python script to perform inspection and guessing.
            - **INSPECTION:** Print `df.info()`, `df.head()`, and `df.columns.tolist()`.
            - **GUESSING:** Determine the 'target_column' and 'chosen_regression' type.
            - **Heuristics:** Target names often include 'price', 'target', 'medv', 'class', etc. If a column has <= 20 unique values, guess 'logistic'. Otherwise, guess 'linear'.
            - **Code Example (must be generated):**
              ```python
              df = SHARED_GLOBALS['raw_data']
              cols = list(df.columns)
              
              # Guess Target (e.g., assuming it's the last one or 'target')
              target_guess = next((c for c in cols if c.lower() in ['price', 'medv', 'target', 'class']), cols[-1])
              
              # Guess Type (Check unique values of the guessed target)
              n_unique = df[target_guess].nunique()
              type_guess = 'logistic' if n_unique <= 20 else 'linear'
              
              # Save guesses to SHARED_GLOBALS
              SHARED_GLOBALS['target_guess'] = target_guess
              SHARED_GLOBALS['type_guess'] = type_guess
              print("Initial guess saved.")
              ```
        
        3. HANDOFF:
            - Output "Initial Data and Guesses Loaded."
            - The next agent (UserChoice_Agent) will handle confirmation and splitting.
        """
    )

def create_user_choice_agent() -> Agent:
    return Agent(
        name="UserChoice_Agent",
        model=config.MODEL_NAME,
        tools=[
            run_python_code, 
            FunctionTool(confirm_experiment_setup) 
        ],
        description="Handles user interaction and data splitting.",
        instruction="""
        <system-reminder>PHASE 2: HUMAN-IN-THE-LOOP & SPLITTING</system-reminder>
 
        You are the Project Manager. Your task is to finalize the target and type, then split the data.

        WORKFLOW (Execute sequentially):

        STEP 1: PREPARE INPUTS
        - Execute a script to read 'raw_data' and save the column list.

        STEP 2: FULL HITL & SPLIT SCRIPT (CRITICAL: One run_python_code call)
        - You **MUST** generate the Python script below and execute it. 
        - The script performs all guessing, the HITL tool call, configuration saving, and data splitting.
        
        - **This single script handles the entire phase:**
          ```python
          from sklearn.model_selection import train_test_split
          
          df = SHARED_GLOBALS['raw_data']
          cols = list(df.columns)
          
          # 1. INFERENCE (Use saved guess or fall back)
          target_guess = SHARED_GLOBALS.get('target_guess', cols[-1])
          type_guess = SHARED_GLOBALS.get('type_guess', 'linear')
          
          # 2. CALL HITL TOOL (Execution pauses for user input)
          hitl_result = confirm_experiment_setup(target_guess, type_guess, cols) 
          
          # 3. SAVE CONFIRMED CONFIGURATION
          SHARED_GLOBALS['target_column'] = hitl_result['target_col']
          SHARED_GLOBALS['chosen_regression'] = hitl_result['regression_type']
          
          print("Configuration saved: " + SHARED_GLOBALS['target_column'] + " is the confirmed target.")
          
          # 4. PERFORM SPLIT
          # Check if stratify is appropriate (not possible here, run simple split)
          train, test = train_test_split(df, test_size=0.2, random_state=42)
          
          # 5. SAVE SPLIT DATA
          SHARED_GLOBALS['train_data'] = train
          SHARED_GLOBALS['test_data'] = test
          print("Data Split Complete. Train Shape: " + str(train.shape))
          ```
        
        DO NOT STOP until you see "Data Split Complete. Train Shape: ..." in the tool output.
        """
    )