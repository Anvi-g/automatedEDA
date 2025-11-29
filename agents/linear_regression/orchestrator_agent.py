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
             
            return "Synced 'linear_working.csv' back to memory."
        return "File not found."
    except Exception as e: return f"Sync Error: {e}"

def create_linear_critic():
    return LlmAgent(
        name="Linear_Critic_Agent",
        model="gemini-2.0-flash", 
        instruction="""
        You are a Strict Data Quality Auditor & Feature Strategist.

        TASK:
        Generate a "Regression Readiness & Engineering Report" for 'linear_working.csv'.

        TOOLS YOU CAN USE (AND NOTHING ELSE):
        - run_local_python
        - strict_statistical_check

        Hard rules about tools:
        - You MUST NOT call any other tools or tool names (no `tolist`, `describe`, `head`, etc.).
        - If you need `.tolist()` or similar, use it only as a normal Python method INSIDE a `run_local_python` code block, e.g.:

            import pandas as pd
            df = pd.read_csv("linear_working.csv")
            print(df.columns.tolist())

        - All statistical quantities (VIF, skewness, correlations, scaling flags, etc.) must come from strict_statistical_check.
        - Do NOT recompute or invent your own numeric values.

        MANDATORY WORKFLOW (EVERY CALL):

        STATISTICAL AUDIT (PRIMARY SOURCE OF TRUTH)
        - First, call: strict_statistical_check("linear_working.csv")
        - Treat the output of this tool as the SINGLE SOURCE OF TRUTH for:
        - VIF and multicollinearity
        - Skewness
        - Scaling needs / high variance
        - Logic issues (invalid or impossible values)
        - Any leakage or warnings it reports
        - You MUST NOT fabricate any numerical values or issues that are not mentioned in this tool output.
        - If a quantity is not reported by the tool, say that it was "not reported by strict_statistical_check".

        OPTIONAL STRUCTURAL INSPECTION (NO EXTRA STATS)
        - ONLY if needed to understand structure, you MAY call run_local_python with a short script such as:

            import pandas as pd
            df = pd.read_csv("linear_working.csv")
            print(df.columns.tolist())
            print(df.dtypes)

        - Use run_local_python ONLY for:
        - Listing columns
        - Inspecting dtypes
        - Seeing a few example rows
        - Do NOT compute new statistics. All stats must come from strict_statistical_check.

        SEMANTIC REASONING (BASED ON TOOL OUTPUT)
        - Use the strict_statistical_check results plus column-name semantics to reason about:
        - Logic errors: e.g., negative values in positive-only features if the tool reports them.
        - Identifiers: e.g., 'ID', 'id', 'index', 'serial'.
        - Feature engineering opportunities: e.g., Length + Width => suggest Area.
        - You may infer semantics from names, but MUST NOT invent statistical problems not mentioned in the tool output.

        COMPREHENSIVE REPORT (STRICT FORMAT, NO CODE FENCES)
        Your final answer MUST follow this exact textual structure:

        --- AUDIT REPORT ---
        TARGET: [Name of Target Column]
        ISSUES:
        1. [Issue 1 with numbers copied exactly from strict_statistical_check]
        2. [Issue 2]
        3. ...
        VERDICT: [READY / NOT READY]

        Rules for ISSUES:
        - For each problem the tool reports (VIF, Skew, Scaling, Logic, Leakage, etc.), add a bullet with:
        - Column name(s)
        - Numeric values EXACTLY as reported
        - If the tool reports no issues, write:
        ISSUES:
        (none reported by strict_statistical_check)

        VERDICT (MUST MATCH TOOL)
        - Always copy the verdict EXACTLY as strict_statistical_check reports.
        - If tool says "VERDICT: READY", you MUST output "VERDICT: READY".
        - If tool says "VERDICT: NOT READY", you MUST output "VERDICT: NOT READY".
        - Never override the verdict.

        HALLUCINATION GUARDRAILS (CRITICAL):
        - Do NOT invent new tool names.
        - Do NOT invent numeric values.
        - Do NOT claim issues not mentioned by strict_statistical_check.
        - If unsure about any detail, state: "not reported by strict_statistical_check".
        - Output must be plain text, with no Markdown code blocks, no extra commentary.

        Your entire answer must be ONLY the final audit report in the required format.
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
                SHARED_GLOBALS['train_data'] = df  
                ```
                    
            2. AUDIT LOOP: 
            - Call 'Linear_Critic_Agent' to audit 'linear_working.csv' and read the report.
            

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
            - After modifying SHARED_GLOBALS['train_data'], call `sync_memory_to_file()` so the critic sees the updated data.
            - **Anti-Looping:** If the Critic complains about the SAME issue for the SAME column more than once, do NOT re-apply the same fix. Instead, proceed toward FINALIZE.

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
            - Use `StandardScaler` for columns the critic flags as "Scaling Required" or with very high variance.
            
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