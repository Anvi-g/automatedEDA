import pandas as pd
import numpy as np
from google.adk.agents import Agent, LlmAgent
from google.adk.tools import AgentTool, FunctionTool
import sys
import os
import traceback
import io
import json

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
        1. Call `strict_statistical_check('logistic_working.csv')` exactly ONCE.
           - Do NOT loop over columns manually.
           - Let the tool compute ALL of:
             - Target diagnostics (binary vs multiclass)
             - Class imbalance
             - Boolean / non-numeric issues
             - Logic errors (invalid ranges, negatives where not allowed, etc.)
             - Skewness and recommended log transforms
             - Scaling needs
             - Multicollinearity (VIF > threshold) for ALL problematic features
        2. Base your entire analysis ONLY on the latest tool output.
        3. Extract and report **all** issues mentioned by the tool:
           - If there are 5 features with VIF > 5, LIST ALL FIVE.
           - If there are multiple skewed columns, LIST ALL of them.
           - If there are multiple boolean columns, LIST ALL of them.
        4. Summarize them in a SINGLE consolidated report so the orchestrator
           can fix everything in one batched script.

        OUTPUT FORMAT (STRICT):
        - Start with a short natural-language summary.
        - Then list issues in a structured way like:

          ISSUES:
          - TARGET: [description, e.g. "5 classes in 'num', suggested binarization >0"]
          - BOOLEAN_TO_INT: ['fbs', 'exang']
          - SKEW_LOG_SUGGESTED: ['oldpeak', 'ca']
          - HIGH_VIF_FEATURES: ['fbs', 'slope_flat', 'slope_upsloping']
          - SCALING_NEEDED: ['id', 'chol']

        - Finally, end with a line:
          VERDICT: READY
          or
          VERDICT: NOT READY

          Rules:
        - Never hide or defer issues; report EVERYTHING the tool found in this call.
        - If `strict_statistical_check` itself reports "VERDICT: READY", you MUST
          also end with: VERDICT: READY.
        
        """,  
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
           - Call 'Logistic_Critic_Agent'. It will ALWAYS recompute from 'logistic_working.csv'
             using a single, batched `strict_statistical_check` call.
           - READ the consolidated "ISSUES:" section from its response.
             The issues may include:
             - BOOLEAN_TO_INT: [...]
             - SKEW_LOG_SUGGESTED: [...]
             - HIGH_VIF_FEATURES: [...]
             - SCALING_NEEDED: [...]
             - TARGET diagnostics, etc.
           - CRITICAL STOPPING CONDITION:
             - IF VERDICT IS "READY": SKIP Step 3 and GO DIRECTLY to Step 4 (FINALIZE).
             - IF VERDICT IS "NOT READY": Proceed to Step 3 (ACTION).

            AFTER EVERY CRITIC CALL (IMPORTANT FOR REPORTING):  
            - After you read the response text from `Logistic_Critic_Agent`, you MUST:
              - Summarize the key issues in 1–3 bullet points in your own words.
              - Then call `run_logistic_code` with a script that appends this summary string to SHARED_GLOBALS['critic_messages'].
              - Example:
                ```python
                msgs = SHARED_GLOBALS.get('critic_messages')
                if msgs is None:
                    msgs = []
                msgs.append("Critic Iteration 1: High skew in 'ca', 'oldpeak'; scaling needed for 'chol'; boolean cols 'fbs', 'exang' should be int.")
                SHARED_GLOBALS['critic_messages'] = msgs
                ```

        3. ACTION: Apply "Smart Strategies" IN ONE BATCH.   # CHANGED
           - Use **exactly ONE** `run_logistic_code` call per critic iteration
             to handle ALL issues mentioned in that critic report.
           - Pattern:

             ```python
             import numpy as np
             import pandas as pd
             from sklearn.preprocessing import StandardScaler

             df = SHARED_GLOBALS['train_data']
             target_col = SHARED_GLOBALS.get('target_column')

             # 1. BOOLEAN_TO_INT: convert ALL boolean columns listed by the critic
             #    Example (you must plug in the column names from the critic message):
             #    for col in ['fbs', 'exang', ...]:
             #        if col in df.columns:
             #            df[col] = df[col].astype(int)

             # 2. LOG-TRANSFORM ALL skewed columns (SKEW_LOG_SUGGESTED)
             #    Use the shared registry 'log_transformed_cols' so each feature is
             #    only transformed once.
             #    Example:
             log_cols = SHARED_GLOBALS.get('log_transformed_cols')
             if log_cols is None:
                 log_cols = []
                 SHARED_GLOBALS['log_transformed_cols'] = log_cols

             # for col in ['oldpeak', 'ca', ...]:  # from critic
             #     if col in df.columns and col not in log_cols:
             #         df[col] = np.log1p(df[col])
             #         log_cols.append(col)
             #         print("Applied log1p to", col)

             SHARED_GLOBALS['log_transformed_cols'] = log_cols

             # 3. MULTICOLLINEARITY: handle ALL HIGH_VIF_FEATURES in one go.
             #    - The critic gives you a list of high-VIF features (e.g. ['fbs', 'slope_flat', 'slope_upsloping']).
             #    - For each problematic group, compute correlation with target and drop feature(s)
             #      with lower |corr|. Do this for EVERY group in this ONE script.

             # 4. SCALING: apply StandardScaler to ALL numeric features that need scaling.
             #    - Before scaling, replace infs with NaN and impute medians:

             num_cols = df.select_dtypes(include=[np.number]).columns
             df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
             df[num_cols] = df[num_cols].fillna(df[num_cols].median())

             scaler = StandardScaler()
             df[num_cols] = scaler.fit_transform(df[num_cols])

             # 5. TARGET HANDLING: ensure target is strictly 0/1 (int) using the robust
             #    pattern defined in the SMART STRATEGIES section.

             SHARED_GLOBALS['train_data'] = df

             # Optionally append a transformation log entry
             tx = SHARED_GLOBALS.get('transformations_applied')
             if tx is None:
                 tx = []
             tx.append({"transformation": "Batch fix of issues from last critic report", "timestamp": "auto"})
             SHARED_GLOBALS['transformations_applied'] = tx
             ```

           - After this ONE batched script:
             - Call `sync_memory_to_file()` exactly once.
             - Then call 'Logistic_Critic_Agent' again.
           - Do **NOT** write separate scripts like:
             "one for booleans", "one for skew", "one for VIF". All fixes from
             the most recent critic call must be combined into **one** script.

        - Anti-Looping:
          - If the critic repeats the SAME issue for the SAME column (e.g. 'ca' skew)
            after you have already applied your transformation once and documented it
            in `log_transformed_cols`, you may safely IGNORE that specific suggestion
            and move toward FINALIZE once the other issues are resolved.

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

              ```python
             import json

             # Save Dataset
             train_data = SHARED_GLOBALS['train_data']
             train_data.to_csv('data/processed/logistic_ready_train.csv', index=False)

             # Mark pipeline as fully ready
             SHARED_GLOBALS['readiness_score'] = 1.0

             # Build Rich Report
             critic_msgs = SHARED_GLOBALS.get('critic_messages', [])
             transformations = SHARED_GLOBALS.get('transformations_applied', [])

             lines = []
             lines.append("# DATA CLEANING REPORT")
             lines.append("")
             lines.append("## 1. Overview")
             lines.append("This report summarizes the automated cleaning and preprocessing performed by the ADEP logistic pipeline.")
             lines.append("")
             lines.append("## 2. Critic Findings Over Iterations")

             if critic_msgs:
                 
                 for entry in critic_msgs:
                     idx = idx + 1
                     if isinstance(entry, dict):
                         msg_text = entry.get("message", "")
                     else:
                         msg_text = str(entry)
                      
                     lines.append("- Iteration " + str(idx) + ": " + msg_text)
             else:
                 lines.append("- No critic messages were recorded.")

             lines.append("")
             lines.append("## 3. Applied Transformations")
             if transformations:
                 for t in transformations:
                     if isinstance(t, dict):
                         desc = t.get("transformation", "")
                         ts = str(t.get("timestamp", ""))
                         lines.append("- " + desc + " (" + ts + ")")
                     else:
                         lines.append("- " + str(t))
             else:
                 lines.append("- No transformations were logged.")

             lines.append("")
             lines.append("## 4. Final Dataset Snapshot")
             lines.append("- Shape: " + str(train_data.shape))
             lines.append("- Columns: " + str(list(train_data.columns)))

             report_text = "\n".join(lines)

             with open('classification_report.md', 'w') as f:
                 f.write(report_text)

             print("DATA CLEANING COMPLETE")
             ```

        SMART STRATEGIES (Logistic Specific):

        1. TARGET HANDLING (CRITICAL):
        - The target column name is confirmed during HITL and stored as:
          `target_col = SHARED_GLOBALS.get('target_column')`.
        - You MUST ensure that, for logistic regression, the final target values are STRICTLY 0 or 1 (integers).
        - Use the following robust pattern ONCE at the start of your ACTION phase (via `run_logistic_code`):

          ```python
          import numpy as np
          import pandas as pd

          df = SHARED_GLOBALS['train_data']
          target_col = SHARED_GLOBALS.get('target_column')

          if target_col is None:
              raise ValueError("Target column not found in SHARED_GLOBALS['target_column'].")

          y = df[target_col]

          # Case 1: Boolean target -> cast to int
          if y.dtype == bool:
              df[target_col] = y.astype(int)

          else:
              unique_vals = pd.unique(y.dropna())

              # Case 2: Already binary numeric like {0, 1}
              if len(unique_vals) == 2 and set(unique_vals) <= set([0, 1]):
                  df[target_col] = y.astype(int)

              # Case 3: Two unique numeric values, but not 0/1 (e.g., {1, 2} or {0, 2})
              elif np.issubdtype(y.dtype, np.number) and len(unique_vals) == 2:
                  # Map the smaller value -> 0, larger value -> 1
                  sorted_vals = sorted(unique_vals)
                  mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
                  df[target_col] = y.map(mapping).astype(int)

              # Case 4: Numeric with more than 2 classes -> binarize based on > 0
              elif np.issubdtype(y.dtype, np.number) and len(unique_vals) > 2:
                  df[target_col] = (y > 0).astype(int)

              # Case 5: Categorical / object with 2 classes -> map to 0/1
              elif y.dtype == object and len(unique_vals) == 2:
                  v0 = unique_vals[0]
                  v1 = unique_vals[1]
                  mapping = {v0: 0, v1: 1}
                  df[target_col] = y.map(mapping).astype(int)

              else:
                  # Fallback: binarize non-numeric with more than 2 classes:
                  # mark the most frequent class as 0, others as 1
                  value_counts = y.value_counts()
                  majority_class = value_counts.index[0]
                  df[target_col] = (y != majority_class).astype(int)

          SHARED_GLOBALS['train_data'] = df
          print("Target column '" + str(target_col) + "' has been encoded as 0/1 for logistic regression.")
          ```

        - This guarantees that, after you run this code once in the ACTION phase,
          the target column confirmed during HITL is always an integer 0/1 label.

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

        6. SKEWNESS & LOG TRANSFORM TRACKING (GENERAL):
           - When the critic reports high skewness for a numeric column and suggests a log transform:
             - Always use a shared registry to ensure each column is log-transformed at most ONCE.
             - Example pattern:
               ```python
               import numpy as np

               df = SHARED_GLOBALS['train_data']

               # Ensure registry exists
               log_cols = SHARED_GLOBALS.get('log_transformed_cols')
               if log_cols is None:
                   log_cols = []
                   SHARED_GLOBALS['log_transformed_cols'] = log_cols

               col_name = 'ca'  # replace this with the actual column name you choose based on the critic's report

               if col_name not in log_cols:
                   # Apply a safe log transform
                   df[col_name] = np.log1p(df[col_name])
                   log_cols.append(col_name)
                   SHARED_GLOBALS['log_transformed_cols'] = log_cols
                   print("Applied log transform to column: " + str(col_name) + " and recorded it in log_transformed_cols.")
               else:
                   print("Skipping log transform for column: " + str(col_name) + " because it was already transformed once.")

               SHARED_GLOBALS['train_data'] = df
               ```

        CRITICAL:
        - NEVER hardcode 'target'. Always use `SHARED_GLOBALS.get('target_column')`.
        - Anti-Looping: If Critic complains about the SAME issue twice for the SAME column, you may stop trying to fix it again and proceed to FINALIZE.
        """,
    )