# agents.py
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool, FunctionTool
from tools import run_local_python

# ---------------------------------------------------------
# Agent 2: Code Executor (The Worker)
# ---------------------------------------------------------
executor_agent = LlmAgent(
    name="code_executor",
    # CHANGED: Use specific experimental version for 2.0
    model="gemini-2.0-flash-lite-001", 
    instruction="""
    You are a Python Data Engineer.
    
    CAPABILITIES:
    1. LIST FILES: Run "import os; print(os.listdir('.'))" to see files.
    2. CLEAN DATA: Load and manipulate pandas dataframes.
    
    RULES:
    - Receive instructions and code from the Orchestrator.
    - Write Python code to manipulate 'housing_data.csv' (or whatever file is found).
    - Execute it using the 'run_local_python' tool.
    - ALWAYS print df.info() or df.head() to verify changes.
    """,
    # CHANGED: Swapped 'code_executor=' for 'tools=[]' with our local function
    # This allows the agent to touch the actual 'housing_data.csv' file.
    tools=[FunctionTool(run_local_python)] 
)

# ---------------------------------------------------------
# Agent 1: Orchestrator (The Planner)
# ---------------------------------------------------------
orchestrator_agent = LlmAgent(
    name="orchestrator",
    # CHANGED: Use 1.5-pro-001 (Stable) for better planning/reasoning
    model="gemini-2.0-flash-lite-001",
    instruction="""
    You are a Senior Data Scientist managing a data cleaning pipeline.
    
    GOAL: Find a CSV file in the root directory, clean it, and save the result.
    
    STEP-BY-STEP LOOP:
    1. DISCOVER: Ask executor to list files in current directory (".").
    2. TARGET: Identify the CSV file (e.g., 'housing_data.csv').
    3. ANALYZE: Ask executor to load that file and show info().
    4. PLAN: Decide the next cleaning step (NaNs, Outliers).
    5. DELEGATE: Call the 'code_executor' to perform the step.
    
    CRITICAL:
    - Do not guess the filename. Find it first.
    - Keep looping until the data is clean.
    - Output "DATA CLEANING COMPLETE" only when 'cleaned_data.csv' is saved.
    """,
    tools=[AgentTool(agent=executor_agent)]
)