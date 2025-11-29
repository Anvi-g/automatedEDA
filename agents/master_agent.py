from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from typing import Dict, Any
import asyncio
import time
import logging
import sys
import os
import uuid
# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.shared_environment import shared_env
from config.agent_config import config
# Import agent creators
from agents.basic_eda_agent import create_basic_eda_agent, create_user_choice_agent
from agents.logistic_regression.orchestrator_agent import create_logistic_orchestrator
from agents.logistic_regression.code_executor_agent import create_logistic_code_executor
from agents.linear_regression.orchestrator_agent import create_linear_orchestrator
# Logging
from observability import TraceLogger
class MasterAgent:
    """Master coordinator agent for the entire workflow"""
    
    def __init__(self):
        self.trace = TraceLogger()
        self.agents = {}
        self.runner = None
        self.session_id = "master_session"
        self.user_id = "data_user"
        self.current_iteration = 0  # Track current iteration
    
    def initialize(self):
        """Initialize the master agent and sub-agents"""
        self._setup_agents()
        self._setup_runner()
    
    def _setup_agents(self):
        """Setup all sub-agents"""
        self.agents = {
            "basic_eda": create_basic_eda_agent(),
            "user_choice": create_user_choice_agent(),
            "logistic_orchestrator": create_logistic_orchestrator(),
            "logistic_code_executor": create_logistic_code_executor(),
            "linear_orchestrator": create_linear_orchestrator(),
        }
    
    def _setup_runner(self):
        """Setup the ADK runner"""
        self.runner = InMemoryRunner(
            agent=self.agents["basic_eda"],  # Start with basic EDA
            app_name="agents"
        )
    
    async def _run_observable_loop(self, session_id, user_msg):
        """
        Runs the agent loop and pipes everything to TraceLogger.
        Replaces standard logging loops.
        """
        try:
            async for event in self.runner.run_async(session_id=session_id, user_id=self.user_id, new_message=user_msg):
                agent_name = event.author if event.author else "unknown"
                
                # Check inside content.parts for thoughts & tools
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        # 1. Thoughts
                        if part.text:
                            self.trace.log_event(agent_name, "THOUGHT", part.text.strip())
                        
                        # 2. Tool Calls
                        if part.function_call:
                            msg = f"{part.function_call.name}({part.function_call.args})"
                            self.trace.log_event(agent_name, "TOOL_CALL", msg)

                        # 3. Tool Outputs
                        if part.function_response:
                            res = part.function_response.response
                            self.trace.log_event("tool", "TOOL_OUTPUT", str(res))
                time.sleep(2)
                            
        except Exception as e:
            self.trace.log_event("system", "ERROR", str(e))

    async def start_workflow(self, file_path: str) -> Dict[str, Any]:
        self.trace.log_event("system", "SYSTEM", "🚀 Starting Master Workflow")
        
        try:
            # Phase 1: Basic EDA
            await self._run_basic_eda(file_path)
            
            # Phase 2: Routing
            chosen_type = shared_env.get_state("chosen_regression")

            if not chosen_type:
                self.trace.log_event("system", "WARNING", "Agent failed to set regression type. Attempting Auto-Detection...")
                
                # Retrieve data from shared memory
                train_data = shared_env.get_state("train_data")
                target_col = shared_env.get_state("target_column")
                
                if train_data is not None and target_col is not None:
                    import pandas as pd
                    # Logic: Few unique values = Classification (Logistic), Many = Regression (Linear)
                    n_unique = train_data[target_col].nunique()
                    dtype = train_data[target_col].dtype
                    
                    if n_unique < 20 or pd.api.types.is_object_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
                        chosen_type = "logistic"
                    else:
                        chosen_type = "linear"
                    
                    # Save the decision back to memory
                    shared_env.update_state("chosen_regression", chosen_type)
                    self.trace.log_event("system", "SYSTEM", f"Auto-Detected Target '{target_col}' ({n_unique} unique vals). Selected: {chosen_type.upper()}")
                else:
                    # Absolute worst case fallback
                    chosen_type = "linear"
                    self.trace.log_event("system", "ERROR", "Could not inspect data. Defaulting to LINEAR.")
      

            self.trace.log_event("system", "SYSTEM", f"Routing to: {str(chosen_type).upper()} REGRESSION")

            if chosen_type == "logistic":
                await self._run_logistic_regression_loop()
            elif chosen_type == "linear":
                await self._run_linear_regression_loop()
            else:
                self.trace.log_event("system", "ERROR", f"Unknown or missing regression type: {chosen_type}")
                
            return shared_env.get_readiness_report()
            
        except Exception as e:
            self.trace.log_event("system", "ERROR", f"Workflow failed: {e}")
            # Raise to stop main.py execution
            raise

    async def _run_basic_eda(self, file_path: str):
        self.trace.log_event("system", "SYSTEM", "--- Phase 1: Basic EDA ---")
        
        sess_id = f"{self.session_id}_basic_eda"
        await self.runner.session_service.create_session(
            session_id=sess_id, 
            app_name="agents", 
            user_id=self.user_id
        )
        
        user_msg = types.Content(role="user", parts=[types.Part(text=f"Perform basic EDA on {file_path}")])
        
        await self._run_observable_loop(sess_id, user_msg)

    async def _run_linear_regression_loop(self):
        self.trace.log_event("system", "SYSTEM", "--- Phase 3: Linear Regression Optimization ---")
        
        # Switch Agent
        self.runner.agent = self.agents["linear_orchestrator"]
        
        sess_id = f"{self.session_id}_linear_{uuid.uuid4().hex[:8]}"
        await self.runner.session_service.create_session(
            session_id=sess_id, 
            app_name="agents", 
            user_id=self.user_id
        )
        
        # Trigger
        user_msg = types.Content(role="user", parts=[types.Part(text="Start Linear Optimization on SHARED_GLOBALS['train_data']")])
        await self._run_observable_loop(sess_id, user_msg)

    async def _run_logistic_regression_loop(self):
        self.trace.log_event("system", "SYSTEM", "--- Phase 3: Logistic Regression Optimization ---")
        
        self.runner.agent = self.agents["logistic_orchestrator"]
        
        iteration = 0
        readiness = shared_env.get_state("readiness_score")
        if readiness is None: readiness = 0.0
        
        while readiness < 0.8 and iteration < 5:
            iteration += 1
            self.trace.log_event("system", "SYSTEM", f"Logistic Iteration {iteration}")
            
            sess_id = f"{self.session_id}_logistic_{iteration}"
            await self.runner.session_service.create_session(
                session_id=sess_id, 
                app_name="agents", 
                user_id=self.user_id
            )
            
            user_msg = types.Content(role="user", parts=[types.Part(text="Optimize for Logistic Regression")])
            await self._run_observable_loop(sess_id, user_msg)
            
            readiness = shared_env.get_state("readiness_score") or 0.0

# Global Instance
master_agent = MasterAgent()