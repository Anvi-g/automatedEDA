from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from typing import Dict, Any
import asyncio
import logging
# Local imports (absolute paths)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.shared_environment import shared_env
from utils.data_flow_manager import data_manager
from config.agent_config import config
# Import agent creators
from agents.basic_eda_agent import create_basic_eda_agent, create_user_choice_agent
from agents.logistic_regression.orchestrator_agent import create_logistic_orchestrator
from agents.logistic_regression.code_executor_agent import create_logistic_code_executor
# Logging
logging.basicConfig(level=logging.INFO)
class MasterAgent:
    """Master coordinator agent for the entire workflow"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
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
            "logistic_code_executor": create_logistic_code_executor()
        }
    
    def _setup_runner(self):
        """Setup the ADK runner"""
        self.runner = InMemoryRunner(
            agent=self.agents["basic_eda"],  # Start with basic EDA
            app_name="data_cleaning_workflow"
        )
    
    async def start_workflow(self, file_path: str) -> Dict[str, Any]:
        """Start the complete data cleaning workflow"""
        try:
            self.logger.info("Starting data cleaning workflow")
            
            # Phase 1: Basic EDA and train/test split
            await self._run_basic_eda(file_path)
            
            # Phase 2: User choice (currently hardcoded to logistic)
            await self._run_user_choice()
            
            # Phase 3: Logistic regression specific EDA
            await self._run_logistic_regression_loop()
            
            # Return final results
            return shared_env.get_readiness_report()
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            raise
    
    async def _run_basic_eda(self, file_path: str):
        """Run basic EDA phase"""
        self.logger.info("Starting Basic EDA phase")
        
        # Create session
        await self.runner.session_service.create_session(
            session_id=f"{self.session_id}_basic_eda",
            app_name="data_cleaning_workflow",
            user_id=self.user_id
        )
        
        # Trigger basic EDA
        user_msg = types.Content(
            role="user",
            parts=[types.Part(text=f"Perform basic EDA on {file_path}")]
        )
        
        async for event in self.runner.run_async(
            session_id=f"{self.session_id}_basic_eda",
            user_id=self.user_id,
            new_message=user_msg
        ):
            if hasattr(event, 'text') and event.text:
                self.logger.info(f"Basic EDA: {event.text}")
    
    async def _run_user_choice(self):
        """Run user choice phase"""
        self.logger.info("Starting User Choice phase")
        
        # For now, hardcoded to logistic regression
        shared_env.update_state("chosen_regression", "logistic")
        self.logger.info("User choice: Logistic Regression")
    
    async def _run_logistic_regression_loop(self):
        """Run logistic regression iterative EDA loop"""
        self.logger.info("Starting Logistic Regression EDA loop")
        
        max_iterations = 10
        iteration = 0
        
        while shared_env.get_state("readiness_score") < config.READINESS_THRESHOLD and iteration < max_iterations:
            iteration += 1
            self.current_iteration = iteration  # Store current iteration
            self.logger.info(f"Logistic EDA iteration {iteration}")
            
            # Run orchestrator
            await self._run_logistic_orchestrator()
            
            # Check if ready
            if shared_env.get_state("readiness_score") >= config.READINESS_THRESHOLD:
                self.logger.info("Dataset ready for logistic regression")
                break
        
        if iteration >= max_iterations:
            self.logger.warning("Max iterations reached, dataset may not be fully ready")
    
    async def _run_logistic_orchestrator(self):
        """Run logistic orchestrator agent"""
        # Switch to logistic orchestrator
        self.runner.agent = self.agents["logistic_orchestrator"]
        
        # Generate unique session ID using stored iteration
        import uuid
        unique_session_id = f"{self.session_id}_logistic_{self.current_iteration}_{uuid.uuid4().hex[:8]}"
        
        await self.runner.session_service.create_session(
            session_id=unique_session_id,
            app_name="data_cleaning_workflow",
            user_id=self.user_id
        )
        
        user_msg = types.Content(
            role="user",
            parts=[types.Part(text="Continue logistic regression EDA")]
        )
        
        async for event in self.runner.run_async(
            session_id=unique_session_id,
            user_id=self.user_id,
            new_message=user_msg
        ):
            if hasattr(event, 'text') and event.text:
                self.logger.info(f"Logistic EDA: {event.text}")
# Global master agent instance
master_agent = MasterAgent()