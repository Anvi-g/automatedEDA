import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from utils.cleaning_tools import standard_cleaning_tool, load_training_data, run_logistic_code, safe_log1p_column
from utils.hitl_tools import confirm_experiment_setup
class SharedEnvironment:
    """Enhanced shared environment for agent communication"""
    
    def __init__(self):
        self.globals = {
            # Libraries
            "pd": pd,
            "np": np,
            "print": print,
            "os": self._import_os(),
            
            
            # Tools
            "standard_cleaning_tool": standard_cleaning_tool,
            "load_training_data": load_training_data,
            "run_logistic_code": run_logistic_code,
            "confirm_experiment_setup": confirm_experiment_setup,
            "safe_log1p_column": safe_log1p_column, 

            # Data Slots (Initialize to None)
            "raw_data": None,
            "train_data": None,
            "test_data": None,
            "target_column": None,
            "chosen_regression": None,
            
           # State & Reporting
            "readiness_score": 0.0,
            "transformations_applied": [],
            "current_stage": "initialized",
            
            
            "eda_progress": {},   
            "analysis_results": {},
            "data_quality_report": {},

            "log_transformed_cols": [],       
            "critic_messages": [],
        }
        self.globals["SHARED_GLOBALS"] = self.globals

    def add_transformation(self, transformation: str):
        """Track applied transformations"""
        self.globals["transformations_applied"].append({
            "transformation": transformation,
            "timestamp": datetime.now()
        })

    # ADDED: helper to register a log transform
    def register_log_transform(self, column: str):
        """Record that a log transform was applied to a column."""
        log_cols = self.globals.get("log_transformed_cols", [])
        if column not in log_cols:
            log_cols.append(column)
            self.globals["log_transformed_cols"] = log_cols
            self.add_transformation(f"log1p applied to '{column}'")

    def add_critic_message(self, message: str):
        """Append a critic message summary for reporting."""
        msgs = self.globals.get("critic_messages", [])
        msgs.append({
            "message": message,
            "timestamp": datetime.now()
        })
        self.globals["critic_messages"] = msgs

    def update_state(self, key: str, value: Any):
        """Update shared state"""
        self.globals[key] = value
    
    def _import_os(self):
        import os
        return os

    def get_state(self, key: str) -> Any:
        """Get shared state"""
        return self.globals.get(key)
    
    def add_transformation(self, transformation: str):
        """Track applied transformations"""
        self.globals["transformations_applied"].append({
            "transformation": transformation,
            "timestamp": datetime.now()
        })
    
    def update_readiness(self, category: str, score: float):
        """Update readiness score"""
        self.globals["eda_progress"][category] = score
        # Calculate total score from all categories
        if self.globals["eda_progress"]:
            total_score = sum(self.globals["eda_progress"].values()) / len(self.globals["eda_progress"])
            self.globals["readiness_score"] = total_score
            print(f"Readiness score updated to: {total_score:.2%}")
        
    def get_readiness_report(self) -> Dict[str, Any]:
        """Get comprehensive readiness report"""
        return {
            "overall_score": self.globals["readiness_score"],
            "category_scores": self.globals["eda_progress"],
            "transformations": self.globals["transformations_applied"],
            "current_stage": self.globals["current_stage"]
        }
        def save_to_csv(self, data: pd.DataFrame, filename: str, description: str = ""):
            """Save DataFrame to CSV and log the action"""
            import os
            os.makedirs('data/processed', exist_ok=True)
            filepath = f"data/processed/{filename}"
            data.to_csv(filepath, index=False)
            print(f"✅ {description}: Saved to {filepath}")
            print(f"   Shape: {data.shape}")
            print(f"   Columns: {list(data.columns)}")
            self.add_transformation(f"Saved {description} to {filename}")
# Global instance
shared_env = SharedEnvironment()
SHARED_GLOBALS = shared_env.globals