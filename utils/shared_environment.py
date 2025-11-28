import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
class SharedEnvironment:
    """Enhanced shared environment for agent communication"""
    
    def __init__(self):
        self.globals = {
            # Core libraries
            "pd": pd,
            "np": np,
            "print": print,
            
            # Data management
            "raw_data": None,
            "train_data": None,
            "test_data": None,
            "processed_data": None,
            
            # State tracking
            "current_stage": "initial",
            "eda_progress": {},
            "readiness_score": 0.0,
            "analysis_start_time": datetime.now(),
            
            # Analysis results
            "analysis_results": {},
            "transformations_applied": [],
            "data_quality_report": {},
            
            # Logistic regression specific
            "class_distribution": {},
            "feature_correlations": {},
            "encoding_decisions": {},
            "outlier_decisions": {}
        }
    
    def update_state(self, key: str, value: Any):
        """Update shared state"""
        self.globals[key] = value
    
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