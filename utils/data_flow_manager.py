from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from .shared_environment import shared_env
class DataFlowManager:
    """Manages data flow and prevents data leakage"""
    
    def __init__(self):
        self.split_performed = False
        self.random_state = 42
    
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """Load raw dataset and store in shared environment"""
        df = pd.read_csv(file_path)
        shared_env.update_state("raw_data", df)
        shared_env.update_state("current_stage", "data_loaded")
        return df
    
    def perform_train_test_split(self, test_size: float = 0.2, target_column: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform train/test split with data leakage prevention"""
        if self.split_performed:
            raise ValueError("Train/test split already performed")
        
        raw_data = shared_env.get_state("raw_data")
        if raw_data is None:
            raise ValueError("No raw data available")
        
        if target_column and target_column in raw_data.columns:
            X = raw_data.drop(columns=[target_column])
            y = raw_data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)
        else:
            train_data, test_data = train_test_split(
                raw_data, test_size=test_size, random_state=self.random_state
            )
        
        shared_env.update_state("train_data", train_data)
        shared_env.update_state("test_data", test_data)
        shared_env.update_state("current_stage", "data_split")
        self.split_performed = True
        
        return train_data, test_data
    
    def get_training_data(self) -> Optional[pd.DataFrame]:
        """Get training data for EDA and processing"""
        return shared_env.get_state("train_data")
    
    def get_testing_data(self) -> Optional[pd.DataFrame]:
        """Get testing data (should not be used for EDA)"""
        return shared_env.get_state("test_data")
    
    def update_processed_data(self, processed_train: pd.DataFrame, processed_test: pd.DataFrame = None):
        """Update processed data after transformations"""
        shared_env.update_state("processed_data", {
            "train": processed_train,
            "test": processed_test
        })
        shared_env.update_state("current_stage", "data_processed")
# Global instance
data_manager = DataFlowManager()