from typing import Dict, Any
class AgentConfig:
    """Configuration for all agents"""
    
    # Model configuration
    MODEL_NAME = "gemini-2.5-flash"
    
    # Readiness assessment weights
    READINESS_WEIGHTS = {
        "missing_values": 0.25,
        "data_types": 0.15,
        "categorical_encoding": 0.20,
        "feature_scaling": 0.15,
        "outlier_handling": 0.15,
        "class_balance": 0.10
    }
    
    # Readiness thresholds
    READINESS_THRESHOLD = 0.80
    MIN_CATEGORY_SCORE = 0.70
    
    # Logistic regression specific checks
    LOGISTIC_CHECKS = {
        "class_distribution": "analyze_class_balance",
        "feature_correlation": "analyze_feature_correlations",
        "categorical_encoding": "assess_encoding_needs",
        "outlier_detection": "detect_classification_outliers",
        "missing_value_strategy": "determine_imputation_strategy"
    }
# Global configuration instance
config = AgentConfig()