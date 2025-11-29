import sys
from typing import List, Dict

def _console_print(msg: str = ""):
    """
    Print directly to the real console, bypassing any stdout redirection
    (e.g., run_python_code's StringIO).
    """
    real_stdout = getattr(sys, "__stdout__", None) or sys.stdout
    print(msg, file=real_stdout, flush=True)

def confirm_experiment_setup(target_guess: str, type_guess: str, all_columns: List[str]) -> Dict[str, str]:
    """
    Pauses execution to ask the user to confirm BOTH the target and the model type.
    Returns: Dictionary with validated 'target_col' and 'regression_type'.
    """
    _console_print("\n" + "="*60)
    _console_print("🤖 [HITL] EXPERIMENT SETUP")
    _console_print("   The Agent proposes the following configuration:")
    _console_print(f"   1. Target Column:  '{target_guess}'")
    _console_print(f"   2. Modeling Type:  '{type_guess.upper()}' (linear/logistic)")
    _console_print(f"\n📄 Available Columns: {all_columns}")
    _console_print("="*60)

    # 1. Confirm Target 
    while True:
        _console_print(f"👉 Confirm Target (Press ENTER for '{target_guess}'):")
        user_target = input().strip()
        if not user_target:
            final_target = target_guess
            break
        if user_target in all_columns:
            final_target = user_target
            break
        _console_print(f"❌ Error: '{user_target}' is not in the column list.")

    # 2. Confirm Type
    while True:
        _console_print(f"👉 Confirm Type (Press ENTER for '{type_guess}') [linear/logistic]:")
        user_type = input().strip().lower()
        if not user_type:
            final_type = type_guess
            break
        if user_type in ["linear", "logistic"]:
            final_type = user_type
            break
        _console_print("❌ Error: Please type 'linear' or 'logistic'.")

    _console_print(f"✅ SETUP FINALIZED: Target='{final_target}' | Type='{final_type}'")
    _console_print("="*60 + "\n")


   
    
    return {"target_col": final_target, "regression_type": final_type}