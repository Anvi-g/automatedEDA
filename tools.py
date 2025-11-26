# tools.py
import subprocess
import sys

def run_local_python(code: str) -> str:
    """
    Executes Python code locally in the current directory.
    Useful for reading local files like CSVs.
    """
    # Write code to a temp file
    with open("temp_worker_script.py", "w") as f:
        f.write(code)
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, "temp_worker_script.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]:\n{result.stderr}"
            
        return output if output.strip() else "(Code ran successfully, no output)"
        
    except Exception as e:
        return f"Execution Failed: {e}"