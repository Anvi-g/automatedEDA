import logging
import sys
from datetime import datetime

# --- ANSI COLOR CODES FOR TERMINAL ---
COLORS = {
    "orchestrator": "\033[94m",      # Blue
    "critic_agent": "\033[93m",      # Yellow
    "advanced_engineer": "\033[92m", # Green
    "basiceda_agent": "\033[96m",    # Cyan
    "linear_orchestrator_agent": "\033[94m", # Blue (Alias)
    "tool": "\033[90m",              # Grey (Tool Output)
    "system": "\033[95m",            # Magenta (System Msgs)
    "RESET": "\033[0m",
    "BOLD": "\033[1m"
}

class TraceLogger:
    def __init__(self, filename="agent_trace.log"):
        """
        Initializes the logger to write to both a file and the console.
        """
        # 1. Setup File Logging (Clean text, no colors)
        # We use a unique logger name to avoid conflicts with other libs
        self.file_logger = logging.getLogger("ADEP_File_Logger")
        self.file_logger.setLevel(logging.INFO)
        # Remove existing handlers to prevent duplicates if re-initialized
        if self.file_logger.hasHandlers():
            self.file_logger.handlers.clear()
            
        file_handler = logging.FileHandler(filename, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
        file_handler.setFormatter(formatter)
        self.file_logger.addHandler(file_handler)

    def log_event(self, agent_name, event_type, content):
        """
        Logs an event with specific formatting based on the actor.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Clean agent name
        safe_agent_name = str(agent_name).lower() if agent_name else "unknown"
        
        # --- 1. FILE LOGGING (Full Content) ---
        self.file_logger.info(f"[{safe_agent_name.upper()}] [{event_type}] {content}")

        # --- 2. CONSOLE LOGGING (Summarized) ---
        color = COLORS.get(safe_agent_name, COLORS["RESET"])
        
        # Header: [TIME] AGENT_NAME
        header = f"{COLORS['BOLD']}{color}[{timestamp}] {safe_agent_name.upper()}{COLORS['RESET']}"
        
        if event_type == "THOUGHT":
            print(f"\n{header} 🤔:")
            print(f"{color}{content}{COLORS['RESET']}")
            
        elif event_type == "TOOL_CALL":
            print(f"\n{header} 🛠️  {COLORS['BOLD']}CALLING TOOL:{COLORS['RESET']}")
            
            # --- CONSOLE CLEANUP: Show only first 5 lines of code ---
            lines = str(content).split('\n')
            if len(lines) > 5:
                preview = "\n".join(lines[:5])
                print(f"{COLORS['tool']}{preview}\n... [Code Truncated for Console. See Log] ...{COLORS['RESET']}")
            else:
                print(f"{COLORS['tool']}{content}{COLORS['RESET']}")
            
        elif event_type == "TOOL_OUTPUT":
            print(f"\n{header} ⚙️  {COLORS['BOLD']}TOOL RESULT:{COLORS['RESET']}")
            
            # --- CONSOLE CLEANUP: Truncate long outputs (df.info, etc) ---
            clean_content = str(content)
            if len(clean_content) > 300:
                short_msg = clean_content[:300] + f"... [Output Truncated. Full result in agent_trace.log] ..."
                print(f"{COLORS['tool']}{short_msg}{COLORS['RESET']}")
            else:
                print(f"{COLORS['tool']}{clean_content}{COLORS['RESET']}")

        elif event_type == "SYSTEM":
            print(f"\n{COLORS['system']}{COLORS['BOLD']}[SYSTEM] {content}{COLORS['RESET']}")
        
        else:
            print(f"\n{header}: {content}")