import logging
import sys
from datetime import datetime

# --- ANSI COLOR CODES FOR TERMINAL ---
COLORS = {
    "orchestrator": "\033[94m",      # Blue
    "critic_agent": "\033[93m",      # Yellow
    "advanced_engineer": "\033[92m", # Green
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
        logging.basicConfig(
            filename=filename,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            filemode='w' # Overwrite the log file each run
        )
        self.console = sys.stdout

    def log_event(self, agent_name, event_type, content):
        """
        Logs an event with specific formatting based on the actor.
        
        Args:
            agent_name (str): 'orchestrator', 'critic_agent', etc.
            event_type (str): 'THOUGHT', 'TOOL_CALL', 'TOOL_OUTPUT', 'SYSTEM'
            content (str): The actual message or code.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Clean agent name for lookup (handle None or unexpected casing)
        safe_agent_name = str(agent_name).lower() if agent_name else "unknown"
        
        # --- 1. FILE LOGGING (Persistent Record) ---
        logging.info(f"[{safe_agent_name.upper()}] [{event_type}] {content}")

        # --- 2. CONSOLE LOGGING (Visual Experience) ---
        color = COLORS.get(safe_agent_name, COLORS["RESET"])
        
        # Create a bold header: [TIME] AGENT_NAME
        header = f"{COLORS['BOLD']}{color}[{timestamp}] {safe_agent_name.upper()}{COLORS['RESET']}"
        
        if event_type == "THOUGHT":
            print(f"\n{header} 🤔:")
            print(f"{color}{content}{COLORS['RESET']}")
            
        elif event_type == "TOOL_CALL":
            print(f"\n{header} 🛠️  {COLORS['BOLD']}CALLING TOOL:{COLORS['RESET']}")
            print(f"{COLORS['tool']}{content}{COLORS['RESET']}")
            
        elif event_type == "TOOL_OUTPUT":
            print(f"\n{header} ⚙️  {COLORS['BOLD']}TOOL RESULT:{COLORS['RESET']}")
            # Truncate extremely long tool outputs (like df.info) for the console
            # but keep them full in the file log.
            display_content = content
            if len(content) > 1000:
                display_content = content[:1000] + "\n... [Output Truncated for Console] ..."
            
            print(f"{COLORS['tool']}{display_content}{COLORS['RESET']}")

        elif event_type == "SYSTEM":
            # System messages stand out
            print(f"\n{COLORS['system']}{COLORS['BOLD']}[SYSTEM] {content}{COLORS['RESET']}")
        
        else:
            # Fallback for unknown event types
            print(f"\n{header}: {content}")