import logging
import sys
from datetime import datetime
import ast  # NEW: for safely parsing dict-like strings

# --- ANSI COLOR CODES FOR TERMINAL ---
COLORS = {
    "orchestrator": "\033[94m",      # Blue
    "critic_agent": "\033[93m",      # Yellow
    "advanced_engineer": "\033[92m", # Green
    "basiceda_agent": "\033[96m",    # Cyan
    "linear_orchestrator_agent": "\033[94m", # Blue (Alias)
    "tool": "\033[90m",              # Grey (Tool Output)
    "system": "\033[95m",            # Magenta (System Msgs)
    "error": "\033[91m",             # NEW: Red for errors
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

    def _normalize_content(self, content) -> str:
        """NEW: Always return a clean string representation for logging."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return str(content)

    def _extract_result_field(self, content_str: str) -> str:
        """
        NEW: If content looks like {'result': '...'}, return just the result.
        Falls back to the original string if parsing fails.
        """
        try:
            parsed = ast.literal_eval(content_str)
            if isinstance(parsed, dict) and "result" in parsed:
                return str(parsed["result"])
        except Exception:
            pass
        return content_str

    def log_event(self, agent_name, event_type, content):
        """
        Logs an event with specific formatting based on the actor.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Clean agent name
        safe_agent_name = str(agent_name).lower() if agent_name else "unknown"
        
        # --- 1. FILE LOGGING (Full Content) ---
        clean_for_file = self._normalize_content(content)  # NEW
        self.file_logger.info(f"[{safe_agent_name.upper()}] [{event_type}] {clean_for_file}")

        # --- 2. CONSOLE LOGGING (Summarized) ---
        # Map event_type ERROR to red, otherwise agent color
        if event_type == "ERROR":  # NEW
            color = COLORS["error"]
        else:
            color = COLORS.get(safe_agent_name, COLORS["RESET"])
        
        # Header: [TIME] AGENT_NAME
        header = f"{COLORS['BOLD']}{color}[{timestamp}] {safe_agent_name.upper()}{COLORS['RESET']}"
        
        # Normalize content for console too
        content_str = self._normalize_content(content)  # NEW

        if event_type == "THOUGHT":
            print(f"\n{header} 🤔:")
            # NEW: truncate very long thought blocks in console
            lines = content_str.split("\n")
            max_lines = 12  # tweak as you like
            if len(lines) > max_lines:
                preview = "\n".join(lines[:max_lines])
                print(f"{color}{preview}\n... [Thought Truncated. Full text in agent_trace.log] ...{COLORS['RESET']}")
            else:
                print(f"{color}{content_str}{COLORS['RESET']}")
            
        elif event_type == "TOOL_CALL":
            print(f"\n{header} 🛠️  {COLORS['BOLD']}CALLING TOOL:{COLORS['RESET']}")
            
            # --- CONSOLE CLEANUP: Show only first 5 lines of code ---
            lines = content_str.split('\n')
            if len(lines) > 5:
                preview = "\n".join(lines[:5])
                print(f"{COLORS['tool']}{preview}\n... [Code Truncated for Console. See Log] ...{COLORS['RESET']}")
            else:
                print(f"{COLORS['tool']}{content_str}{COLORS['RESET']}")
            
        elif event_type == "TOOL_OUTPUT":
            print(f"\n{header} ⚙️  {COLORS['BOLD']}TOOL RESULT:{COLORS['RESET']}")
            
            # NEW: Extract just the 'result' field if present
            result_text = self._extract_result_field(content_str)

            # --- CONSOLE CLEANUP: Truncate long outputs (df.info, etc) ---
            if len(result_text) > 300:
                short_msg = result_text[:300] + "... [Output Truncated. Full result in agent_trace.log] ..."
                print(f"{COLORS['tool']}{short_msg}{COLORS['RESET']}")
            else:
                print(f"{COLORS['tool']}{result_text}{COLORS['RESET']}")

        elif event_type == "SYSTEM":
            print(f"\n{COLORS['system']}{COLORS['BOLD']}[SYSTEM] {content_str}{COLORS['RESET']}")
        
        elif event_type == "ERROR":  # NEW explicit branch
            print(f"\n{COLORS['error']}{COLORS['BOLD']}[ERROR]{COLORS['RESET']} {COLORS['error']}{content_str}{COLORS['RESET']}")
        
        else:
            print(f"\n{header}: {content_str}")
