import asyncio
import time
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Import your Agents
from agents import orchestrator_agent
# Import your new Logger
from observability import TraceLogger

APP_NAME = "agents"
USER_ID = "user_01"
SESSION_ID = "session_01"

async def main():
    # 1. Initialize Observability
    logger = TraceLogger()
    logger.log_event("system", "SYSTEM", "🚀 ADEP v2.0 Pipeline Started...")
    logger.log_event("system", "SYSTEM", "   [Orchestrator] <---> [Tools] <---> [Critic]")
    
    # 2. Setup Session
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME, 
        user_id=USER_ID, 
        session_id=SESSION_ID
    )
    
    # 3. Setup Runner
    runner = Runner(
        agent=orchestrator_agent, 
        app_name=APP_NAME,
        session_service=session_service
    )

    # 4. Trigger Prompt
    query = "Start the Linear Regression Data Prep pipeline for 'housing_data.csv'."
    content = types.Content(role="user", parts=[types.Part(text=query)])

    # 5. Run Async Loop
    try:
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            
            agent_name = event.author if event.author else "unknown"
            
            # CRITICAL FIX: Check inside content.parts for everything
            if event.content and event.content.parts:
                for part in event.content.parts:
                    
                    # A. LOG THOUGHTS (Text)
                    if part.text:
                        logger.log_event(agent_name, "THOUGHT", part.text.strip())
                    
                    # B. LOG TOOL CALLS (Function Calls)
                    # The SDK stores tool calls here
                    if part.function_call:
                        func_name = part.function_call.name
                        func_args = part.function_call.args
                        msg = f"{func_name}({func_args})"
                        logger.log_event(agent_name, "TOOL_CALL", msg)

                    # C. LOG TOOL OUTPUTS (Function Responses)
                    # The SDK stores tool results here
                    if part.function_response:
                        resp_name = part.function_response.name
                        resp_content = part.function_response.response
                        msg = f"Result from {resp_name}: {resp_content}"
                        logger.log_event("tool", "TOOL_OUTPUT", str(msg))

            # --- RATE LIMIT PROTECTION ---
            time.sleep(2) 

    except Exception as e:
        logger.log_event("system", "SYSTEM", f"CRITICAL RUNTIME ERROR: {e}")

    logger.log_event("system", "SYSTEM", "🏁 Pipeline Finished. Check 'cleaning_report.md' and 'agent_trace.log'.")

if __name__ == "__main__":
    asyncio.run(main())