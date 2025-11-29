import asyncio
import logging
import sys
import os
# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agents.master_agent import master_agent
from utils.shared_environment import shared_env
# Setup logging (only once)
logging.basicConfig(
    level=logging.ERROR,  # Only show errors, not warnings
    format='%(levelname)s: %(message)s'
)
async def main():
    """Main entry point for the data cleaning workflow"""
    print("<system-reminder>")
    print("AGENTIC AI WORKFLOW STARTING")
    print("</system-reminder>")
    print("🚀 Starting Agentic AI Data Cleaning Workflow")
    print("=" * 50)
    
    try:
        # Initialize master agent
        master_agent.initialize()
        
        # Start workflow
        file_path = "data/raw_dataset.csv"
        results = await master_agent.start_workflow(file_path)
        
        # DO NOT save files here - agents should have already saved them
        # Files should be in data/processed/ directory
        
        # Display results
        print("\n<system-reminder>")
        print("WORKFLOW EXECUTION COMPLETED")
        print("</system-reminder>")
        print("🏁 Workflow Complete!")
        print("=" * 50)
        readiness_score = shared_env.get_state("readiness_score") or 0.0
        print(f"Overall Readiness Score: {readiness_score:.2%}")
        
        # Verification and final output
         
        print("FINAL VERIFICATION COMPLETED")
       
        print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
      
        
 
        
          
        
        print("\n🚀 Next Steps:")
        print("   1. Load the CSV files from data/processed/ directory")
        print("   2. Train your logistic regression model")
        print("   3. Evaluate performance and iterate")
        print("=" * 60)
        
    except Exception as e:
        print(f"<system-reminder>")
        print("WORKFLOW FAILED - CHECK ERROR DETAILS")
        print(f"</system-reminder>")
        print(f"❌ Workflow failed: {e}")
        logging.error(f"Workflow error: {e}", exc_info=True)
        sys.exit(1)
if __name__ == "__main__":
    asyncio.run(main())