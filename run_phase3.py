#!/usr/bin/env python3
"""
Run Phase 3: Topic Clustering Evaluation
Simple script to evaluate models on topic clustering task
"""

import os
import sys
from dotenv import load_dotenv


load_dotenv()


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phases.phase3_topic_clustering import Phase3Evaluator

def main():
    """Run Phase 3 evaluation"""
    print("🚀 Starting Phase 3: Topic Clustering Evaluation")
    print("=" * 60)
    
    
    csv_path = os.path.join("data", "Synthetic_Slack_Messages.csv")
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found!")
        print("Please ensure the CSV file exists in the data directory.")
        return
    
    
    required_keys = ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print("⚠️  Warning: Some API keys are missing:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\nYou can still run the evaluation with available models.")
    
    
    try:
        evaluator = Phase3Evaluator()
        evaluator.run_evaluation()
        print("\n✅ Phase 3 evaluation completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
