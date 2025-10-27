#!/usr/bin/env python3
"""
Quick test script for Phase 3 evaluation with a single model
"""

import os
import json
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_clients import call_model_with_retry
from config.model_config import get_model_cost

def test_single_model():
    """Test a single model to verify setup"""
    
    # Test prompt
    prompt = """
You are an expert at analyzing Slack conversations and grouping messages into coherent topics.

Given a set of Slack messages, your task is to group them into topic clusters.

**Messages to analyze:**
ID: 1 | Channel: #general | User: @alice | Thread: thread_001 | Text: Let's discuss the Q1 project planning...
ID: 2 | Channel: #general | User: @bob | Thread: thread_001 | Text: I agree, we should set clear milestones...
ID: 3 | Channel: #general | User: @charlie | Thread: thread_002 | Text: The budget review is due next week...

**Output Format (JSON):**
{
  "clusters": [
    {
      "cluster_id": "cluster_001",
      "message_ids": [1, 2],
      "draft_title": "Q1 Project Planning",
      "participants": ["@alice", "@bob"],
      "channel": "#general"
    }
  ]
}

Analyze the messages and provide the clustering results in the specified JSON format.
"""

    # Test with a single model (change as needed)
    provider = "openai"
    model_name = "gpt-3.5-turbo"
    
    print(f"🧪 Testing {provider}/{model_name}...")
    
    try:
        # Call model
        result = call_model_with_retry(provider, model_name, prompt, max_retries=3)
        
        if result["success"]:
            print("✅ Success!")
            print(f"Duration: {result['duration']:.2f}s")
            print(f"Response: {result['response'][:200]}...")
            
            # Calculate cost
            cost_config = get_model_cost(provider, model_name)
            usage = result.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            
            input_cost = (input_tokens / 1000) * cost_config.get("input", 0)
            output_cost = (output_tokens / 1000) * cost_config.get("output", 0)
            total_cost = input_cost + output_cost
            
            print(f"Cost: ${total_cost:.6f}")
            print(f"Tokens: {input_tokens} input, {output_tokens} output")
            
        else:
            print("❌ Failed!")
            print(f"Error: {result['error']}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_single_model()
