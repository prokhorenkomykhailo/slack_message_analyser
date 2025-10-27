#!/usr/bin/env python3
"""
Working clustering test with DialoGPT-medium (already cached and working)
"""

import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import os

def main():
    print("🚀 Testing Slack Message Clustering with DialoGPT-medium")
    print("=" * 60)
    
    # Load data
    csv_path = "data/Synthetic_Slack_Messages.csv"
    messages_df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(messages_df)} messages")
    
    # Use the working DialoGPT model
    model_id = "microsoft/DialoGPT-medium"
    print(f"Loading model: {model_id}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        
        # Add padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Model loaded successfully!")
        
        # Test with first 10 messages
        sample_messages = messages_df["text"].head(10).tolist()
        
        # Create a clustering prompt
        prompt = f"""Analyze these Slack messages and group them into 6 topic clusters:

Messages:
{chr(10).join([f"Message {i+1}: {msg}" for i, msg in enumerate(sample_messages)])}

Group them into 6 clusters and provide JSON format:
{{
  "clusters": [
    {{"cluster_id": "topic_001", "message_ids": [1, 2], "title": "Topic description"}},
    {{"cluster_id": "topic_002", "message_ids": [3, 4], "title": "Topic description"}}
  ]
}}"""
        
        # Tokenize
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
        
        print("Generating clustering response...")
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_new_tokens=300, 
                do_sample=True, 
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated part
        model_response = response[len(prompt):]
        
        # Save results
        output_path = "output/dialoGPT_clustering_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results = {
            "model": model_id,
            "timestamp": datetime.now().isoformat(),
            "response": model_response,
            "sample_messages": sample_messages,
            "method": "DialoGPT-medium clustering"
        }
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to: {output_path}")
        print("\n🎯 Clustering Analysis:")
        print("-" * 50)
        print(model_response)
        
        # Show comparison with your existing benchmarks
        print("\n📊 Comparison with your benchmarks:")
        print("Step 1 (6 clusters): phases/phase3_clusters.json")
        print("Step 2 (15 clusters): phases/phase4_clusters_refined.json")
        print("This test: DialoGPT-medium clustering")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
