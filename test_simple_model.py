#!/usr/bin/env python3
"""
Simple test with a smaller, faster model
"""

import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import os

def main():
    print("🚀 Testing with a smaller, faster model")
    print("=" * 50)
    
    # Load data
    csv_path = "data/Synthetic_Slack_Messages.csv"
    messages_df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(messages_df)} messages")
    
    # Use a smaller, faster model
    model_id = "microsoft/DialoGPT-medium"  # Much smaller model
    print(f"Loading model: {model_id}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        
        print("Model loaded successfully!")
        
        # Simple test with first 3 messages
        sample_messages = messages_df["text"].head(3).tolist()
        prompt = f"Analyze these messages and group them into topics: {' '.join(sample_messages)}"
        
        # Tokenize
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate
        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_new_tokens=200, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Save results
        output_path = "output/simple_model_test_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results = {
            "model": model_id,
            "timestamp": datetime.now().isoformat(),
            "response": response,
            "sample_messages": sample_messages
        }
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to: {output_path}")
        print("\n�� Model Response:")
        print("-" * 40)
        print(response)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
