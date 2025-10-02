#!/usr/bin/env python3
"""
Final test script for Cohere model - avoids downloading issues
"""

import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import os

def main():
    print("🚀 Testing CohereLabs model (Final Version)")
    print("=" * 60)
    
    # Load data
    csv_path = "data/Synthetic_Slack_Messages.csv"
    messages_df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(messages_df)} messages")
    
    # Try different Cohere models in order of preference
    models_to_try = [
        "CohereLabs/c4ai-command-r-v01",  # Smaller 35B model
        "CohereLabs/c4ai-command-r-08-2024",  # 32B model
    ]
    
    for model_id in models_to_try:
        print(f"\n🔍 Trying model: {model_id}")
        
        try:
            # Load with minimal memory usage
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            print("✅ Model loaded successfully!")
            
            # Simple test with first 5 messages
            sample_messages = messages_df["text"].head(5).tolist()
            prompt = f"""Analyze these Slack messages and group them into 6 topic clusters:

Messages:
{chr(10).join([f"Message {i+1}: {msg}" for i, msg in enumerate(sample_messages)])}

Provide your analysis in JSON format."""
            
            # Format as conversation
            messages = [{"role": "user", "content": prompt}]
            
            # Apply chat template
            input_ids = tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                input_ids = input_ids.to(model.device)
            
            print("Generating response...")
            with torch.no_grad():
                gen_tokens = model.generate(
                    input_ids, 
                    max_new_tokens=800, 
                    do_sample=True, 
                    temperature=0.3,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            model_response = response[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
            
            # Save results
            output_path = f"output/cohere_{model_id.replace('/', '_')}_results.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            results = {
                "model": model_id,
                "timestamp": datetime.now().isoformat(),
                "response": model_response,
                "sample_messages": sample_messages
            }
            
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ Results saved to: {output_path}")
            print("\n🎯 Model Response:")
            print("-" * 50)
            print(model_response)
            
            return  # Success, exit after first working model
            
        except Exception as e:
            print(f"❌ Failed with {model_id}: {str(e)}")
            continue
    
    print("❌ All models failed. Try using a different approach or check your internet connection.")

if __name__ == "__main__":
    main()
