#!/usr/bin/env python3
"""
Test CohereLabs c4ai-command-r-plus-4bit with CPU offloading
"""

import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from datetime import datetime
import os

def main():
    print("🚀 Testing CohereLabs c4ai-command-r-plus-4bit with CPU offloading")
    print("=" * 70)
    
    # Load data
    csv_path = "data/Synthetic_Slack_Messages.csv"
    messages_df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(messages_df)} messages")
    
    model_id = "CohereLabs/c4ai-command-r-plus-4bit"
    print(f"Loading model: {model_id}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        print("✅ Tokenizer loaded")
        
        # Configure 4-bit quantization with CPU offloading
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading
        )
        
        # Load model with CPU offloading
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=True,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        print("✅ Model loaded with CPU offloading!")
        
        # Simple test with first 3 messages
        sample_messages = messages_df["text"].head(3).tolist()
        prompt = f"""Analyze these Slack messages and group them into topic clusters:

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
        
        print("Generating response...")
        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids, 
                max_new_tokens=500, 
                do_sample=True, 
                temperature=0.3,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        model_response = response[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
        
        # Save results
        output_path = "output/cohere_c4ai_command_r_plus_4bit_cpu_offload_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results = {
            "model": model_id,
            "timestamp": datetime.now().isoformat(),
            "response": model_response,
            "sample_messages": sample_messages,
            "method": "CPU offloading"
        }
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to: {output_path}")
        print("\n🎯 Model Response:")
        print("-" * 50)
        print(model_response)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("The model might still be too large even with CPU offloading.")

if __name__ == "__main__":
    main()
