#!/usr/bin/env python3
"""
Simple Cohere test after authentication
"""

import os
import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_cohere_clustering():
    """Test Cohere for Step 1 clustering"""
    
    print("🚀 Testing Cohere Step 1 Clustering")
    print("=" * 40)
    
    
    messages_file = "data/Synthetic_Slack_Messages.csv"
    if not os.path.exists(messages_file):
        print(f"❌ File not found: {messages_file}")
        return
    
    try:
        df = pd.read_csv(messages_file)
        messages = []
        
        for _, row in df.iterrows():
            messages.append({
                "id": row.get("id", len(messages) + 1),
                "user": row.get("user", "Unknown"),
                "content": row.get("content", ""),
                "channel": row.get("channel", "#general")
            })
        
        print(f"✅ Loaded {len(messages)} messages")
        
        
        test_messages = messages[:20]
        
        
        model_name = "CohereLabs/c4ai-command-r-plus-08-2024"
        print(f"🔄 Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ Model loaded")
        
        
        messages_text = "\n".join([
            f"{i+1}. {msg['user']}: {msg['content'][:100]}"
            for i, msg in enumerate(test_messages)
        ])
        
        prompt = f"""Group these messages into topic clusters:

{messages_text}

Return JSON with clusters containing message numbers, titles, and participants."""
        
        
        messages_chat = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages_chat, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        print("🔄 Generating clusters...")
        
        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids,
                max_new_tokens=800,
                do_sample=True,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        
        print("✅ Generation completed")
        print("\n📋 Response:")
        print(response[:500] + "..." if len(response) > 500 else response)
        
        print("\n🎉 Cohere Step 1 clustering test successful!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_cohere_clustering()
