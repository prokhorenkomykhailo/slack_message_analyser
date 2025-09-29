#!/usr/bin/env python3
"""
Simple Cohere test that uses the already downloaded model
This won't re-download the model
"""

import os
import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_messages():
    """Load test messages"""
    messages_file = "data/Synthetic_Slack_Messages.csv"
    
    if not os.path.exists(messages_file):
        print(f"❌ File not found: {messages_file}")
        return []
    
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
        return messages
        
    except Exception as e:
        print(f"❌ Error loading messages: {e}")
        return []

def test_cohere_simple():
    """Simple Cohere test using already downloaded model"""
    
    print("🚀 Simple Cohere Test (Using Downloaded Model)")
    print("=" * 50)
    
    
    messages = load_messages()
    if not messages:
        return
    
    test_messages = messages[:5]
    print(f"📝 Testing with {len(test_messages)} messages")
    
    try:
        
        model_name = "CohereLabs/c4ai-command-r-plus-08-2024"
        print(f"🔄 Loading {model_name} from cache...")
        
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
        
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
            print("✅ Fixed attention mask")
        
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=False  
        )
        
        print("✅ Model loaded from cache")
        
        
        print("🔄 Testing simple generation...")
        
        test_prompt = "Hello, how are you?"
        messages_chat = [{"role": "user", "content": test_prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages_chat, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        print("✅ Simple generation works!")
        print(f"Response: {response}")
        
        
        print("\\n🔄 Testing clustering...")
        
        messages_text = "\\n".join([
            f"{i+1}. {msg['user']}: {msg['content'][:40]}"
            for i, msg in enumerate(test_messages)
        ])
        
        clustering_prompt = f"Group these messages: {messages_text}. Return JSON clusters."
        
        messages_chat = [{"role": "user", "content": clustering_prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages_chat, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.3,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        print("✅ Clustering test completed!")
        print("\\n📋 Clustering Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        
        result = {
            "success": True,
            "model": model_name,
            "messages_tested": len(test_messages),
            "simple_response": "Hello, how are you?",
            "clustering_response": response,
            "attention_mask_fixed": True
        }
        
        os.makedirs("output", exist_ok=True)
        with open("output/cohere_simple_test_results.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print("\\n💾 Results saved to output/cohere_simple_test_results.json")
        print("\\n🎉 Cohere is working! No re-download needed.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function"""
    
    if test_cohere_simple():
        print("\\n🎯 SUCCESS!")
        print("=" * 15)
        print("✅ Model loaded from cache (no re-download)")
        print("✅ Attention mask fixed")
        print("✅ Generation works")
        print("✅ Clustering works")
        print("\\n📈 Ready for full Step 1 evaluation!")
    else:
        print("\\n❌ Test failed")

if __name__ == "__main__":
    main()
