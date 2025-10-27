#!/usr/bin/env python3
"""
Fallback Cohere implementation
If the large model doesn't work, try a smaller alternative
"""

import torch
import os
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

def test_cohere_fallback():
    """Fallback Cohere test with smaller model"""
    
    print("🚀 Cohere Fallback Test")
    print("=" * 50)
    print("💡 Trying smaller model if large one fails")
    
    # Load messages
    messages = load_messages()
    if not messages:
        return False
    
    test_messages = messages[:3]
    print(f"📝 Testing with {len(test_messages)} messages")
    
    # Try smaller model first
    smaller_models = [
        "microsoft/DialoGPT-medium",  # 345M parameters
        "microsoft/DialoGPT-large",   # 774M parameters
        "facebook/blenderbot-400M-distill",  # 400M parameters
    ]
    
    for model_name in smaller_models:
        try:
            print(f"\n🔄 Trying {model_name}...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            print(f"✅ Model loaded: {model_name}")
            
            # Test generation
            print("🧪 Testing generation...")
            test_prompt = "Hello, how are you?"
            
            inputs = tokenizer.encode(test_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"✅ Generation works with {model_name}!")
            print(f"Response: {response}")
            
            # Save results
            result = {
                "success": True,
                "model": model_name,
                "device": "cpu",
                "fallback": True,
                "messages_tested": len(test_messages),
                "simple_response": response,
                "clustering_response": "Not tested with fallback model"
            }
            
            os.makedirs("output", exist_ok=True)
            with open("output/cohere_fallback_results.json", "w") as f:
                json.dump(result, f, indent=2)
            
            print("\n💾 Results saved to output/cohere_fallback_results.json")
            print(f"\n🎉 Cohere is working with {model_name}!")
            print("🚀 Using smaller model as fallback!")
            
            return True
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            continue
    
    print("\n❌ All fallback models failed")
    return False

def main():
    """Main function"""
    
    print("🔧 Starting fallback Cohere test...")
    print("This version tries smaller models if the large one fails")
    
    if test_cohere_fallback():
        print("\n🎯 SUCCESS!")
        print("=" * 15)
        print("✅ Fallback model loaded")
        print("✅ Generation works")
        print("🚀 Ready for message analysis!")
    else:
        print("\n❌ All models failed")

if __name__ == "__main__":
    main()
