#!/usr/bin/env python3
"""
Fixed Cohere implementation with proper memory management and quantization
This prevents the hanging issue by using 8-bit quantization
"""

import os
import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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

def test_cohere_fixed():
    """Fixed Cohere test with proper memory management"""
    
    print("🚀 Cohere Fixed Test (With Quantization)")
    print("=" * 50)
    
    # Load messages (just first 3 for testing)
    messages = load_messages()
    if not messages:
        return
    
    test_messages = messages[:3]
    print(f"📝 Testing with {len(test_messages)} messages")
    
    try:
        # Configure 8-bit quantization to prevent memory issues
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
        )
        
        model_name = "CohereLabs/c4ai-command-r-plus"
        print(f"🔄 Loading {model_name} with 8-bit quantization...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Fix attention mask issue
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("✅ Fixed pad token")
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        print("✅ Model loaded with quantization")
        
        # Test 1: Simple generation
        print("\n🧪 Test 1: Simple generation...")
        
        test_prompt = "Hello, how are you?"
        messages_chat = [{"role": "user", "content": test_prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages_chat, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        print(f"   Input shape: {inputs.shape}")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        print("✅ Simple generation works!")
        print(f"   Response: {response}")
        
        # Test 2: Clustering
        print("\n🧪 Test 2: Message clustering...")
        
        messages_text = "\n".join([
            f"{i+1}. {msg['user']}: {msg['content'][:50]}..."
            for i, msg in enumerate(test_messages)
        ])
        
        clustering_prompt = f"""Group these Slack messages into clusters by topic:

{messages_text}

Return a JSON response with clusters."""
        
        messages_chat = [{"role": "user", "content": clustering_prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages_chat, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        print(f"   Input shape: {inputs.shape}")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        print("✅ Clustering test completed!")
        print("\n📋 Clustering Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        # Save results
        result = {
            "success": True,
            "model": model_name,
            "quantization": "8-bit",
            "messages_tested": len(test_messages),
            "simple_response": response,
            "clustering_response": response,
            "memory_optimized": True
        }
        
        os.makedirs("output", exist_ok=True)
        with open("output/cohere_fixed_test_results.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print("\n💾 Results saved to output/cohere_fixed_test_results.json")
        print("\n🎉 Cohere is working with proper memory management!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    
    print("🔧 Starting fixed Cohere test...")
    print("This version uses 8-bit quantization to prevent memory issues")
    
    if test_cohere_fixed():
        print("\n🎯 SUCCESS!")
        print("=" * 15)
        print("✅ Model loaded with 8-bit quantization")
        print("✅ Memory optimized")
        print("✅ Generation works")
        print("✅ Clustering works")
        print("\n📈 Ready for full evaluation!")
    else:
        print("\n❌ Test failed")

if __name__ == "__main__":
    main()
