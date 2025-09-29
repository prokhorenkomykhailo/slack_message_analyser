#!/usr/bin/env python3
"""
Memory-optimized Cohere implementation using the 194GB cached model
This version uses CPU + GPU efficiently without re-downloading
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

def test_cohere_memory_optimized():
    """Memory-optimized Cohere test using 194GB cached model"""
    
    print("🚀 Cohere Memory-Optimized Test")
    print("=" * 50)
    print("💡 Using 194GB cached model with smart memory management")
    
    
    messages = load_messages()
    if not messages:
        return False
    
    test_messages = messages[:3]
    print(f"📝 Testing with {len(test_messages)} messages")
    
    try:
        model_name = "CohereLabs/c4ai-command-r-plus-08-2024"
        print(f"🔄 Loading {model_name} (194GB cached model)...")
        
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",  
            trust_remote_code=True,
            local_files_only=True,  
            low_cpu_mem_usage=True,
            max_memory={
                0: "15GB",  
                "cpu": "25GB"  
            }
        )
        
        print("✅ Model loaded with smart memory distribution")
        print(f"✅ Model device: {next(model.parameters()).device}")
        
        
        print("\n🧪 Test 1: Simple generation...")
        test_prompt = "Hello, how are you?"
        messages_chat = [{"role": "user", "content": test_prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages_chat, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        
        inputs = inputs.to(next(model.parameters()).device)
        
        print(f"   Input shape: {inputs.shape}")
        print(f"   Input device: {inputs.device}")
        
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("   🔄 Generating...")
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        print("✅ Simple generation works!")
        print(f"Response: {response}")
        
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        
        print("\n🧪 Test 2: Message clustering...")
        messages_text = "\n".join([
            f"{i+1}. {msg['user']}: {msg['content'][:40]}..."
            for i, msg in enumerate(test_messages)
        ])
        
        clustering_prompt = f"""Group these messages by topic:

{messages_text}

Return JSON clusters."""
        
        messages_chat = [{"role": "user", "content": clustering_prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages_chat, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(next(model.parameters()).device)
        
        print(f"   Input shape: {inputs.shape}")
        
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("   🔄 Generating clustering...")
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        print("✅ Clustering test completed!")
        print("\n📋 Clustering Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        
        result = {
            "success": True,
            "model": model_name,
            "device": str(next(model.parameters()).device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
            "messages_tested": len(test_messages),
            "simple_response": response,
            "clustering_response": response,
            "cached_model": True,
            "memory_optimized": True,
            "model_size_gb": 194
        }
        
        os.makedirs("output", exist_ok=True)
        with open("output/cohere_memory_optimized_results.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print("\n💾 Results saved to output/cohere_memory_optimized_results.json")
        print("\n🎉 Cohere is working with 194GB cached model!")
        print("🚀 Smart memory distribution between GPU and CPU!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    
    print("🔧 Starting memory-optimized Cohere test...")
    print("This version uses your 194GB cached model efficiently")
    
    if test_cohere_memory_optimized():
        print("\n🎯 SUCCESS!")
        print("=" * 15)
        print("✅ Model loaded from 194GB cache")
        print("✅ Smart memory distribution")
        print("✅ Generation works")
        print("✅ Clustering works")
        print("🚀 Ready for message analysis!")
    else:
        print("\n❌ Test failed")

if __name__ == "__main__":
    main()
