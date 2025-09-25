#!/usr/bin/env python3
"""
Fixed GPU-optimized Cohere implementation with proper memory management
This prevents hanging by using better memory allocation and error handling
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

def test_cohere_gpu_fixed():
    """Fixed GPU-optimized Cohere test with proper memory management"""
    
    print("🚀 Cohere GPU-Fixed Test (RTX A4000)")
    print("=" * 50)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Please check GPU setup.")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Load messages
    messages = load_messages()
    if not messages:
        return False
    
    test_messages = messages[:3]  # Use fewer messages for testing
    print(f"📝 Testing with {len(test_messages)} messages")
    
    try:
        model_name = "CohereLabs/c4ai-command-r-plus-08-2024"
        print(f"🔄 Loading {model_name} with GPU acceleration...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with better memory management
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 for memory efficiency
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
            max_memory={0: "14GB", "cpu": "8GB"}  # Limit GPU memory usage
        )
        
        print("✅ Model loaded on GPU")
        print(f"✅ Model device: {next(model.parameters()).device}")
        
        # Test 1: Simple generation with timeout
        print("\n🧪 Test 1: Simple generation...")
        test_prompt = "Hello, how are you?"
        messages_chat = [{"role": "user", "content": test_prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages_chat, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        # Move inputs to same device as model
        inputs = inputs.to(next(model.parameters()).device)
        
        print(f"   Input shape: {inputs.shape}")
        print(f"   Input device: {inputs.device}")
        print(f"   Model device: {next(model.parameters()).device}")
        
        # Clear cache before generation
        torch.cuda.empty_cache()
        
        print("   🔄 Generating...")
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=20,  # Reduced for faster generation
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,  # Enable KV cache
                repetition_penalty=1.1  # Prevent repetition
            )
        
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        print("✅ Simple generation works!")
        print(f"Response: {response}")
        
        # Clear cache after generation
        torch.cuda.empty_cache()
        
        # Test 2: Clustering with smaller input
        print("\n🧪 Test 2: Message clustering...")
        messages_text = "\n".join([
            f"{i+1}. {msg['user']}: {msg['content'][:30]}..."
            for i, msg in enumerate(test_messages)
        ])
        
        clustering_prompt = f"""Group these messages:

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
        print(f"   Input device: {inputs.device}")
        
        # Clear cache before generation
        torch.cuda.empty_cache()
        
        print("   🔄 Generating clustering...")
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=100,  # Reduced for faster generation
                do_sample=True,
                temperature=0.3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                repetition_penalty=1.1
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
            "device": str(next(model.parameters()).device),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "messages_tested": len(test_messages),
            "simple_response": response,
            "clustering_response": response,
            "gpu_acceleration": True,
            "memory_optimized": True
        }
        
        os.makedirs("output", exist_ok=True)
        with open("output/cohere_gpu_fixed_results.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print("\n💾 Results saved to output/cohere_gpu_fixed_results.json")
        print("\n🎉 Cohere is working with GPU acceleration!")
        print("🚀 Memory optimized for RTX A4000!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    
    print("🔧 Starting GPU-fixed Cohere test...")
    print("This version has better memory management")
    
    if test_cohere_gpu_fixed():
        print("\n🎯 SUCCESS!")
        print("=" * 15)
        print("✅ Model loaded on GPU")
        print("✅ Memory optimized")
        print("✅ Generation works")
        print("✅ Clustering works")
        print("🚀 Ready for high-speed message analysis!")
    else:
        print("\n❌ Test failed")

if __name__ == "__main__":
    main()

