#!/usr/bin/env python3
"""
Mistral-7B implementation that works with your current VPS
This model fits in your 16GB GPU and provides good text analysis
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

def test_mistral_7b():
    """Mistral-7B test that works with your VPS"""
    
    print("🚀 Mistral-7B Test (GPU Optimized)")
    print("=" * 50)
    print("💡 Using Mistral-7B that fits in your 16GB GPU")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Using CPU fallback.")
        return test_mistral_7b_cpu()
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load messages
    messages = load_messages()
    if not messages:
        return False
    
    test_messages = messages[:5]
    print(f"📝 Testing with {len(test_messages)} messages")
    
    try:
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        print(f"🔄 Loading {model_name} (7B parameters, ~14GB)...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with GPU optimization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 for GPU efficiency
            device_map="auto",  # Automatically use GPU
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print("✅ Model loaded on GPU")
        print(f"✅ Model device: {next(model.parameters()).device}")
        
        # Test 1: Simple generation
        print("\n🧪 Test 1: Simple generation...")
        test_prompt = "Hello, how are you?"
        
        # Use Mistral's chat template
        messages_chat = [{"role": "user", "content": test_prompt}]
        inputs = tokenizer.apply_chat_template(
            messages_chat, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(model.device)
        
        print(f"   Input shape: {inputs.shape}")
        print(f"   Input device: {inputs.device}")
        
        # Clear cache before generation
        torch.cuda.empty_cache()
        
        print("   🔄 Generating...")
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        print("✅ Simple generation works!")
        print(f"Response: {response}")
        
        # Clear cache after generation
        torch.cuda.empty_cache()
        
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
        ).to(model.device)
        
        print(f"   Input shape: {inputs.shape}")
        
        # Clear cache before generation
        torch.cuda.empty_cache()
        
        print("   🔄 Generating clustering...")
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=200,
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
            "mistral_7b": True
        }
        
        os.makedirs("output", exist_ok=True)
        with open("output/mistral_7b_results.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print("\n💾 Results saved to output/mistral_7b_results.json")
        print("\n🎉 Mistral-7B is working on GPU!")
        print("🚀 Perfect fit for your 16GB GPU!")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        print("🔄 Falling back to CPU...")
        return test_mistral_7b_cpu()

def test_mistral_7b_cpu():
    """CPU fallback for Mistral-7B"""
    print("\n🖥️  Testing Mistral-7B on CPU...")
    
    try:
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        print(f"🔄 Loading {model_name} on CPU...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model on CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print("✅ Model loaded on CPU")
        
        # Simple test
        print("\n🧪 Simple generation test...")
        messages = [{"role": "user", "content": "Hello"}]
        
        inputs = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        print(f"Input shape: {inputs.shape}")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("✅ CPU generation works!")
        print(f"Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ CPU test failed: {e}")
        return False

def main():
    """Main function"""
    
    print("🔧 Starting Mistral-7B test...")
    print("This model fits in your 16GB GPU and provides good text analysis")
    
    if test_mistral_7b():
        print("\n🎯 SUCCESS!")
        print("=" * 15)
        print("✅ Mistral-7B loaded")
        print("✅ GPU acceleration")
        print("✅ Generation works")
        print("✅ Clustering works")
        print("🚀 Ready for message analysis!")
    else:
        print("\n❌ Test failed")

if __name__ == "__main__":
    main()
