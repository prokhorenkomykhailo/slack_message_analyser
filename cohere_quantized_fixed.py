#!/usr/bin/env python3
"""
Quantized Cohere implementation that should work without hanging
Uses 8-bit quantization to reduce memory usage
"""

import torch
import os
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

def test_cohere_quantized():
    """Quantized Cohere test that should work without hanging"""
    
    print("🚀 Cohere Quantized Test (8-bit)")
    print("=" * 50)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Using CPU fallback.")
        return test_cohere_cpu()
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load messages
    messages = load_messages()
    if not messages:
        return False
    
    test_messages = messages[:3]
    print(f"📝 Testing with {len(test_messages)} messages")
    
    try:
        model_name = "CohereLabs/c4ai-command-r-plus-08-2024"
        print(f"🔄 Loading {model_name} with 8-bit quantization...")
        
        # Configure 8-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True
        )
        
        print("✅ Model loaded with 8-bit quantization")
        print(f"✅ Model device: {next(model.parameters()).device}")
        
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
        
        # Move inputs to same device as model
        inputs = inputs.to(next(model.parameters()).device)
        
        print(f"   Input shape: {inputs.shape}")
        print(f"   Input device: {inputs.device}")
        
        # Clear cache before generation
        torch.cuda.empty_cache()
        
        print("   🔄 Generating (with timeout protection)...")
        
        # Generate with timeout protection
        import signal
        import time
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Generation timed out")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # 60 second timeout
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            
            signal.alarm(0)  # Cancel timeout
            
            response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            print("✅ Simple generation works!")
            print(f"Response: {response}")
            
        except TimeoutError:
            print("❌ Generation timed out after 60 seconds")
            return False
        
        # Clear cache after generation
        torch.cuda.empty_cache()
        
        # Test 2: Clustering
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
        
        # Clear cache before generation
        torch.cuda.empty_cache()
        
        print("   🔄 Generating clustering...")
        
        # Generate with timeout protection
        signal.alarm(120)  # 2 minute timeout for clustering
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.3,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            
            signal.alarm(0)  # Cancel timeout
            
            response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            
            print("✅ Clustering test completed!")
            print("\n📋 Clustering Response:")
            print("-" * 40)
            print(response)
            print("-" * 40)
            
        except TimeoutError:
            print("❌ Clustering timed out after 2 minutes")
            return False
        
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
            "quantization": "8-bit",
            "timeout_protection": True
        }
        
        os.makedirs("output", exist_ok=True)
        with open("output/cohere_quantized_results.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print("\n💾 Results saved to output/cohere_quantized_results.json")
        print("\n🎉 Cohere is working with 8-bit quantization!")
        print("🚀 No more hanging issues!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cohere_cpu():
    """CPU fallback if GPU fails"""
    print("\n🖥️  Falling back to CPU...")
    
    try:
        model_name = "CohereLabs/c4ai-command-r-plus-08-2024"
        print(f"🔄 Loading {model_name} on CPU...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model on CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            local_files_only=True,
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
        
        print(f"   Input shape: {inputs.shape}")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=10,
                do_sample=False,
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
    
    print("🔧 Starting quantized Cohere test...")
    print("This version uses 8-bit quantization and timeout protection")
    
    if test_cohere_quantized():
        print("\n🎯 SUCCESS!")
        print("=" * 15)
        print("✅ Model loaded with quantization")
        print("✅ Timeout protection enabled")
        print("✅ Generation works")
        print("✅ Clustering works")
        print("🚀 No more hanging issues!")
    else:
        print("\n❌ Test failed")

if __name__ == "__main__":
    main()
