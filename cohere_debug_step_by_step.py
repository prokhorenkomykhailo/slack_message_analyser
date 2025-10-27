#!/usr/bin/env python3
"""
Step-by-step debug of Cohere model to find where it hangs
This will test each component separately
"""

import os
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_step_1_tokenizer():
    """Test 1: Just load tokenizer"""
    print("🧪 Test 1: Loading tokenizer...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("CohereLabs/c4ai-command-r-plus-08-2024")
        print("✅ Tokenizer loaded successfully")
        
        # Test basic tokenization
        test_text = "Hello world"
        tokens = tokenizer.encode(test_text)
        print(f"✅ Tokenization works: {len(tokens)} tokens")
        
        return tokenizer
        
    except Exception as e:
        print(f"❌ Tokenizer failed: {e}")
        return None

def test_step_2_model_loading(tokenizer):
    """Test 2: Load model"""
    print("\\n🧪 Test 2: Loading model...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "CohereLabs/c4ai-command-r-plus-08-2024",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def test_step_3_simple_generation(tokenizer, model):
    """Test 3: Simple generation"""
    print("\\n🧪 Test 3: Simple generation...")
    
    try:
        # Very simple input
        inputs = tokenizer("Hello", return_tensors="pt")
        print(f"✅ Input prepared: {inputs.input_ids.shape}")
        
        print("🔄 Generating...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=5,  # Very small
                do_sample=False,   # Greedy
                pad_token_id=tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        print(f"✅ Generation completed in {generation_time:.2f}s")
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple generation failed: {e}")
        return False

def test_step_4_chat_template(tokenizer, model):
    """Test 4: Chat template"""
    print("\\n🧪 Test 4: Chat template...")
    
    try:
        messages = [{"role": "user", "content": "Hello"}]
        
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        print(f"✅ Chat template applied: {inputs.shape}")
        
        print("🔄 Generating with chat template...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=10,  # Small
                do_sample=False,    # Greedy
                pad_token_id=tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        print(f"✅ Chat generation completed in {generation_time:.2f}s")
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Chat response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chat template failed: {e}")
        return False

def test_step_5_clustering_prompt(tokenizer, model):
    """Test 5: Clustering prompt"""
    print("\\n🧪 Test 5: Clustering prompt...")
    
    try:
        # Simple clustering prompt
        prompt = "Group these messages: 1. Alice: Hello 2. Bob: Hi there. Return JSON."
        
        messages = [{"role": "user", "content": prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        print(f"✅ Clustering prompt prepared: {inputs.shape}")
        
        print("🔄 Generating clustering response...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=50,  # Small
                do_sample=True,     # Sampling
                temperature=0.3,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        print(f"✅ Clustering generation completed in {generation_time:.2f}s")
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Clustering response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Clustering prompt failed: {e}")
        return False

def main():
    """Main debug function"""
    
    print("🔧 Cohere Debug - Step by Step")
    print("=" * 35)
    
    # Step 1: Tokenizer
    tokenizer = test_step_1_tokenizer()
    if not tokenizer:
        print("\\n❌ Stopping at tokenizer")
        return
    
    # Step 2: Model
    model = test_step_2_model_loading(tokenizer)
    if not model:
        print("\\n❌ Stopping at model loading")
        return
    
    # Step 3: Simple generation
    if not test_step_3_simple_generation(tokenizer, model):
        print("\\n❌ Stopping at simple generation")
        return
    
    # Step 4: Chat template
    if not test_step_4_chat_template(tokenizer, model):
        print("\\n❌ Stopping at chat template")
        return
    
    # Step 5: Clustering prompt
    if not test_step_5_clustering_prompt(tokenizer, model):
        print("\\n❌ Stopping at clustering prompt")
        return
    
    print("\\n🎉 All tests passed! Cohere is working.")
    print("\\n📋 The issue was likely:")
    print("   - Too many input tokens")
    print("   - Too many output tokens")
    print("   - Complex prompts")
    print("   - Memory issues")
    
    print("\\n💡 Solution: Use smaller inputs and outputs")

if __name__ == "__main__":
    main()
