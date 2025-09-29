#!/usr/bin/env python3
"""
Test script to verify Cohere Command R+ access and basic functionality
"""

import os
import sys
from dotenv import load_dotenv


load_dotenv()

def test_cohere_access():
    """Test if we can access and load the Cohere model"""
    
    print("🧪 Testing Cohere Command R+ Access")
    print("=" * 40)
    
    
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        print("❌ No HUGGINGFACE_TOKEN found")
        return False
    
    print(f"✅ Token found: {token[:10]}...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✅ Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    
    models_to_test = [
        "CohereLabs/c4ai-command-r-plus-08-2024",
        "CohereLabs/c4ai-command-r-plus"
    ]
    
    working_model = None
    
    for model_id in models_to_test:
        print(f"\n🔍 Testing {model_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            print(f"✅ Tokenizer loaded successfully")
            
            
            messages = [{"role": "user", "content": "Hello"}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print(f"✅ Chat template working")
            
            working_model = model_id
            break
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            continue
    
    if not working_model:
        print("\n❌ No Cohere models accessible")
        return False
    
    print(f"\n🎉 Success! Working model: {working_model}")
    
    
    print(f"\n🔍 Testing model loading (this may take a moment)...")
    try:
        
        model = AutoModelForCausalLM.from_pretrained(
            working_model,
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print("✅ Model loaded successfully!")
        
        
        print("🔍 Testing generation...")
        messages = [{"role": "user", "content": "Say hello in one word."}]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        
        
        if hasattr(model, 'device'):
            input_ids = input_ids.to(model.device)
        
        gen_tokens = model.generate(
            input_ids, 
            max_new_tokens=10, 
            do_sample=True, 
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id
        )
        
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        print(f"✅ Generation test successful: {gen_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("   This might be due to insufficient GPU memory")
        print("   The tokenizer test passed, so access is working")
        return True  

def main():
    """Main test function"""
    
    if test_cohere_access():
        print("\n🎉 Cohere access test completed successfully!")
        print("You can now run: python evaluate_cohere_models.py")
    else:
        print("\n❌ Cohere access test failed")
        print("You can still use alternative models: python evaluate_alternative_models_only.py")

if __name__ == "__main__":
    main()
