#!/usr/bin/env python3
"""
Working test for Cohere Command R+ after authentication fix
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_cohere_model():
    """Test Cohere Command R+ model loading and basic functionality"""
    
    model_name = "CohereLabs/c4ai-command-r-plus"
    
    print(f"🧪 Testing {model_name}")
    print("=" * 50)
    
    try:
        # Step 1: Load tokenizer
        print("1️⃣ Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("   ✅ Tokenizer loaded")
        
        # Step 2: Test tokenizer
        print("2️⃣ Testing tokenizer...")
        test_messages = [{"role": "user", "content": "Hello, how are you?"}]
        input_ids = tokenizer.apply_chat_template(
            test_messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        print("   ✅ Chat template applied")
        
        # Step 3: Load model (small test first)
        print("3️⃣ Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("   ✅ Model loaded")
        
        # Step 4: Test generation
        print("4️⃣ Testing generation...")
        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.3,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        print("   ✅ Generation successful")
        print(f"   Response: {response[:100]}...")
        
        print("\n🎉 Cohere Command R+ is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        if "401" in str(e) or "403" in str(e):
            print("\n🔧 Authentication issue detected:")
            print("   1. Make sure you're logged in with the correct account")
            print("   2. Verify you have access to the gated model")
            print("   3. Try: huggingface-cli login")
        elif "gated" in str(e).lower():
            print("\n🔧 Gated model issue:")
            print("   1. Request access at: https://huggingface.co/CohereLabs/c4ai-command-r-plus")
            print("   2. Wait for approval")
        
        return False

if __name__ == "__main__":
    test_cohere_model()
