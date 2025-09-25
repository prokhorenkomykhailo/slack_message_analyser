#!/usr/bin/env python3
"""
Minimal Cohere implementation following Hugging Face documentation
This uses the exact pattern from the official docs with 8-bit quantization
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def main():
    print("🚀 Cohere Minimal Test (No Hanging)")
    print("=" * 40)
    
    try:
        # Configure 8-bit quantization to prevent memory issues
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
        )
        
        model_id = "CohereLabs/c4ai-command-r-plus-08-2024"
        print(f"🔄 Loading tokenizer from {model_id} (using cached version)...")
        
        # Load tokenizer from cache
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        print(f"   pad_token: {tokenizer.pad_token}")
        print(f"   eos_token: {tokenizer.eos_token}")
        
        print("🔄 Loading model...")
        
        # Load model with quantization from cache
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        
        print("✅ Model loaded")
        
        # Format message with the command-r-plus chat template
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        input_ids = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        print(f"🧪 Test 1: Simple generation...")
        print(f"   Input shape: {input_ids.shape}")
        
        # Generate with proper parameters
        gen_tokens = model.generate(
            input_ids, 
            max_new_tokens=50, 
            do_sample=True, 
            temperature=0.3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        print("✅ Generation successful!")
        print(f"Response: {gen_text}")
        
        print("\n🎉 Cohere is working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()