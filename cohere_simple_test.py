#!/usr/bin/env python3
"""
Simple test to debug the hanging issue
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def simple_test():
    print("🔍 Simple Cohere Test (Debug Mode)")
    print("=" * 40)
    
    try:
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        model_name = "CohereLabs/c4ai-command-r-plus-08-2024"
        print(f"\n🔄 Loading {model_name}...")
        
        
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
            max_memory={0: "12GB"}  
        )
        
        print("✅ Model loaded")
        print(f"Model device: {next(model.parameters()).device}")
        
        
        print("\n🧪 Simple generation test...")
        messages = [{"role": "user", "content": "Hi"}]
        
        inputs = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        
        inputs = inputs.to(next(model.parameters()).device)
        print(f"Input device: {inputs.device}")
        print(f"Input shape: {inputs.shape}")
        
        
        torch.cuda.empty_cache()
        
        print("🔄 Generating (this should be fast)...")
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=5,  
                do_sample=False,   
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("✅ Generation successful!")
        print(f"Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    simple_test()
