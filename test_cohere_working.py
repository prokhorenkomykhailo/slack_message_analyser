#!/usr/bin/env python3
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import os

def main():
    print("🚀 Testing CohereLabs c4ai-command-r-plus-08-2024 (Working Model)")
    print("=" * 70)
    
    # Load data
    csv_path = "data/Synthetic_Slack_Messages.csv"
    messages_df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(messages_df)} messages")
    
    # Use the working model from your existing code
    model_id = "CohereLabs/c4ai-command-r-plus-08-2024"
    print(f"Loading model: {model_id}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("Model loaded successfully!")
        
        # Simple test prompt with first 5 messages
        sample_messages = messages_df["text"].head(5).tolist()
        prompt = f"""Analyze these Slack messages and group them into 6 topic clusters:

Messages:
{chr(10).join([f"Message {i+1}: {msg}" for i, msg in enumerate(sample_messages)])}

Please provide your response in JSON format with clusters."""
        
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            input_ids = input_ids.to(model.device)
        
        print("Generating response...")
        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids, 
                max_new_tokens=800, 
                do_sample=True, 
                temperature=0.3,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        model_response = response[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
        
        # Save results
        output_path = "output/cohere_c4ai_command_r_plus_08_2024_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results = {
            "model": "CohereLabs/c4ai-command-r-plus-08-2024",
            "timestamp": datetime.now().isoformat(),
            "response": model_response,
            "sample_messages": sample_messages
        }
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to: {output_path}")
        print("\n🎯 Model Response:")
        print("-" * 50)
        print(model_response)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("The model files might be corrupted. Let me check what models you have available...")
        
        # List available models
        import glob
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
        cohere_models = glob.glob(f"{cache_dir}models--CohereLabs--*")
        print(f"\nAvailable CohereLabs models in cache:")
        for model_path in cohere_models:
            model_name = model_path.split("--")[-1]
            print(f"  - {model_name}")

if __name__ == "__main__":
    main()
