#!/usr/bin/env python3
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import os

def main():
    print("🚀 Testing CohereLabs c4ai-command-r-plus-4bit (44GB cached)")
    print("=" * 60)
    
    # Load data
    csv_path = "data/Synthetic_Slack_Messages.csv"
    messages_df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(messages_df)} messages")
    
    # Load model from cache
    model_id = "CohereLabs/c4ai-command-r-plus-4bit"
    print(f"Loading model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Model loaded successfully!")
    
    # Simple test prompt
    prompt = "Analyze these Slack messages and group them into 6 topic clusters. Messages: " + " ".join(messages_df["text"].head(10).tolist())
    
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    
    if torch.cuda.is_available():
        input_ids = input_ids.to(model.device)
    
    print("Generating response...")
    with torch.no_grad():
        gen_tokens = model.generate(input_ids, max_new_tokens=500, do_sample=True, temperature=0.3)
    
    response = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    model_response = response[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
    
    # Save results
    output_path = "output/cohere_c4ai_command_r_plus_4bit_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    results = {
        "model": "CohereLabs/c4ai-command-r-plus-4bit",
        "timestamp": datetime.now().isoformat(),
        "response": model_response
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {output_path}")
    print("\n🎯 Model Response:")
    print("-" * 40)
    print(model_response)

if __name__ == "__main__":
    main()
