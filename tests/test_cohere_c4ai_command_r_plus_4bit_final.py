#!/usr/bin/env python3
"""
Test script for CohereLabs c4ai-command-r-plus-4bit model
for Slack message clustering evaluation
"""

import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import os

def load_slack_data(csv_path):
    """Load Slack messages from CSV"""
    df = pd.read_csv(csv_path)
    return df

def create_clustering_prompt(messages_df, num_clusters=6):
    """Create prompt for clustering task"""
    # Use first 50 messages to manage context length
    sample_size = min(50, len(messages_df))
    messages_text = []
    for idx in range(sample_size):
        row = messages_df.iloc[idx]
        messages_text.append(f"Message {idx+1}: {row['text']}")
    
    prompt = f"""
You are an expert at analyzing Slack conversations and organizing them into meaningful topic clusters.

Given the following Slack messages, please group them into {num_clusters} distinct topic clusters based on their content and conversation flow.

Messages:
{chr(10).join(messages_text)}

Please provide your response in the following JSON format:
{{
    "clusters": [
        {{
            "cluster_id": "topic_001",
            "message_ids": [1, 2, 3],
            "draft_title": "Brief topic description",
            "participants": ["@user1", "@user2"]
        }}
    ]
}}

Analyze the messages and create {num_clusters} well-organized topic clusters.
"""
    return prompt

def test_cohere_model(messages_df, model_id="CohereLabs/c4ai-command-r-plus-4bit"):
    """Test the CohereLabs c4ai-command-r-plus-4bit model"""
    print(f"Loading model: {model_id}")
    
    # Load tokenizer and model with 4-bit quantization
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True  # Enable 4-bit quantization
    )
    
    print("Model loaded successfully with 4-bit quantization!")
    
    # Create clustering prompt
    prompt = create_clustering_prompt(messages_df, num_clusters=6)
    
    # Format as conversation
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template
    input_ids = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    )
    
    # Move to same device as model
    if torch.cuda.is_available():
        input_ids = input_ids.to(model.device)
    
    print("Generating clustering response...")
    
    # Generate response
    with torch.no_grad():
        gen_tokens = model.generate(
            input_ids,
            max_new_tokens=1500,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=1,
            early_stopping=True
        )
    
    # Decode response
    response = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    
    # Extract the model's response (remove the input prompt)
    model_response = response[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
    
    return model_response

def save_results(response, output_path):
    """Save the model response"""
    results = {
        "model": "CohereLabs/c4ai-command-r-plus-4bit",
        "timestamp": datetime.now().isoformat(),
        "response": response
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")

def main():
    """Main execution function"""
    print("�� Testing CohereLabs c4ai-command-r-plus-4bit for Slack Clustering")
    print("=" * 70)
    
    # Load data
    csv_path = "data/Synthetic_Slack_Messages.csv"
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    print(f"📊 Loading data from: {csv_path}")
    messages_df = load_slack_data(csv_path)
    print(f"✅ Loaded {len(messages_df)} messages")
    
    # Test model
    try:
        response = test_cohere_model(messages_df)
        
        # Save results
        output_path = "output/cohere_c4ai_command_r_plus_4bit_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_results(response, output_path)
        
        print("\n🎯 Model Response:")
        print("-" * 50)
        print(response)
        
    except Exception as e:
        print(f"❌ Error during model testing: {str(e)}")
        print("Make sure you have access to the gated model and sufficient GPU memory.")

if __name__ == "__main__":
    main()
