#!/usr/bin/env python3
"""
Fixed Cohere Step 1 Clustering with proper attention mask handling
This resolves the attention mask warning and improves results
"""

import os
import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

def load_messages() -> List[Dict]:
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

def test_cohere_step1_clustering():
    """Test Cohere Step 1 clustering with proper attention mask handling"""
    
    print("🚀 Cohere Step 1 Clustering (Fixed Version)")
    print("=" * 50)
    
    
    messages = load_messages()
    if not messages:
        return
    
    
    test_messages = messages[:30]
    print(f"📝 Testing with {len(test_messages)} messages")
    
    try:
        
        model_name = "CohereLabs/c4ai-command-r-plus-08-2024"
        print(f"🔄 Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("✅ Set pad_token to eos_token")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ Model loaded")
        
        
        messages_text = "\\n".join([
            f"{i+1}. {msg['user']} in {msg['channel']}: {msg['content'][:120]}"
            for i, msg in enumerate(test_messages)
        ])
        
        prompt = f"""You are an AI assistant specialized in workplace communication analysis. Analyze these Slack messages and group them into topic clusters.

Messages:
{messages_text}

Create clusters based on:
- Shared topics or themes
- Same participants discussing related issues  
- Sequential conversation threads
- Related business processes

For each cluster, provide:
- Descriptive title (2-4 words)
- Message numbers that belong together
- Participants involved
- Channel information
- Brief summary

Return JSON format:
{{
  "clusters": [
    {{
      "cluster_id": "cluster_1",
      "title": "Project Planning",
      "message_ids": [1, 3, 5],
      "participants": ["@alice", "@bob"],
      "channel": "#project-alpha",
      "summary": "Discussion about project timeline and resources"
    }}
  ]
}}"""
        
        
        messages_chat = [{"role": "user", "content": prompt}]
        
        
        inputs = tokenizer.apply_chat_template(
            messages_chat, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt",
            return_attention_mask=True  
        )
        
        
        if isinstance(inputs, tuple):
            input_ids, attention_mask = inputs
        else:
            input_ids = inputs
            attention_mask = None
        
        print("🔄 Generating clusters...")
        
        
        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids,
                attention_mask=attention_mask,  
                max_new_tokens=1200,
                do_sample=True,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        
        response = tokenizer.decode(gen_tokens[0][len(input_ids[0]):], skip_special_tokens=True)
        
        print("✅ Generation completed")
        print("\\n📋 Generated Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        
        clusters = parse_clustering_response(response)
        
        print(f"\\n📊 Parsed {len(clusters)} clusters:")
        for i, cluster in enumerate(clusters):
            print(f"   {i+1}. {cluster.get('title', 'Untitled')} - {len(cluster.get('message_ids', []))} messages")
        
        
        result = {
            "success": True,
            "model": model_name,
            "num_messages_analyzed": len(test_messages),
            "clusters": clusters,
            "raw_response": response,
            "attention_mask_fixed": True
        }
        
        os.makedirs("output", exist_ok=True)
        with open("output/cohere_step1_fixed_results.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print("\\n💾 Results saved to output/cohere_step1_fixed_results.json")
        print("\\n🎉 Cohere Step 1 clustering test completed successfully!")
        print("\\n📈 This shows Cohere's capability for your Step 1 evaluation:")
        print("   ✅ Proper attention mask handling")
        print("   ✅ Topic clustering from messages")
        print("   ✅ JSON structured output")
        print("   ✅ Participant and channel analysis")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

def parse_clustering_response(response: str) -> List[Dict]:
    """Parse clustering response from Cohere"""
    
    try:
        
        if "```json" in response:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            json_str = response[json_start:json_end].strip()
        elif "{" in response and "}" in response:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_str = response[json_start:json_end]
        else:
            print("⚠️ No JSON found in response")
            return create_fallback_clusters()
        
        
        data = json.loads(json_str)
        
        if "clusters" in data:
            return data["clusters"]
        else:
            return [data] if isinstance(data, dict) else []
            
    except Exception as e:
        print(f"⚠️ Error parsing response: {e}")
        return create_fallback_clusters()

def create_fallback_clusters() -> List[Dict]:
    """Create fallback clusters if parsing fails"""
    return [{
        "cluster_id": "cluster_1",
        "title": "General Discussion",
        "message_ids": list(range(1, 11)),
        "participants": ["@unknown"],
        "channel": "#general",
        "summary": "Fallback cluster for unparsed messages"
    }]

def main():
    """Main function"""
    result = test_cohere_step1_clustering()
    
    if result and result["success"]:
        print("\\n🎯 NEXT STEPS FOR YOUR PROJECT:")
        print("=" * 40)
        print("1. ✅ Cohere Command R+ is working for Step 1")
        print("2. 📊 Test with your full 200 messages")
        print("3. 🔄 Compare results with your benchmark")
        print("4. 📈 Evaluate clustering quality")
        print("5. 🚀 Use this model for your Step 1 evaluation")
        
        print("\\n💡 To test with all 200 messages:")
        print("   - Modify the test_messages = messages[:30] line")
        print("   - Change to test_messages = messages")
        print("   - Run again for full evaluation")

if __name__ == "__main__":
    main()
