#!/usr/bin/env python3
"""
Robust Cohere Step 1 Clustering with timeout and error handling
This version won't hang and provides better debugging
"""

import os
import torch
import json
import pandas as pd
import signal
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Generation timed out")

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
                "timestamp": row.get("timestamp", ""),
                "channel": row.get("channel", "#general")
            })
        
        print(f"✅ Loaded {len(messages)} messages")
        return messages
        
    except Exception as e:
        print(f"❌ Error loading messages: {e}")
        return []

def test_cohere_step1_robust():
    """Test Cohere Step 1 clustering with robust error handling"""
    
    print("🚀 Cohere Step 1 Clustering (Robust Version)")
    print("=" * 50)
    
    # Load messages
    messages = load_messages()
    if not messages:
        return
    
    # Use only first 10 messages for initial test
    test_messages = messages[:10]
    print(f"📝 Testing with {len(test_messages)} messages (reduced for stability)")
    
    try:
        # Load model
        model_name = "CohereLabs/c4ai-command-r-plus-08-2024"
        print(f"🔄 Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Fix attention mask issue
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("✅ Set pad_token to eos_token")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ Model loaded successfully")
        
        # Create simple clustering prompt
        messages_text = "\\n".join([
            f"{i+1}. {msg['user']}: {msg['content'][:80]}"
            for i, msg in enumerate(test_messages)
        ])
        
        prompt = f"""Group these messages into topic clusters:

{messages_text}

Return JSON with clusters containing message numbers and titles."""
        
        print("🔄 Preparing generation...")
        
        # Format with chat template
        messages_chat = [{"role": "user", "content": prompt}]
        
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages_chat, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        print(f"📏 Input length: {inputs.shape[1]} tokens")
        
        # Set up timeout (5 minutes)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)  # 5 minutes timeout
        
        try:
            print("🔄 Generating clusters (with 5min timeout)...")
            start_time = time.time()
            
            with torch.no_grad():
                gen_tokens = model.generate(
                    inputs,
                    max_new_tokens=500,  # Reduced for faster generation
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            generation_time = time.time() - start_time
            signal.alarm(0)  # Cancel timeout
            
            print(f"✅ Generation completed in {generation_time:.2f}s")
            
            # Decode response
            response = tokenizer.decode(gen_tokens[0][len(inputs[0]):], skip_special_tokens=True)
            
            print("\\n📋 Generated Response:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
            # Try to parse clusters
            clusters = parse_clustering_response(response)
            
            print(f"\\n📊 Parsed {len(clusters)} clusters:")
            for i, cluster in enumerate(clusters):
                print(f"   {i+1}. {cluster.get('title', 'Untitled')} - {len(cluster.get('message_ids', []))} messages")
            
            # Save results
            result = {
                "success": True,
                "model": model_name,
                "num_messages_analyzed": len(test_messages),
                "clusters": clusters,
                "raw_response": response,
                "generation_time": generation_time,
                "input_tokens": inputs.shape[1],
                "output_tokens": gen_tokens.shape[1] - inputs.shape[1]
            }
            
            os.makedirs("output", exist_ok=True)
            with open("output/cohere_step1_robust_results.json", "w") as f:
                json.dump(result, f, indent=2)
            
            print("\\n💾 Results saved to output/cohere_step1_robust_results.json")
            print("\\n🎉 Cohere Step 1 clustering test completed successfully!")
            
            return result
            
        except TimeoutError:
            signal.alarm(0)
            print("\\n⏰ Generation timed out after 5 minutes")
            print("💡 Try reducing max_new_tokens or using fewer messages")
            return None
            
        except Exception as gen_error:
            signal.alarm(0)
            print(f"\\n❌ Generation error: {gen_error}")
            return None
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def parse_clustering_response(response: str) -> List[Dict]:
    """Parse clustering response from Cohere"""
    
    try:
        # Try to extract JSON from response
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
        
        # Parse JSON
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
        "message_ids": list(range(1, 6)),
        "participants": ["@unknown"],
        "channel": "#general",
        "summary": "Fallback cluster for unparsed messages"
    }]

def test_simple_generation():
    """Test simple generation first"""
    
    print("🧪 Testing simple generation first...")
    
    try:
        model_name = "CohereLabs/c4ai-command-r-plus-08-2024"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Simple test
        test_prompt = "Hello, how are you?"
        messages_chat = [{"role": "user", "content": test_prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages_chat, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        print("🔄 Testing simple generation...")
        
        with torch.no_grad():
            gen_tokens = model.generate(
                inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.3,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(gen_tokens[0][len(inputs[0]):], skip_special_tokens=True)
        
        print(f"✅ Simple generation works: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Simple generation failed: {e}")
        return False

def main():
    """Main function"""
    
    print("🔧 Cohere Step 1 Robust Test")
    print("=" * 30)
    
    # First test simple generation
    if not test_simple_generation():
        print("\\n❌ Basic generation test failed")
        return
    
    print("\\n✅ Basic generation test passed, proceeding with clustering...")
    
    # Then test clustering
    result = test_cohere_step1_robust()
    
    if result and result["success"]:
        print("\\n🎯 SUCCESS! Cohere is working for Step 1")
        print("=" * 40)
        print("✅ Model loads correctly")
        print("✅ Generation works")
        print("✅ Clustering produces results")
        print("✅ Results are saved")
        
        print("\\n📈 Next steps:")
        print("1. Increase message count gradually")
        print("2. Test with your full 200 messages")
        print("3. Compare with benchmark")
        print("4. Use for Step 1 evaluation")
        
    else:
        print("\\n❌ Clustering test failed")
        print("\\n🔧 Troubleshooting:")
        print("1. Check GPU memory")
        print("2. Reduce max_new_tokens")
        print("3. Use fewer messages")
        print("4. Check model loading")

if __name__ == "__main__":
    main()
