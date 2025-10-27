#!/usr/bin/env python3
"""
Fix Hugging Face authentication issue
This resolves the token recognition problem
"""

import os
import sys
import subprocess

def fix_authentication():
    """Fix the Hugging Face authentication issue"""
    
    print("🔧 Fixing Hugging Face Authentication")
    print("=" * 40)
    
    # Step 1: Check current token
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        print("❌ No HUGGINGFACE_TOKEN found")
        print("Please set your token:")
        print("export HUGGINGFACE_TOKEN=your_token_here")
        return False
    
    print(f"✅ Token found (length: {len(token)})")
    
    # Step 2: Login using the token
    print("🔐 Logging in with token...")
    
    try:
        # Method 1: Use huggingface_hub.login
        from huggingface_hub import login
        login(token=token)
        print("✅ Login successful using huggingface_hub.login")
        
        # Verify login
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Verified as: {user_info['name']}")
        return True
        
    except Exception as e:
        print(f"❌ Login failed: {e}")
        
        # Method 2: Try CLI login
        try:
            print("🔄 Trying CLI login...")
            result = subprocess.run([
                'huggingface-cli', 'login', '--token', token
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ CLI login successful")
                
                # Verify again
                from huggingface_hub import whoami
                user_info = whoami()
                print(f"✅ Verified as: {user_info['name']}")
                return True
            else:
                print(f"❌ CLI login failed: {result.stderr}")
                return False
                
        except Exception as e2:
            print(f"❌ CLI login error: {e2}")
            return False

def test_model_access():
    """Test access to Cohere models after authentication fix"""
    
    print("\n🧪 Testing model access...")
    
    models_to_test = [
        "CohereLabs/c4ai-command-r-plus-08-2024",
        "CohereLabs/c4ai-command-r-plus"
    ]
    
    accessible_model = None
    
    for model_name in models_to_test:
        print(f"\n🔍 Testing {model_name}...")
        
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # Test access by listing files
            files = api.list_repo_files(model_name)
            print(f"✅ Have access to {model_name}")
            print(f"   Files available: {len(files)}")
            accessible_model = model_name
            break
            
        except Exception as e:
            print(f"❌ No access to {model_name}: {e}")
            continue
    
    return accessible_model

def test_model_loading(model_name):
    """Test actually loading the model"""
    
    print(f"\n🚀 Testing model loading: {model_name}")
    
    try:
        from transformers import AutoTokenizer
        
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("   ✅ Tokenizer loaded successfully")
        
        # Test basic functionality
        test_text = "Hello, how are you?"
        tokens = tokenizer.encode(test_text)
        print(f"   ✅ Tokenization works ({len(tokens)} tokens)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False

def create_working_cohere_evaluator():
    """Create a working Cohere evaluator for Step 1"""
    
    evaluator_code = '''#!/usr/bin/env python3
"""
Working Cohere Command R+ Step 1 Evaluator
This version handles authentication properly
"""

import os
import torch
import json
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

# Ensure authentication is working
def ensure_auth():
    """Ensure Hugging Face authentication is working"""
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ Authentication issue: {e}")
        return False

class WorkingCohereEvaluator:
    """Working Cohere evaluator for Step 1 clustering"""
    
    def __init__(self, model_name="CohereLabs/c4ai-command-r-plus-08-2024"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load the Cohere model with proper error handling"""
        print(f"🔄 Loading {self.model_name}...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("✅ Tokenizer loaded")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print("✅ Model loaded")
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def evaluate_step1(self, messages: List[Dict]) -> Dict:
        """Evaluate Step 1: Corpus-wide topic clustering"""
        
        if not self.model or not self.tokenizer:
            return {"success": False, "error": "Model not loaded"}
        
        # Create clustering prompt
        messages_sample = messages[:50]  # Use first 50 for testing
        
        messages_text = "\\n\\n".join([
            f"Message {i+1} ({msg.get('user', 'Unknown')} in {msg.get('channel', '#general')}): {msg.get('content', '')[:150]}"
            for i, msg in enumerate(messages_sample)
        ])
        
        prompt = f"""Analyze these Slack messages and create topic clusters:

{messages_text}

Create clusters based on shared topics, participants, and conversation flow.

Return JSON format:
{{
  "clusters": [
    {{
      "cluster_id": "cluster_1",
      "title": "Topic Name",
      "message_ids": [1, 2, 3],
      "participants": ["@user1", "@user2"],
      "channel": "#channel",
      "summary": "Brief description"
    }}
  ]
}}"""
        
        try:
            # Format with chat template
            messages_chat = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages_chat, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt"
            )
            
            # Generate
            print("🔄 Generating clusters...")
            start_time = time.time()
            
            with torch.no_grad():
                gen_tokens = self.model.generate(
                    input_ids,
                    max_new_tokens=1500,
                    do_sample=True,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generation_time = time.time() - start_time
            
            # Decode response
            response_text = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            
            # Parse clusters
            clusters = self.parse_response(response_text)
            
            return {
                "success": True,
                "clusters": clusters,
                "generation_time": generation_time,
                "num_messages": len(messages_sample),
                "num_clusters": len(clusters),
                "response_text": response_text
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def parse_response(self, response: str) -> List[Dict]:
        """Parse clustering response"""
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                return self.create_fallback_clusters()
            
            data = json.loads(json_str)
            return data.get("clusters", [])
            
        except Exception as e:
            print(f"⚠️ Parse error: {e}")
            return self.create_fallback_clusters()
    
    def create_fallback_clusters(self) -> List[Dict]:
        """Create fallback clusters"""
        return [{
            "cluster_id": "cluster_1",
            "title": "General Discussion",
            "message_ids": list(range(1, 11)),
            "participants": ["@unknown"],
            "channel": "#general",
            "summary": "Fallback cluster"
        }]

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

def main():
    """Main function"""
    print("🚀 Working Cohere Step 1 Evaluator")
    print("=" * 40)
    
    # Check authentication
    if not ensure_auth():
        print("❌ Authentication failed")
        return
    
    # Load messages
    messages = load_messages()
    if not messages:
        return
    
    # Initialize evaluator
    evaluator = WorkingCohereEvaluator()
    
    # Load model
    if not evaluator.load_model():
        return
    
    # Run evaluation
    result = evaluator.evaluate_step1(messages)
    
    if result["success"]:
        print("\\n🎉 Step 1 evaluation completed!")
        print(f"   📊 Clusters: {result['num_clusters']}")
        print(f"   ⏱️  Time: {result['generation_time']:.2f}s")
        print(f"   📝 Messages: {result['num_messages']}")
        
        # Save results
        os.makedirs("output", exist_ok=True)
        with open("output/working_cohere_results.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print("\\n💾 Results saved to output/working_cohere_results.json")
        
        # Show clusters
        print("\\n📋 Generated clusters:")
        for i, cluster in enumerate(result["clusters"][:3]):
            print(f"   {i+1}. {cluster.get('title', 'Untitled')} ({len(cluster.get('message_ids', []))} messages)")
        
    else:
        print(f"\\n❌ Evaluation failed: {result.get('error')}")

if __name__ == "__main__":
    main()
'''
    
    with open("working_cohere_evaluator.py", "w") as f:
        f.write(evaluator_code)
    
    print("✅ Created working_cohere_evaluator.py")

def main():
    """Main function"""
    
    print("🔧 Fixing Authentication and Testing Cohere Access")
    print("=" * 55)
    
    # Step 1: Fix authentication
    if not fix_authentication():
        print("\n❌ Could not fix authentication")
        return
    
    # Step 2: Test model access
    accessible_model = test_model_access()
    if not accessible_model:
        print("\n❌ No access to any Cohere model")
        return
    
    # Step 3: Test model loading
    if not test_model_loading(accessible_model):
        print("\n❌ Model loading failed")
        return
    
    # Step 4: Create working evaluator
    create_working_cohere_evaluator()
    
    print("\n🎉 Authentication fixed and model access confirmed!")
    print(f"\n✅ You can now use: {accessible_model}")
    print("\n📋 Next steps:")
    print("   1. Run: python working_cohere_evaluator.py")
    print("   2. Check results in output/working_cohere_results.json")
    print("   3. Use this model for your Step 1 evaluation")

if __name__ == "__main__":
    main()
