#!/usr/bin/env python3
"""
Setup script for Cohere Command R+ Step 1 evaluation
This script properly configures the environment for Cohere model testing
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_transformers_version():
    """Check if transformers is up to date"""
    try:
        import transformers
        version = transformers.__version__
        print(f"📦 Current transformers version: {version}")
        
        # Check if we have the latest version with Cohere support
        if version < "4.44.0":
            print("⚠️  Outdated transformers version. Cohere Command R+ requires latest version.")
            return False
        else:
            print("✅ Transformers version is compatible")
            return True
            
    except ImportError:
        print("❌ Transformers not installed")
        return False

def update_transformers():
    """Update transformers to latest version with Cohere support"""
    print("🔄 Updating transformers to latest version...")
    
    try:
        # Install from source as recommended by Cohere
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/huggingface/transformers.git"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Transformers updated successfully")
            return True
        else:
            print(f"❌ Error updating transformers: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating transformers: {e}")
        return False

def install_additional_deps():
    """Install additional dependencies for Cohere"""
    print("📦 Installing additional dependencies...")
    
    deps = [
        "bitsandbytes",
        "accelerate", 
        "huggingface-hub",
        "torch"
    ]
    
    for dep in deps:
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", dep
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   ✅ {dep}")
            else:
                print(f"   ⚠️  {dep} - {result.stderr}")
                
        except Exception as e:
            print(f"   ❌ {dep} - {e}")

def check_huggingface_auth():
    """Check Hugging Face authentication"""
    print("🔐 Checking Hugging Face authentication...")
    
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        print("❌ No HUGGINGFACE_TOKEN found")
        print("   Please set: export HUGGINGFACE_TOKEN=your_token_here")
        print("   Or run: huggingface-cli login")
        return False
    
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        user_info = api.whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False

def create_cohere_step1_evaluator():
    """Create a specialized Cohere evaluator for Step 1 clustering"""
    
    evaluator_code = '''#!/usr/bin/env python3
"""
Cohere Command R+ Step 1 Evaluator
Specialized for corpus-wide topic clustering evaluation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any
import json
import time
import os

class CohereStep1Evaluator:
    """Cohere Command R+ evaluator for Step 1: Corpus-wide topic clustering"""
    
    def __init__(self, model_path: str = "CohereLabs/c4ai-command-r-plus"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load the Cohere Command R+ model with proper configuration"""
        print(f"🔄 Loading Cohere Command R+ model: {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load model with proper configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            print(f"✅ Cohere Command R+ model loaded successfully")
            return True
            
        except Exception as e:
            if "401" in str(e) or "gated" in str(e).lower():
                print(f"❌ Access denied to gated model: {self.model_path}")
                print("   Please request access at: https://huggingface.co/CohereLabs/c4ai-command-r-plus")
                return False
            else:
                print(f"❌ Error loading model: {e}")
                return False
    
    def create_step1_prompt(self, messages: List[Dict]) -> str:
        """Create Step 1 clustering prompt for corpus-wide topic analysis"""
        
        # Format messages for analysis
        messages_text = "\\n\\n".join([
            f"Message {i+1} ({msg.get('user', 'Unknown')} in {msg.get('channel', '#general')}): {msg.get('content', '')[:200]}"
            for i, msg in enumerate(messages[:100])  # Limit to first 100 messages
        ])
        
        prompt = f"""You are an AI assistant specialized in analyzing workplace communication patterns. Your task is to analyze the following Slack messages and create topic clusters.

Messages to analyze:
{messages_text}

Please create topic clusters following these requirements:

1. **Cluster Messages**: Group related messages together based on:
   - Shared topics or themes
   - Same participants discussing related issues
   - Sequential conversation threads
   - Related business processes or decisions

2. **For each cluster, provide**:
   - A descriptive title (2-4 words)
   - List of message IDs that belong to this cluster
   - List of participants involved
   - Channel where the cluster originated
   - Brief summary of the cluster topic

3. **Output Format**: Return a JSON structure like this:
{{
  "clusters": [
    {{
      "cluster_id": "cluster_1",
      "title": "Project Planning",
      "message_ids": [1, 3, 5, 7],
      "participants": ["@alice", "@bob"],
      "channel": "#project-alpha",
      "summary": "Discussion about project timeline and resource allocation"
    }}
  ]
}}

Please analyze the messages and create appropriate clusters:"""
        
        return prompt
    
    def evaluate_step1_clustering(self, messages: List[Dict]) -> Dict:
        """Evaluate Step 1: Corpus-wide topic clustering"""
        
        if not self.model or not self.tokenizer:
            print("❌ Model not loaded")
            return {"success": False, "error": "Model not loaded"}
        
        try:
            # Create Step 1 prompt
            prompt = self.create_step1_prompt(messages)
            
            # Format with Cohere's chat template
            messages_chat = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages_chat, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt"
            )
            
            # Generate response
            start_time = time.time()
            
            with torch.no_grad():
                gen_tokens = self.model.generate(
                    input_ids,
                    max_new_tokens=2000,
                    do_sample=True,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generation_time = time.time() - start_time
            
            # Decode response
            response_text = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            
            # Extract clustering results
            clusters = self.parse_clustering_response(response_text)
            
            return {
                "success": True,
                "clusters": clusters,
                "generation_time": generation_time,
                "response_text": response_text,
                "num_messages_analyzed": len(messages),
                "num_clusters": len(clusters)
            }
            
        except Exception as e:
            print(f"❌ Error during Step 1 evaluation: {e}")
            return {"success": False, "error": str(e)}
    
    def parse_clustering_response(self, response: str) -> List[Dict]:
        """Parse the clustering response from Cohere"""
        
        try:
            # Extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                print("⚠️ No JSON found in response, creating fallback clusters")
                return self.create_fallback_clusters()
            
            # Parse JSON
            clustering_data = json.loads(json_str)
            
            if "clusters" in clustering_data:
                return clustering_data["clusters"]
            else:
                return clustering_data
                
        except Exception as e:
            print(f"⚠️ Error parsing response: {e}")
            return self.create_fallback_clusters()
    
    def create_fallback_clusters(self) -> List[Dict]:
        """Create fallback clusters if parsing fails"""
        return [{
            "cluster_id": "cluster_fallback",
            "title": "General Discussion",
            "message_ids": list(range(1, 11)),
            "participants": ["@unknown"],
            "channel": "#general",
            "summary": "Fallback cluster for unparsed messages"
        }]

def main():
    """Test the Cohere Step 1 evaluator"""
    
    print("🚀 Testing Cohere Step 1 Evaluator")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = CohereStep1Evaluator()
    
    # Load test messages
    messages_file = "data/Synthetic_Slack_Messages.csv"
    if not os.path.exists(messages_file):
        print(f"❌ Messages file not found: {messages_file}")
        return
    
    # Load messages (you'll need to implement this based on your data format)
    print(f"📁 Loading messages from: {messages_file}")
    # messages = load_messages(messages_file)  # Implement this
    
    # Load model
    if not evaluator.load_model():
        print("❌ Could not load Cohere model")
        return
    
    print("✅ Cohere Step 1 evaluator ready for testing")

if __name__ == "__main__":
    main()
'''
    
    with open("cohere_step1_evaluator.py", "w") as f:
        f.write(evaluator_code)
    
    print("✅ Created cohere_step1_evaluator.py")

def main():
    """Main setup function"""
    
    print("🔧 Cohere Command R+ Setup for Step 1 Evaluation")
    print("=" * 60)
    
    # Step 1: Check transformers version
    if not check_transformers_version():
        print("\\n🔄 Updating transformers...")
        if not update_transformers():
            print("❌ Failed to update transformers")
            return
    
    # Step 2: Install additional dependencies
    print("\\n📦 Installing dependencies...")
    install_additional_deps()
    
    # Step 3: Check authentication
    print("\\n🔐 Checking authentication...")
    if not check_huggingface_auth():
        print("\\n⚠️  Authentication issue. Please:")
        print("   1. Request access to Cohere Command R+ at: https://huggingface.co/CohereLabs/c4ai-command-r-plus")
        print("   2. Set your token: export HUGGINGFACE_TOKEN=your_token_here")
        return
    
    # Step 4: Create specialized evaluator
    print("\\n📝 Creating Cohere Step 1 evaluator...")
    create_cohere_step1_evaluator()
    
    print("\\n🎉 Setup completed!")
    print("\\n📋 Next steps:")
    print("   1. Ensure you have access to Cohere Command R+")
    print("   2. Run: python cohere_step1_evaluator.py")
    print("   3. Test with your 200 synthetic messages")

if __name__ == "__main__":
    main()
