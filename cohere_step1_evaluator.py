#!/usr/bin/env python3
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
        messages_text = "\n\n".join([
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
