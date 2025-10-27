#!/usr/bin/env python3
"""
Fix for model loading error in CohereCommandRPlusClustering
This script patches the cohere_clustering.py to handle different model types correctly
"""

def create_fixed_cohere_clustering():
    """Create a fixed version of the CohereCommandRPlusClustering class"""
    
    fixed_code = '''#!/usr/bin/env python3
"""
Fixed Cohere Command R+ clustering implementation for Phase 3 evaluation
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    AutoConfig
)
from typing import List, Dict, Any
import json
import time
import os

class CohereCommandRPlusClustering:
    """Fixed Cohere Command R+ clustering implementation"""
    
    def __init__(self, model_path: str, quantization: str = None):
        self.model_path = model_path
        self.quantization = quantization
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = None
        
    def detect_model_type(self):
        """Detect if model is causal LM or seq2seq"""
        try:
            config = AutoConfig.from_pretrained(self.model_path)
            
            # Check model architecture
            if hasattr(config, 'is_encoder_decoder') and config.is_encoder_decoder:
                return "seq2seq"
            elif "t5" in config.model_type.lower():
                return "seq2seq"
            elif "gpt" in config.model_type.lower() or "llama" in config.model_type.lower():
                return "causal"
            elif "command" in self.model_path.lower() or "cohere" in self.model_path.lower():
                return "causal"
            else:
                # Default to causal for most modern LLMs
                return "causal"
                
        except Exception as e:
            print(f"⚠️ Could not detect model type: {e}")
            # Try to infer from model name
            if "t5" in self.model_path.lower():
                return "seq2seq"
            else:
                return "causal"
        
    def load_model(self):
        """Load the model with correct architecture"""
        print(f"🔄 Loading model: {self.model_path}")
        
        try:
            # Detect model type first
            self.model_type = self.detect_model_type()
            print(f"🔍 Detected model type: {self.model_type}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization if specified
            model_kwargs = {}
            if self.quantization == "8bit":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs.update({
                    "quantization_config": bnb_config,
                    "device_map": "auto"
                })
            elif self.quantization == "4bit":
                bnb_config = BitsAndBytesConfig(load_in_4bit=True)
                model_kwargs.update({
                    "quantization_config": bnb_config,
                    "device_map": "auto"
                })
            else:
                # Load full precision model
                model_kwargs.update({
                    "torch_dtype": torch.float16,
                    "device_map": "auto"
                })
            
            # Load model with correct architecture
            if self.model_type == "seq2seq":
                print("📚 Loading as Seq2Seq model...")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_path, 
                    **model_kwargs
                )
            else:
                print("🗣️ Loading as Causal LM model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
            
            print(f"✅ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def create_clustering_prompt(self, messages: List[Dict]) -> str:
        """Create clustering prompt based on model type"""
        
        # Create a simple prompt for clustering
        prompt_text = "Please analyze these messages and group them into topic clusters:\\n\\n"
        
        for i, msg in enumerate(messages[:50]):  # Limit to first 50 messages to avoid token limits
            content = msg.get('content', '')[:200]  # Limit message length
            user = msg.get('user', 'Unknown')
            prompt_text += f"Message {i+1} ({user}): {content}\\n"
        
        prompt_text += "\\nPlease create JSON clusters with this format:\\n"
        prompt_text += '{"clusters": [{"cluster_id": "1", "message_ids": [1,2,3], "draft_title": "Topic Name", "participants": ["user1"], "channel": "#general"}]}'
        
        return prompt_text
    
    def cluster_messages(self, messages: List[Dict]) -> List[Dict]:
        """Perform clustering using the loaded model"""
        
        if not self.model or not self.tokenizer:
            print("❌ Model not loaded")
            return {"clusters": [], "generation_time": 0, "response_text": ""}
        
        try:
            # Create clustering prompt
            prompt = self.create_clustering_prompt(messages)
            
            # Tokenize input
            if self.model_type == "seq2seq":
                # For seq2seq models, we use different generation approach
                inputs = self.tokenizer(prompt, return_tensors="pt", truncate=True, max_length=512)
            else:
                # For causal models
                inputs = self.tokenizer(prompt, return_tensors="pt", truncate=True, max_length=1024)
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate clustering response
            start_time = time.time()
            
            with torch.no_grad():
                if self.model_type == "seq2seq":
                    # Seq2seq generation
                    gen_tokens = self.model.generate(
                        **inputs,
                        max_new_tokens=500,
                        do_sample=True,
                        temperature=0.1,
                        num_beams=2,
                        early_stopping=True
                    )
                else:
                    # Causal LM generation
                    gen_tokens = self.model.generate(
                        inputs["input_ids"],
                        max_new_tokens=500,
                        do_sample=True,
                        temperature=0.1,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            
            generation_time = time.time() - start_time
            
            # Decode response
            if self.model_type == "seq2seq":
                response_text = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            else:
                # For causal models, skip the input part
                response_text = self.tokenizer.decode(gen_tokens[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            
            # Extract clustering results
            clusters = self.parse_clustering_response(response_text, messages)
            
            return {
                "clusters": clusters,
                "generation_time": generation_time,
                "response_text": response_text
            }
            
        except Exception as e:
            print(f"❌ Error during clustering: {e}")
            return {"clusters": [], "generation_time": 0, "response_text": ""}
    
    def parse_clustering_response(self, response: str, messages: List[Dict]) -> List[Dict]:
        """Parse the clustering response"""
        
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
                # Fallback: create basic clusters
                return self.create_fallback_clusters(messages)
            
            # Parse JSON
            clustering_data = json.loads(json_str)
            
            if "clusters" in clustering_data:
                return clustering_data["clusters"]
            else:
                return clustering_data
                
        except Exception as e:
            print(f"⚠️ Error parsing response, using fallback: {e}")
            return self.create_fallback_clusters(messages)
    
    def create_fallback_clusters(self, messages: List[Dict]) -> List[Dict]:
        """Create fallback clusters if parsing fails"""
        
        # Simple fallback: group by conversation flow
        clusters = []
        current_cluster = {
            "cluster_id": "cluster_001",
            "message_ids": [],
            "draft_title": "General Conversation",
            "participants": set(),
            "channel": "#general"
        }
        
        for i, msg in enumerate(messages):
            current_cluster["message_ids"].append(i + 1)
            if "user" in msg:
                current_cluster["participants"].add(msg["user"])
        
        current_cluster["participants"] = list(current_cluster["participants"])
        clusters.append(current_cluster)
        
        return clusters
'''
    
    # Write the fixed version
    with open("cohere_clustering_fixed.py", "w") as f:
        f.write(fixed_code)
    
    print("✅ Created fixed version: cohere_clustering_fixed.py")

if __name__ == "__main__":
    create_fixed_cohere_clustering()