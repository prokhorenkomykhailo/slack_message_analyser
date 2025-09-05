#!/usr/bin/env python3
"""
Cohere Command R+ clustering implementation for Phase 3 evaluation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Any
import json
import time
import os

class CohereCommandRPlusClustering:
    """Cohere Command R+ clustering implementation"""
    
    def __init__(self, model_path: str, quantization: str = None):
        self.model_path = model_path
        self.quantization = quantization
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load the Cohere Command R+ model"""
        print(f"🔄 Loading Cohere Command R+ model: {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Configure quantization if specified
            if self.quantization == "8bit":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    quantization_config=bnb_config,
                    device_map="auto"
                )
            elif self.quantization == "4bit":
                bnb_config = BitsAndBytesConfig(load_in_4bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    quantization_config=bnb_config,
                    device_map="auto"
                )
            else:
                # Load full precision model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            print(f"✅ Cohere Command R+ model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading Cohere Command R+ model: {e}")
            return False
    
    def create_clustering_prompt(self, messages: List[Dict]) -> str:
        """Create clustering prompt using Command R+ format"""
        
        # Convert messages to Command R+ chat format
        conversation = []
        for msg in messages:
            if msg.get('role') == 'user':
                conversation.append({"role": "user", "content": msg['content']})
            else:
                conversation.append({"role": "assistant", "content": msg['content']})
        
        # Add clustering instruction
        clustering_instruction = {
            "role": "user",
            "content": """Please analyze the conversation above and create topic clusters. For each cluster:
1. Identify the main topic/theme
2. List the message IDs that belong to that cluster
3. Provide a descriptive title
4. List the participants involved

Format your response as a JSON structure with clusters array."""
        }
        conversation.append(clustering_instruction)
        
        # Apply Command R+ chat template
        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def cluster_messages(self, messages: List[Dict]) -> List[Dict]:
        """Perform clustering using Cohere Command R+"""
        
        if not self.model or not self.tokenizer:
            print("❌ Model not loaded")
            return []
        
        try:
            # Create clustering prompt
            prompt = self.create_clustering_prompt(messages)
            
            # Tokenize input
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            if self.device == "cuda":
                input_ids = input_ids.cuda()
            
            # Generate clustering response
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
        """Parse the clustering response from Command R+"""
        
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
