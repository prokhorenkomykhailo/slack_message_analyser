#!/usr/bin/env python3
"""
Phase 4: FIXED AI Decision-Making for Merge/Split
Uses _clusters.json files when available (old successful runs)
"""

import os
import json
import time
import csv
from typing import Dict, List, Any
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_clients import get_model_client

class Phase4AIDecisionMakingFixed:
    """AI models make merge/split decisions - FIXED to use _clusters.json files"""
    
    def __init__(self):
        self.phase_name = "phase4_ai_decision_making_fixed"
        self.output_dir = os.path.join("output", self.phase_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load Step 1 results (including _clusters.json files)
        self.step1_results = self.load_all_step1_results()
        self.messages = self.load_messages()
        
        print(f"✅ Loaded {len(self.step1_results)} Step 1 results (including old successful runs)")
        print(f"✅ Loaded {len(self.messages)} messages")
    
    def load_all_step1_results(self) -> List[Dict]:
        """Load Step 1 results including _clusters.json files"""
        step1_dir = "output/phase3_topic_clustering"
        results_map = {}  # Use dict to avoid duplicates
        
        for filename in os.listdir(step1_dir):
            if not filename.endswith('.json'):
                continue
            
            # Skip meta files
            if filename.startswith(('comprehensive_', 'detailed_', 'best_', 'results_', 'gpt_test')):
                continue
            
            filepath = os.path.join(step1_dir, filename)
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Handle _clusters.json files (old format)
                if filename.endswith('_clusters.json'):
                    # This is a clusters array, need to create proper result structure
                    model_name = filename.replace('_clusters.json', '')
                    
                    if isinstance(data, list) and len(data) > 0:
                        # Extract provider and model from filename
                        parts = model_name.split('_', 1)
                        provider = parts[0] if len(parts) > 0 else 'unknown'
                        model = parts[1] if len(parts) > 1 else model_name
                        
                        result = {
                            'provider': provider,
                            'model': model,
                            'success': True,
                            'clusters': data,
                            'source': 'clusters_file'
                        }
                        
                        results_map[model_name] = result
                        print(f"✅ Loaded {provider}/{model} from _clusters.json - {len(data)} clusters")
                
                # Handle regular .json files (new format)
                elif filename.endswith('.json'):
                    model_name = filename.replace('.json', '')
                    
                    # Only use if successful and has clusters
                    if data.get('success', False) and data.get('clusters'):
                        results_map[model_name] = data
                        print(f"✅ Loaded {data['provider']}/{data['model']} from .json - {len(data['clusters'])} clusters")
                    
                    # If no clusters array but _clusters.json doesn't exist, mark as available for loading
                    elif not os.path.exists(os.path.join(step1_dir, f"{model_name}_clusters.json")):
                        # This model failed and has no _clusters.json backup
                        pass
                
            except Exception as e:
                print(f"⚠️ Skipped {filename}: {e}")
        
        return list(results_map.values())
    
    def load_messages(self) -> List[Dict]:
        """Load messages from CSV"""
        csv_path = "data/Synthetic_Slack_Messages.csv"
        messages = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                messages.append({
                    "id": idx + 1,
                    "channel": row["channel"],
                    "user": row["user_name"],
                    "text": row["text"],
                    "timestamp": row["timestamp"],
                    "thread_id": row["thread_id"] if row["thread_id"] else None
                })
        
        return messages
    
    def get_cluster_messages_text(self, cluster: Dict, limit: int = 10) -> str:
        """Get message texts for a cluster"""
        message_ids = cluster.get('message_ids', [])
        texts = []
        for msg_id in message_ids[:limit]:
            if msg_id <= len(self.messages):
                msg = self.messages[msg_id - 1]
                texts.append(f"[{msg_id}] {msg['text'][:100]}...")
        return "\n".join(texts)
    
    def create_ai_prompt(self, clusters: List[Dict], provider: str, model: str) -> str:
        """Create prompt for AI to make merge/split decisions"""
        
        cluster_summaries = []
        for cluster in clusters:
            msg_count = len(cluster.get('message_ids', []))
            sample_messages = self.get_cluster_messages_text(cluster, limit=5)
            
            cluster_summaries.append(f"""
Cluster: {cluster.get('cluster_id', 'unknown')}
Title: {cluster.get('draft_title', 'No title')}
Messages: {msg_count}
Sample Messages:
{sample_messages}
""")
        
        clusters_text = "\n---\n".join(cluster_summaries)
        
        prompt = f"""You are an expert at analyzing Slack conversation clusters.

**TASK**: Analyze these {len(clusters)} clusters and decide merge/split operations.

**CLUSTERS**:
{clusters_text}

**RULES**:
- MERGE if: Same project AND semantically related
- SPLIT if: Multiple distinct topics OR >25 messages
- Target: 12-15 final clusters

**OUTPUT** (JSON only):
{{
  "merge_operations": [
    {{"clusters": ["cluster_001", "cluster_006"], "reason": "Same project EcoBloom", "confidence": 0.9}}
  ],
  "split_operations": [
    {{"cluster": "cluster_007", "reason": "Too large, multiple phases", 
      "suggested_splits": [
        {{"new_id": "cluster_007_a", "message_ids": [89, 90], "title": "Phase A"}},
        {{"new_id": "cluster_007_b", "message_ids": [91, 92], "title": "Phase B"}}
      ],
      "confidence": 0.8}}
  ],
  "reasoning": "Your overall reasoning"
}}
"""
        return prompt
    
    def parse_ai_response(self, response_text: str) -> Dict:
        """Parse AI model's JSON response"""
        try:
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end]
            else:
                return None
            
            return json.loads(json_str)
        except Exception as e:
            print(f"⚠️ Parse error: {e}")
            return None
    
    def apply_operations(self, clusters: List[Dict], operations: Dict) -> List[Dict]:
        """Apply merge/split operations"""
        refined_clusters = []
        processed_ids = set()
        
        # Process splits
        split_map = {}
        for split_op in operations.get('split_operations', []):
            cluster_id = split_op.get('cluster')
            suggested_splits = split_op.get('suggested_splits', [])
            if cluster_id and suggested_splits:
                split_map[cluster_id] = suggested_splits
        
        # Process merges
        merge_map = {}
        for merge_op in operations.get('merge_operations', []):
            cluster_ids = merge_op.get('clusters', [])
            if len(cluster_ids) >= 2:
                merge_map[cluster_ids[0]] = cluster_ids
        
        # Apply operations
        for cluster in clusters:
            cluster_id = cluster.get('cluster_id')
            
            if cluster_id in processed_ids:
                continue
            
            # Split
            if cluster_id in split_map:
                for split_cluster in split_map[cluster_id]:
                    refined_clusters.append(split_cluster)
                processed_ids.add(cluster_id)
            
            # Merge
            elif cluster_id in merge_map:
                merge_ids = merge_map[cluster_id]
                combined_message_ids = []
                combined_participants = set()
                
                for mid in merge_ids:
                    for c in clusters:
                        if c.get('cluster_id') == mid:
                            combined_message_ids.extend(c.get('message_ids', []))
                            combined_participants.update(c.get('participants', []))
                            processed_ids.add(mid)
                
                refined_clusters.append({
                    "cluster_id": f"merged_{'_'.join(merge_ids)}",
                    "message_ids": combined_message_ids,
                    "draft_title": f"Merged: {' + '.join(merge_ids)}",
                    "participants": list(combined_participants)
                })
            
            # Keep as-is
            else:
                refined_clusters.append(cluster)
                processed_ids.add(cluster_id)
        
        return refined_clusters
    
    def evaluate_model(self, provider: str, model_name: str, step1_result: Dict) -> Dict:
        """Evaluate a single model's AI decision-making"""
        print(f"\n{'='*60}")
        print(f"🤖 Testing {provider}/{model_name}")
        print(f"{'='*60}")
        
        try:
            clusters = step1_result.get('clusters', [])
            print(f"📊 Step 1 clusters: {len(clusters)}")
            
            if not clusters:
                print(f"⚠️ No clusters found, skipping")
                return {
                    "provider": provider,
                    "model": model_name,
                    "success": False,
                    "error": "No clusters in Step 1"
                }
            
            # Create AI prompt
            prompt = self.create_ai_prompt(clusters, provider, model_name)
            
            # Call AI model
            print(f"🤖 Calling {model_name}...")
            client = get_model_client(provider, model_name)
            
            if not client:
                return {
                    "provider": provider,
                    "model": model_name,
                    "success": False,
                    "error": "No API client available"
                }
            
            start_time = time.time()
            result = client.call_model(prompt, max_tokens=4000, temperature=0.3)
            duration = time.time() - start_time
            
            if not result.get('success'):
                return {
                    "provider": provider,
                    "model": model_name,
                    "success": False,
                    "error": result.get('error', 'Unknown error'),
                    "duration": duration
                }
            
            # Parse AI's decisions
            ai_response = result.get('response', '')
            operations = self.parse_ai_response(ai_response)
            
            if not operations:
                return {
                    "provider": provider,
                    "model": model_name,
                    "success": False,
                    "error": "Failed to parse AI decisions",
                    "duration": duration,
                    "raw_response": ai_response
                }
            
            # Apply operations
            refined_clusters = self.apply_operations(clusters, operations)
            
            # Calculate metrics
            num_merges = len(operations.get('merge_operations', []))
            num_splits = len(operations.get('split_operations', []))
            
            print(f"✅ Success!")
            print(f"   Merges: {num_merges}")
            print(f"   Splits: {num_splits}")
            print(f"   Step 1: {len(clusters)} → Step 2: {len(refined_clusters)} clusters")
            
            return {
                "provider": provider,
                "model": model_name,
                "phase": "step2_ai_decision_making_fixed",
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "duration": duration,
                "step1_clusters": clusters,
                "ai_operations": operations,
                "refined_clusters": refined_clusters,
                "metrics": {
                    "step1_clusters": len(clusters),
                    "refined_clusters": len(refined_clusters),
                    "merge_operations": num_merges,
                    "split_operations": num_splits,
                    "cluster_change": len(refined_clusters) - len(clusters)
                },
                "usage": result.get('usage', {}),
                "raw_response": ai_response,
                "error": ""
            }
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return {
                "provider": provider,
                "model": model_name,
                "success": False,
                "error": str(e),
                "duration": 0
            }
    
    def run_evaluation(self):
        """Run AI decision-making evaluation on all models"""
        print("🚀 Phase 4: FIXED AI Decision-Making (Uses _clusters.json files)")
        print("=" * 70)
        
        if not self.step1_results:
            print("❌ No Step 1 results found")
            return
        
        results = {}
        successful = 0
        
        for step1_result in self.step1_results:
            provider = step1_result['provider']
            model_name = step1_result['model']
            
            result = self.evaluate_model(provider, model_name, step1_result)
            results[f"{provider}_{model_name}"] = result
            
            # Save individual result
            output_file = os.path.join(self.output_dir, f"{provider}_{model_name}_ai_decisions.json")
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            
            if result['success']:
                successful += 1
                print(f"💾 Saved: {output_file}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown')}")
        
        # Save comprehensive
        comprehensive_file = os.path.join(self.output_dir, "comprehensive_results.json")
        with open(comprehensive_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"📊 COMPLETE")
        print(f"✅ Successful: {successful}/{len(self.step1_results)}")
        print(f"💾 Results: {self.output_dir}")

def main():
    evaluator = Phase4AIDecisionMakingFixed()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
