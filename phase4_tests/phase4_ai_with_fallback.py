#!/usr/bin/env python3
"""
Phase 4: AI Decision-Making with Model Fallback
Uses Step 1 clusters but falls back to working models for Step 2 decisions
"""

import os
import json
import time
import csv
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_clients import get_model_client

class Phase4AIWithFallback:
    """AI decision-making with fallback to working models"""
    
    # Fallback models that are known to work
    FALLBACK_MODELS = {
        'google': 'gemini-2.0-flash',  # Fast and works
        'openai': 'gpt-4o',
        'xai': 'grok-3'
    }
    
    def __init__(self):
        self.phase_name = "phase4_ai_with_fallback"
        self.output_dir = os.path.join("output", self.phase_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.step1_results = self.load_all_step1_results()
        self.messages = self.load_messages()
        
        print(f"✅ Loaded {len(self.step1_results)} Step 1 results")
        print(f"✅ Loaded {len(self.messages)} messages")
    
    def load_all_step1_results(self) -> List[Dict]:
        """Load Step 1 results including _clusters.json files"""
        step1_dir = "output/phase3_topic_clustering"
        results_map = {}
        
        for filename in os.listdir(step1_dir):
            if not filename.endswith('.json'):
                continue
            
            if filename.startswith(('comprehensive_', 'detailed_', 'best_', 'results_', 'gpt_test')):
                continue
            
            filepath = os.path.join(step1_dir, filename)
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Handle _clusters.json files (old successful runs)
                if filename.endswith('_clusters.json'):
                    model_name = filename.replace('_clusters.json', '')
                    
                    if isinstance(data, list) and len(data) > 0:
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
                        print(f"✅ {provider}/{model} - {len(data)} clusters (from _clusters.json)")
                
                # Handle regular .json files
                elif filename.endswith('.json'):
                    model_name = filename.replace('.json', '')
                    
                    if data.get('success', False) and data.get('clusters'):
                        # Only add if not already loaded from _clusters.json
                        if model_name not in results_map:
                            results_map[model_name] = data
                            print(f"✅ {data['provider']}/{data['model']} - {len(data['clusters'])} clusters (from .json)")
                
            except Exception as e:
                pass
        
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
    
    def create_ai_prompt(self, clusters: List[Dict]) -> str:
        """Create prompt for AI merge/split decisions"""
        
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
        
        prompt = f"""Analyze these {len(clusters)} Slack conversation clusters and decide merge/split operations.

**CLUSTERS**:
{clusters_text}

**DECISION RULES**:
- MERGE: Same project + semantically related
- SPLIT: Large cluster (>20 msgs) OR multiple distinct topics  
- TARGET: 12-15 final clusters

**OUTPUT (JSON)**:
{{
  "merge_operations": [{{"clusters": ["cluster_001", "cluster_006"], "reason": "Same EcoBloom project", "confidence": 0.9}}],
  "split_operations": [{{"cluster": "cluster_007", "reason": "Too large", "suggested_splits": [{{"new_id": "cluster_007_a", "message_ids": [89,90], "title": "Part A"}}], "confidence": 0.8}}],
  "reasoning": "Overall reasoning"
}}
"""
        return prompt
    
    def parse_ai_response(self, response_text: str) -> Optional[Dict]:
        """Parse AI JSON response"""
        try:
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            elif "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end]
            else:
                return None
            
            return json.loads(json_str)
        except:
            return None
    
    def apply_operations(self, clusters: List[Dict], operations: Dict) -> List[Dict]:
        """Apply merge/split operations"""
        refined_clusters = []
        processed_ids = set()
        
        # Build split map
        split_map = {}
        for split_op in operations.get('split_operations', []):
            cluster_id = split_op.get('cluster')
            splits = split_op.get('suggested_splits', [])
            if cluster_id and splits:
                split_map[cluster_id] = splits
        
        # Build merge map
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
                refined_clusters.extend(split_map[cluster_id])
                processed_ids.add(cluster_id)
            
            # Merge
            elif cluster_id in merge_map:
                merge_ids = merge_map[cluster_id]
                combined_msg_ids = []
                combined_parts = set()
                
                for mid in merge_ids:
                    for c in clusters:
                        if c.get('cluster_id') == mid:
                            combined_msg_ids.extend(c.get('message_ids', []))
                            combined_parts.update(c.get('participants', []))
                            processed_ids.add(mid)
                
                refined_clusters.append({
                    "cluster_id": f"merged_{'_'.join(merge_ids)}",
                    "message_ids": combined_msg_ids,
                    "draft_title": f"Merged: {' + '.join(merge_ids)}",
                    "participants": list(combined_parts)
                })
            
            # Keep as-is
            else:
                refined_clusters.append(cluster)
                processed_ids.add(cluster_id)
        
        return refined_clusters
    
    def get_working_model(self, provider: str, original_model: str) -> tuple:
        """Get a working model for this provider, or fallback"""
        # Try original model first
        client = get_model_client(provider, original_model)
        if client:
            return (provider, original_model, False)  # Not using fallback
        
        # Use fallback
        fallback_model = self.FALLBACK_MODELS.get(provider)
        if fallback_model:
            client = get_model_client(provider, fallback_model)
            if client:
                return (provider, fallback_model, True)  # Using fallback
        
        return (None, None, False)
    
    def evaluate_model(self, provider: str, model_name: str, step1_result: Dict) -> Dict:
        """Evaluate with fallback to working models"""
        print(f"\n{'='*60}")
        print(f"🤖 Processing {provider}/{model_name}")
        print(f"{'='*60}")
        
        try:
            clusters = step1_result.get('clusters', [])
            print(f"📊 Step 1 clusters: {len(clusters)}")
            
            if not clusters:
                return {
                    "provider": provider,
                    "model": model_name,
                    "success": False,
                    "error": "No clusters"
                }
            
            # Get working model (original or fallback)
            work_provider, work_model, is_fallback = self.get_working_model(provider, model_name)
            
            if not work_provider:
                return {
                    "provider": provider,
                    "model": model_name,
                    "success": False,
                    "error": "No working model available"
                }
            
            if is_fallback:
                print(f"⚠️ {model_name} not available, using fallback: {work_model}")
            
            # Create prompt
            prompt = self.create_ai_prompt(clusters)
            
            # Call AI
            print(f"🤖 Calling {work_model} for decisions...")
            client = get_model_client(work_provider, work_model)
            
            start_time = time.time()
            result = client.call_model(prompt, max_tokens=4000, temperature=0.3)
            duration = time.time() - start_time
            
            if not result.get('success'):
                return {
                    "provider": provider,
                    "model": model_name,
                    "success": False,
                    "error": result.get('error'),
                    "duration": duration
                }
            
            # Parse and apply
            ai_response = result.get('response', '')
            operations = self.parse_ai_response(ai_response)
            
            if not operations:
                return {
                    "provider": provider,
                    "model": model_name,
                    "success": False,
                    "error": "Failed to parse",
                    "duration": duration
                }
            
            refined_clusters = self.apply_operations(clusters, operations)
            
            num_merges = len(operations.get('merge_operations', []))
            num_splits = len(operations.get('split_operations', []))
            
            print(f"✅ Success! {len(clusters)} → {len(refined_clusters)} clusters (M:{num_merges}, S:{num_splits})")
            
            return {
                "provider": provider,
                "model": model_name,
                "step2_model_used": work_model if is_fallback else model_name,
                "used_fallback": is_fallback,
                "phase": "step2_ai_with_fallback",
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
            return {
                "provider": provider,
                "model": model_name,
                "success": False,
                "error": str(e),
                "duration": 0
            }
    
    def run_evaluation(self):
        """Run evaluation with fallback strategy"""
        print("🚀 Phase 4: AI Decision-Making WITH FALLBACK")
        print("=" * 70)
        print("�� Strategy: Use Step 1 clusters, but fallback to working models for Step 2")
        print("=" * 70)
        
        if not self.step1_results:
            print("❌ No Step 1 results")
            return
        
        results = {}
        successful = 0
        
        for step1_result in self.step1_results:
            provider = step1_result['provider']
            model_name = step1_result['model']
            
            result = self.evaluate_model(provider, model_name, step1_result)
            results[f"{provider}_{model_name}"] = result
            
            output_file = os.path.join(self.output_dir, f"{provider}_{model_name}_fallback.json")
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            
            if result['success']:
                successful += 1
                fallback_info = f" (using {result['step2_model_used']})" if result.get('used_fallback') else ""
                print(f"💾 Saved{fallback_info}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown')[:50]}")
        
        comprehensive_file = os.path.join(self.output_dir, "comprehensive_results.json")
        with open(comprehensive_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"📊 COMPLETE")
        print(f"✅ Successful: {successful}/{len(self.step1_results)}")
        print(f"💾 Results: {self.output_dir}")

def main():
    evaluator = Phase4AIWithFback()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
