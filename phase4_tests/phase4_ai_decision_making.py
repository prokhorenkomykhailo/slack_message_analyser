#!/usr/bin/env python3
"""
Phase 4: TRUE AI Decision-Making for Merge/Split
AI models make the actual decisions, not fixed thresholds
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
from config.model_config import get_available_models, get_model_config

class Phase4AIDecisionMaking:
    """AI models make merge/split decisions based on content analysis"""
    
    def __init__(self):
        self.phase_name = "phase4_ai_decision_making"
        self.output_dir = os.path.join("output", self.phase_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load Step 1 results
        self.step1_results = self.load_successful_step1_results()
        self.messages = self.load_messages()
        
        print(f"✅ Loaded {len(self.step1_results)} successful Step 1 results")
        print(f"✅ Loaded {len(self.messages)} messages")
    
    def load_successful_step1_results(self) -> List[Dict]:
        """Load only successful Step 1 results"""
        step1_dir = "output/phase3_topic_clustering"
        successful_results = []
        
        for filename in os.listdir(step1_dir):
            if filename.endswith('.json') and not filename.startswith(('comprehensive_', 'detailed_', 'best_', 'results_', 'gpt_test')):
                filepath = os.path.join(step1_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        result = json.load(f)
                    
                    # Only include successful results with clusters
                    if result.get('success', False) and result.get('clusters'):
                        successful_results.append(result)
                        print(f"✅ Loaded {result['provider']}/{result['model']} - {len(result['clusters'])} clusters")
                except Exception as e:
                    print(f"⚠️ Skipped {filename}: {e}")
        
        return successful_results
    
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
    
    def get_cluster_messages_text(self, cluster: Dict) -> str:
        """Get all message texts for a cluster"""
        message_ids = cluster.get('message_ids', [])
        texts = []
        for msg_id in message_ids[:10]:  # Limit to first 10 to save tokens
            if msg_id <= len(self.messages):
                msg = self.messages[msg_id - 1]
                texts.append(f"[{msg_id}] {msg['text']}")
        return "\n".join(texts)
    
    def create_ai_prompt(self, clusters: List[Dict], provider: str, model: str) -> str:
        """Create prompt for AI to make merge/split decisions"""
        
        # Create cluster summary with sample messages
        cluster_summaries = []
        for cluster in clusters:
            msg_count = len(cluster.get('message_ids', []))
            sample_messages = self.get_cluster_messages_text(cluster)
            
            cluster_summaries.append(f"""
Cluster: {cluster.get('cluster_id', 'unknown')}
Title: {cluster.get('draft_title', 'No title')}
Messages: {msg_count}
Participants: {', '.join(cluster.get('participants', []))}
Sample Messages (first 10):
{sample_messages}
""")
        
        clusters_text = "\n---\n".join(cluster_summaries)
        
        prompt = f"""You are an expert at analyzing Slack conversation clusters and making intelligent decisions about whether clusters should be merged or split.

**YOUR TASK**: Analyze these {len(clusters)} clusters from Step 1 and decide which operations to perform.

**STEP 1 CLUSTERS**:
{clusters_text}

**DECISION CRITERIA**:

**For MERGING clusters**, consider:
- Are they about the SAME PROJECT (e.g., both EcoBloom, both FitFusion)?
- Do they discuss the SAME TOPIC (e.g., both about planning, both about design)?
- Are messages semantically related even if different projects?
- Would merging help users find related conversations?

**For SPLITTING clusters**, consider:
- Does cluster mix MULTIPLE PROJECTS (EcoBloom + FitFusion together)?
- Does cluster mix DIFFERENT PHASES (planning + approval + execution)?
- Are messages about DISTINCT TOPICS that should be separate?
- Is cluster too large (>30 messages) or incoherent?

**IMPORTANT**: 
- Different projects (EcoBloom vs FitFusion) should generally NOT be merged
- Same project different phases (planning vs approval) CAN be split or kept together
- Consider what makes sense for someone searching conversations

**OUTPUT FORMAT** (provide valid JSON only):
{{
  "merge_operations": [
    {{
      "clusters": ["cluster_001", "cluster_006"],
      "reason": "Both are EcoBloom campaign discussions - different phases of same project",
      "confidence": 0.85
    }}
  ],
  "split_operations": [
    {{
      "cluster": "cluster_002",
      "reason": "Mixes FitFusion and TechNova - different projects should be separate",
      "suggested_splits": [
        {{"new_id": "cluster_002_fitfusion", "message_ids": [14, 15, 16], "title": "FitFusion Rebranding"}},
        {{"new_id": "cluster_002_technova", "message_ids": [29, 30, 31], "title": "TechNova Launch"}}
      ],
      "confidence": 0.90
    }}
  ],
  "keep_as_is": ["cluster_003", "cluster_004"],
  "reasoning": "Your overall reasoning for these decisions"
}}

Analyze the clusters and provide your merge/split decisions in JSON format.
"""
        return prompt
    
    def parse_ai_response(self, response_text: str) -> Dict:
        """Parse AI model's JSON response"""
        try:
            # Extract JSON from response
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
            print(f"⚠️ Failed to parse JSON: {e}")
            return None
    
    def apply_operations(self, clusters: List[Dict], operations: Dict) -> List[Dict]:
        """Apply merge/split operations to create refined clusters"""
        refined_clusters = []
        processed_ids = set()
        
        # Process merge operations
        merge_map = {}
        for merge_op in operations.get('merge_operations', []):
            cluster_ids = merge_op.get('clusters', [])
            if len(cluster_ids) >= 2:
                # First cluster becomes the merged cluster
                merge_map[cluster_ids[0]] = cluster_ids
        
        # Process splits
        split_results = {}
        for split_op in operations.get('split_operations', []):
            cluster_id = split_op.get('cluster')
            suggested_splits = split_op.get('suggested_splits', [])
            if cluster_id and suggested_splits:
                split_results[cluster_id] = suggested_splits
        
        # Build refined clusters
        for cluster in clusters:
            cluster_id = cluster.get('cluster_id')
            
            if cluster_id in processed_ids:
                continue
            
            # Check if this cluster should be split
            if cluster_id in split_results:
                for split_cluster in split_results[cluster_id]:
                    refined_clusters.append(split_cluster)
                processed_ids.add(cluster_id)
            
            # Check if this cluster should be merged
            elif cluster_id in merge_map:
                merge_ids = merge_map[cluster_id]
                # Combine all message IDs from merged clusters
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
            
            # Create AI prompt
            prompt = self.create_ai_prompt(clusters, provider, model_name)
            
            # Call AI model to make decisions
            print(f"🤖 Calling {model_name} to make merge/split decisions...")
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
            
            print(f"✅ AI Decisions:")
            print(f"   Merges: {num_merges}")
            print(f"   Splits: {num_splits}")
            print(f"   Step 1: {len(clusters)} clusters")
            print(f"   Step 2: {len(refined_clusters)} refined clusters")
            print(f"   Duration: {duration:.2f}s")
            
            return {
                "provider": provider,
                "model": model_name,
                "phase": "step2_ai_decision_making",
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
        print("🚀 Starting Phase 4: TRUE AI Decision-Making for Merge/Split")
        print("=" * 70)
        print("🤖 AI models will make the actual merge/split decisions")
        print("📊 NOT using fixed thresholds - each AI decides independently")
        print("=" * 70)
        
        if not self.step1_results:
            print("❌ No successful Step 1 results found")
            return
        
        results = {}
        successful = 0
        
        for step1_result in self.step1_results:
            provider = step1_result['provider']
            model_name = step1_result['model']
            
            print(f"\n🔍 Processing {provider}/{model_name}...")
            
            result = self.evaluate_model(provider, model_name, step1_result)
            results[f"{provider}_{model_name}"] = result
            
            # Save individual result
            output_file = os.path.join(self.output_dir, f"{provider}_{model_name}_ai_decisions.json")
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            
            if result['success']:
                successful += 1
                print(f"✅ Saved to: {output_file}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown')}")
        
        # Save comprehensive results
        comprehensive_file = os.path.join(self.output_dir, "comprehensive_results.json")
        with open(comprehensive_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"📊 EVALUATION COMPLETE")
        print(f"{'='*70}")
        print(f"✅ Successful: {successful}/{len(self.step1_results)}")
        print(f"❌ Failed: {len(self.step1_results) - successful}/{len(self.step1_results)}")
        print(f"💾 Results saved to: {self.output_dir}")

def main():
    evaluator = Phase4AIDecisionMaking()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
