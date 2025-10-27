#!/usr/bin/env python3
"""
Phase 4: AI Decision-Making FOCUSED ON SPLITTING
AI models analyze clusters and split them into more granular topics
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
from config.model_config import get_available_models

class Phase4AISplitFocused:
    """AI models make split decisions to create more granular clusters"""
    
    def __init__(self):
        self.phase_name = "phase4_ai_split_focused"
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
                    
                    if result.get('success', False) and result.get('clusters'):
                        successful_results.append(result)
                        print(f"✅ {result['provider']}/{result['model']} - {len(result['clusters'])} clusters")
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
    
    def get_cluster_messages_text(self, cluster: Dict, limit: int = 20) -> str:
        """Get message texts for a cluster"""
        message_ids = cluster.get('message_ids', [])
        texts = []
        for msg_id in message_ids[:limit]:
            if msg_id <= len(self.messages):
                msg = self.messages[msg_id - 1]
                texts.append(f"[{msg_id}] {msg['text'][:150]}...")  # Truncate for token efficiency
        return "\n".join(texts)
    
    def create_split_focused_prompt(self, clusters: List[Dict], provider: str, model: str) -> str:
        """Create prompt that encourages AI to split clusters into granular topics"""
        
        # Create detailed cluster analysis
        cluster_analysis = []
        for cluster in clusters:
            msg_count = len(cluster.get('message_ids', []))
            sample_messages = self.get_cluster_messages_text(cluster, limit=20)
            
            cluster_analysis.append(f"""
Cluster ID: {cluster.get('cluster_id')}
Title: {cluster.get('draft_title')}
Messages: {msg_count}
Participants: {', '.join(cluster.get('participants', []))}
Thread: {cluster.get('thread_id', 'N/A')}

Sample Messages:
{sample_messages}
""")
        
        clusters_text = "\n" + "="*80 + "\n".join(cluster_analysis)
        
        prompt = f"""You are an expert at analyzing Slack conversation clusters and splitting them into more granular, focused topics.

**YOUR PRIMARY TASK**: SPLIT these {len(clusters)} clusters into MORE SPECIFIC, GRANULAR topics.

**GOAL**: Increase cluster count from {len(clusters)} to approximately 15-20 clusters by identifying sub-topics within each cluster.

**CURRENT CLUSTERS FROM STEP 1**:
{clusters_text}

**SPLITTING CRITERIA** (Focus on these):

1. **Project Phases**: Split by planning, execution, review, approval phases
   - Example: "EcoBloom Campaign" → "EcoBloom Planning", "EcoBloom Design", "EcoBloom Approval"

2. **Discussion Topics**: Split by distinct sub-topics within a project
   - Example: "FitFusion Rebranding" → "FitFusion Legal Review", "FitFusion Brand Guidelines", "FitFusion Tagline"

3. **Temporal Sequences**: Split by time-based conversation flow
   - Early discussions vs later discussions
   - Initial planning vs final delivery

4. **Deliverable Types**: Split by specific deliverables
   - Content creation vs Design work vs Legal approval

5. **Large Clusters**: Any cluster with >20 messages should be split into 2-3 sub-clusters

**MERGING CRITERIA** (Secondary - only if very obvious):
- Only merge if they're the EXACT same discussion topic AND same time period
- Prefer splitting over merging

**OUTPUT FORMAT** (valid JSON only):
{{
  "split_operations": [
    {{
      "cluster": "cluster_001",
      "reason": "Contains multiple phases: planning (msgs 1-5), design revisions (msgs 6-10), final approval (msgs 11-15)",
      "suggested_splits": [
        {{
          "new_id": "cluster_001_planning",
          "message_ids": [1, 2, 3, 4, 5],
          "title": "EcoBloom Campaign: Initial Planning & Kickoff",
          "phase": "planning"
        }},
        {{
          "new_id": "cluster_001_design",
          "message_ids": [6, 7, 8, 9, 10],
          "title": "EcoBloom Campaign: Design Revisions",
          "phase": "execution"
        }},
        {{
          "new_id": "cluster_001_approval",
          "message_ids": [11, 12, 13, 14, 15],
          "title": "EcoBloom Campaign: Final Approval & Delivery",
          "phase": "approval"
        }}
      ],
      "confidence": 0.9
    }}
  ],
  "merge_operations": [
    {{
      "clusters": ["cluster_002", "cluster_003"],
      "reason": "Both discuss exact same topic in same timeframe",
      "confidence": 0.85
    }}
  ],
  "reasoning": "Focused on splitting clusters by project phases and sub-topics to create more granular organization. Target: 15-20 final clusters."
}}

**IMPORTANT**: 
- Analyze message content carefully to identify natural split points
- Look for phase transitions (planning→execution→approval)
- Look for topic changes within clusters
- Aim for 15-20 final clusters total
- Each split should create 2-4 sub-clusters

Provide your split/merge analysis in JSON format.
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
            print(f"⚠️ Failed to parse JSON: {e}")
            return None
    
    def apply_split_operations(self, clusters: List[Dict], operations: Dict) -> List[Dict]:
        """Apply split and merge operations"""
        refined_clusters = []
        processed_ids = set()
        
        # Process splits first
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
            
            # Split this cluster
            if cluster_id in split_map:
                for split_cluster in split_map[cluster_id]:
                    refined_clusters.append(split_cluster)
                processed_ids.add(cluster_id)
            
            # Merge this cluster
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
        """Evaluate AI model's split-focused decisions"""
        print(f"\n{'='*70}")
        print(f"🤖 Testing {provider}/{model_name}")
        print(f"{'='*70}")
        
        try:
            clusters = step1_result.get('clusters', [])
            print(f"📊 Step 1: {len(clusters)} clusters")
            
            # Create split-focused prompt
            prompt = self.create_split_focused_prompt(clusters, provider, model_name)
            
            # Call AI model
            print(f"🤖 Calling {model_name} to make SPLIT decisions...")
            client = get_model_client(provider, model_name)
            
            if not client:
                return {
                    "provider": provider,
                    "model": model_name,
                    "success": False,
                    "error": "No API client available"
                }
            
            start_time = time.time()
            result = client.call_model(prompt, max_tokens=6000, temperature=0.3)
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
            refined_clusters = self.apply_split_operations(clusters, operations)
            
            # Calculate metrics
            num_splits = len(operations.get('split_operations', []))
            num_merges = len(operations.get('merge_operations', []))
            
            print(f"✅ AI Decisions:")
            print(f"   Splits: {num_splits}")
            print(f"   Merges: {num_merges}")
            print(f"   Step 1: {len(clusters)} clusters")
            print(f"   Step 2: {len(refined_clusters)} refined clusters")
            print(f"   Change: {len(refined_clusters) - len(clusters):+d}")
            print(f"   Duration: {duration:.2f}s")
            
            return {
                "provider": provider,
                "model": model_name,
                "phase": "step2_ai_split_focused",
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "duration": duration,
                "step1_clusters": clusters,
                "ai_operations": operations,
                "refined_clusters": refined_clusters,
                "metrics": {
                    "step1_clusters": len(clusters),
                    "refined_clusters": len(refined_clusters),
                    "split_operations": num_splits,
                    "merge_operations": num_merges,
                    "cluster_change": len(refined_clusters) - len(clusters),
                    "expansion_ratio": len(refined_clusters) / len(clusters) if len(clusters) > 0 else 0
                },
                "usage": result.get('usage', {}),
                "cost": self.calculate_cost(provider, model_name, result.get('usage', {})),
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
    
    def calculate_cost(self, provider: str, model_name: str, usage: Dict) -> Dict:
        """Calculate API costs"""
        # Simplified cost calculation
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        
        # Rough cost estimates (per 1M tokens)
        cost_rates = {
            "openai": {"input": 5.0, "output": 15.0},
            "google": {"input": 0.075, "output": 0.3},
            "anthropic": {"input": 3.0, "output": 15.0},
            "xai": {"input": 5.0, "output": 15.0},
            "groq": {"input": 0.0, "output": 0.0}
        }
        
        rates = cost_rates.get(provider, {"input": 1.0, "output": 3.0})
        
        input_cost = (input_tokens / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    
    def run_evaluation(self):
        """Run AI split-focused evaluation"""
        print("🚀 Starting Phase 4: AI SPLIT-FOCUSED Decision-Making")
        print("=" * 70)
        print("🎯 GOAL: Split clusters into more granular topics (6 → 15+)")
        print("🤖 AI models will analyze and split clusters by phases/sub-topics")
        print("=" * 70)
        
        if not self.step1_results:
            print("❌ No successful Step 1 results found")
            return
        
        results = {}
        successful = 0
        
        for step1_result in self.step1_results:
            provider = step1_result['provider']
            model_name = step1_result['model']
            
            result = self.evaluate_model(provider, model_name, step1_result)
            results[f"{provider}_{model_name}"] = result
            
            # Save individual result
            output_file = os.path.join(self.output_dir, f"{provider}_{model_name}_split_focused.json")
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            
            if result['success']:
                successful += 1
                metrics = result['metrics']
                print(f"💾 Saved: {output_file}")
                print(f"📈 Cluster expansion: {metrics['step1_clusters']} → {metrics['refined_clusters']} ({metrics['expansion_ratio']:.2f}x)")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown')}")
        
        # Save comprehensive results
        comprehensive_file = os.path.join(self.output_dir, "comprehensive_results.json")
        with open(comprehensive_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        self.print_summary(results)
    
    def print_summary(self, results: Dict):
        """Print evaluation summary"""
        print(f"\n{'='*70}")
        print(f"📊 EVALUATION SUMMARY")
        print(f"{'='*70}")
        
        successful_results = {k: v for k, v in results.items() if v['success']}
        
        print(f"✅ Successful: {len(successful_results)}/{len(results)}")
        
        if successful_results:
            print(f"\n🏆 Best Performers:")
            
            # Most splits
            best_splits = max(successful_results.items(), 
                            key=lambda x: x[1]['metrics']['split_operations'])
            print(f"  Most Splits: {best_splits[0]} ({best_splits[1]['metrics']['split_operations']} splits)")
            
            # Highest expansion ratio
            best_expansion = max(successful_results.items(),
                                key=lambda x: x[1]['metrics']['expansion_ratio'])
            print(f"  Best Expansion: {best_expansion[0]} " +
                  f"({best_expansion[1]['metrics']['step1_clusters']} → " +
                  f"{best_expansion[1]['metrics']['refined_clusters']} clusters)")
            
            # Most cost-effective
            best_cost = min(successful_results.items(),
                          key=lambda x: x[1]['cost']['total_cost'])
            print(f"  Most Cost-Effective: {best_cost[0]} (${best_cost[1]['cost']['total_cost']:.6f})")
        
        print(f"\n💾 Results saved to: {self.output_dir}")

def main():
    evaluator = Phase4AISplitFocused()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
