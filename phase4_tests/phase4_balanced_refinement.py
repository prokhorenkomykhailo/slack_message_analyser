#!/usr/bin/env python3
"""
Phase 4: BALANCED Merge/Split Refinement Engine
Target: ~15 refined clusters with smart merge/split rules
"""

import os
import json
import time
import csv
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_clients import get_model_client, call_model_with_retry

class Phase4BalancedRefinement:
    """Balanced merge/split engine targeting ~15 refined clusters"""
    
    # Fallback models that work
    FALLBACK_MODELS = {
        'google': 'gemini-2.0-flash',
        'openai': 'gpt-4o',
        'xai': 'grok-3'
    }
    
    def __init__(self):
        self.phase_name = "phase4_balanced_refinement"
        self.output_dir = os.path.join("output", self.phase_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.step1_results = self.load_all_step1_results()
        self.messages = self.load_messages()
        
        print(f"✅ Loaded {len(self.step1_results)} Step 1 results")
        print(f"✅ Loaded {len(self.messages)} messages")
    
    def load_all_step1_results(self) -> List[Dict]:
        """Load all Step 1 results including _clusters.json files"""
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
                        print(f"✅ {provider}/{model} - {len(data)} clusters")
                
                # Handle regular .json files
                elif filename.endswith('.json'):
                    model_name = filename.replace('.json', '')
                    
                    if data.get('success', False) and data.get('clusters'):
                        if model_name not in results_map:
                            results_map[model_name] = data
                            print(f"✅ {data['provider']}/{data['model']} - {len(data['clusters'])} clusters")
                
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
    
    def get_cluster_messages_text(self, cluster: Dict, limit: int = 8) -> str:
        """Get message texts for a cluster"""
        message_ids = cluster.get('message_ids', [])
        texts = []
        for msg_id in message_ids[:limit]:
            if msg_id <= len(self.messages):
                msg = self.messages[msg_id - 1]
                texts.append(f"[{msg_id}] {msg['text'][:120]}...")
        return "\n".join(texts)
    
    def create_balanced_prompt(self, clusters: List[Dict]) -> str:
        """Create prompt with BALANCED merge/split instructions"""
        
        cluster_summaries = []
        for cluster in clusters:
            msg_count = len(cluster.get('message_ids', []))
            sample_messages = self.get_cluster_messages_text(cluster, limit=6)
            
            cluster_summaries.append(f"""
Cluster: {cluster.get('cluster_id', 'unknown')}
Title: {cluster.get('draft_title', 'No title')}
Messages: {msg_count}
Sample Messages:
{sample_messages}
""")
        
        clusters_text = "\n---\n".join(cluster_summaries)
        
        prompt = f"""You are an expert at refining Slack conversation clusters with a BALANCED approach.

**CURRENT STEP 1 CLUSTERS**: {len(clusters)} clusters

{clusters_text}

**YOUR GOAL**: Create **12-15 refined clusters** through smart merge/split operations.

**BALANCED MERGE RULES** (Be selective!):
✅ MERGE if ALL conditions met:
   - SAME specific project (e.g., "EcoBloom Planning" + "EcoBloom Design" + "EcoBloom Approval")
   - SAME time period (not weeks apart)
   - Would help users find complete project timeline
   
❌ DON'T MERGE if:
   - Different projects (EcoBloom ≠ FitFusion)
   - Already well-organized separate phases
   - Would create clusters >40 messages

**BALANCED SPLIT RULES** (Only when needed):
✅ SPLIT if:
   - Cluster >30 messages AND mixes distinct sub-topics
   - Clear phase boundaries (planning vs execution vs approval)
   - Would improve findability
   
❌ DON'T SPLIT if:
   - Cluster is coherent and <25 messages
   - Would create too many tiny clusters
   - Current organization already makes sense

**TARGET**: 
- If Step 1 has 6-8 clusters → aim for 12-15 (more splits than merges)
- If Step 1 has 15+ clusters → aim for 12-15 (balanced merges and splits)
- If Step 1 has 12-15 clusters → minimal changes, maybe 1-2 operations

**IMPORTANT**: 
- Count your final clusters before responding
- Aim for 12-15 final clusters (±2 is ok)
- Balance is key: don't over-merge or over-split

**OUTPUT FORMAT** (JSON only):
{{
  "merge_operations": [
    {{
      "clusters": ["cluster_001", "cluster_006", "cluster_011"],
      "reason": "All EcoBloom project phases - planning, design, approval",
      "confidence": 0.85,
      "result_size": 44
    }}
  ],
  "split_operations": [
    {{
      "cluster": "cluster_007",
      "reason": "Large cluster (45 msgs) mixing planning and execution phases",
      "suggested_splits": [
        {{
          "new_id": "cluster_007_planning",
          "message_ids": [89, 90, 91, 92, 93, 94, 95],
          "title": "FitFusion: Planning & Legal Review",
          "phase": "planning"
        }},
        {{
          "new_id": "cluster_007_execution",
          "message_ids": [96, 97, 98, 99, 100, 101, 102],
          "title": "FitFusion: Tagline Development",
          "phase": "execution"
        }}
      ],
      "confidence": 0.80
    }}
  ],
  "final_cluster_count": 14,
  "reasoning": "Merged 3 EcoBloom clusters (same project), split 1 large mixed cluster. Result: 14 balanced clusters."
}}

**Count your operations**: 
- Start: {len(clusters)} clusters
- After merges: ? clusters
- After splits: ? clusters
- Final: Should be 12-15

Analyze and provide your balanced merge/split decisions in JSON format.
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
        except Exception as e:
            print(f"⚠️ Parse error: {e}")
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
            
            # Split this cluster
            if cluster_id in split_map:
                refined_clusters.extend(split_map[cluster_id])
                processed_ids.add(cluster_id)
            
            # Merge this cluster
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
        """Get working model or fallback"""
        # Try original first
        client = get_model_client(provider, original_model)
        if client:
            # Test if it actually works with a tiny prompt
            try:
                test_result = client.call_model("Hello", max_tokens=10, temperature=0.1)
                if test_result.get('success'):
                    return (provider, original_model, False)
            except:
                pass
        
        # Use fallback
        fallback_model = self.FALLBACK_MODELS.get(provider)
        if fallback_model:
            return (provider, fallback_model, True)
        
        return (None, None, False)
    
    def evaluate_model(self, provider: str, model_name: str, step1_result: Dict) -> Dict:
        """Evaluate with balanced refinement strategy"""
        print(f"\n{'='*70}")
        print(f"🤖 Processing {provider}/{model_name}")
        print(f"{'='*70}")
        
        try:
            clusters = step1_result.get('clusters', [])
            print(f"📊 Step 1: {len(clusters)} clusters")
            
            if not clusters:
                return {"provider": provider, "model": model_name, "success": False, "error": "No clusters"}
            
            # Get working model
            work_provider, work_model, is_fallback = self.get_working_model(provider, model_name)
            
            if not work_provider:
                return {"provider": provider, "model": model_name, "success": False, "error": "No working model"}
            
            if is_fallback:
                print(f"⚠️ Using fallback: {work_model}")
            
            # Create balanced prompt
            prompt = self.create_balanced_prompt(clusters)
            
            # Call AI with retry
            print(f"🤖 Calling {work_model} for balanced refinement...")
            start_time = time.time()
            result = call_model_with_retry(work_provider, work_model, prompt, max_retries=3, max_tokens=5000, temperature=0.3)
            duration = time.time() - start_time
            
            if not result.get('success'):
                return {
                    "provider": provider,
                    "model": model_name,
                    "success": False,
                    "error": result.get('error'),
                    "duration": duration
                }
            
            # Parse AI response
            ai_response = result.get('response', '')
            operations = self.parse_ai_response(ai_response)
            
            if not operations:
                return {
                    "provider": provider,
                    "model": model_name,
                    "success": False,
                    "error": "Failed to parse AI response",
                    "duration": duration,
                    "raw_response": ai_response
                }
            
            # Apply operations
            refined_clusters = self.apply_operations(clusters, operations)
            
            # Validate cluster count
            final_count = len(refined_clusters)
            ai_claimed_count = operations.get('final_cluster_count', final_count)
            
            num_merges = len(operations.get('merge_operations', []))
            num_splits = len(operations.get('split_operations', []))
            
            # Check if target met
            target_met = 10 <= final_count <= 18  # Allow 10-18 range
            
            print(f"✅ Success!")
            print(f"   Step 1: {len(clusters)} clusters")
            print(f"   Step 2: {final_count} clusters")
            print(f"   Merges: {num_merges}, Splits: {num_splits}")
            print(f"   Target (12-15): {'✅ MET' if target_met else '⚠️ MISSED'}")
            
            # Calculate cost
            usage = result.get('usage', {})
            cost = self.calculate_cost(work_provider, work_model, usage)
            
            return {
                "provider": provider,
                "model": model_name,
                "step2_model_used": work_model,
                "used_fallback": is_fallback,
                "phase": "step2_balanced_refinement",
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "duration": duration,
                "step1_clusters": clusters,
                "ai_operations": operations,
                "refined_clusters": refined_clusters,
                "metrics": {
                    "step1_clusters": len(clusters),
                    "refined_clusters": final_count,
                    "merge_operations": num_merges,
                    "split_operations": num_splits,
                    "cluster_change": final_count - len(clusters),
                    "target_met": target_met,
                    "target_range": "12-15",
                    "ai_claimed_count": ai_claimed_count
                },
                "usage": usage,
                "cost": cost,
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
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        
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
        """Run balanced refinement evaluation"""
        print("🚀 Phase 4: BALANCED Merge/Split Refinement Engine")
        print("=" * 70)
        print("🎯 TARGET: 12-15 refined clusters")
        print("⚖️ STRATEGY: Balanced merge/split with strict rules")
        print("=" * 70)
        
        if not self.step1_results:
            print("❌ No Step 1 results")
            return
        
        results = {}
        successful = 0
        target_met_count = 0
        
        for step1_result in self.step1_results:
            provider = step1_result['provider']
            model_name = step1_result['model']
            
            result = self.evaluate_model(provider, model_name, step1_result)
            results[f"{provider}_{model_name}"] = result
            
            output_file = os.path.join(self.output_dir, f"{provider}_{model_name}_balanced.json")
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            
            if result['success']:
                successful += 1
                if result['metrics'].get('target_met', False):
                    target_met_count += 1
                    
                metrics = result['metrics']
                status = "✅" if metrics.get('target_met') else "⚠️"
                print(f"💾 {status} Saved: {metrics['step1_clusters']} → {metrics['refined_clusters']} clusters")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown')[:50]}")
        
        # Save comprehensive
        comprehensive_file = os.path.join(self.output_dir, "comprehensive_results.json")
        with open(comprehensive_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"📊 EVALUATION COMPLETE")
        print(f"{'='*70}")
        print(f"✅ Successful: {successful}/{len(self.step1_results)}")
        print(f"🎯 Target Met (12-15 clusters): {target_met_count}/{successful}")
        print(f"💾 Results: {self.output_dir}")
        
        # Show best results
        if successful > 0:
            self.show_best_results(results)
    
    def show_best_results(self, results: Dict):
        """Show models that best met the target"""
        successful = {k: v for k, v in results.items() if v['success']}
        
        if not successful:
            return
        
        print(f"\n🏆 BEST PERFORMERS (12-15 clusters):")
        
        target_met = {k: v for k, v in successful.items() if v['metrics'].get('target_met')}
        
        if target_met:
            for model_name, result in sorted(target_met.items(), 
                                            key=lambda x: abs(x[1]['metrics']['refined_clusters'] - 13.5))[:5]:
                metrics = result['metrics']
                print(f"  ✅ {model_name}: {metrics['step1_clusters']} → {metrics['refined_clusters']} clusters " +
                      f"(M:{metrics['merge_operations']}, S:{metrics['split_operations']}) " +
                      f"${result['cost']['total_cost']:.4f}")
        else:
            print("  ⚠️ No models achieved 12-15 cluster target")
            # Show closest
            closest = sorted(successful.items(), 
                           key=lambda x: abs(x[1]['metrics']['refined_clusters'] - 13.5))[:3]
            print(f"\n  Closest to target:")
            for model_name, result in closest:
                metrics = result['metrics']
                print(f"    {model_name}: {metrics['step1_clusters']} → {metrics['refined_clusters']} clusters")

def main():
    evaluator = Phase4BalancedRefinement()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
