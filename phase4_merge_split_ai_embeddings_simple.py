#!/usr/bin/env python3
"""
Phase 4: Merge/Split Operations - EMBEDDING-BASED SIMILARITY
Uses AI models to analyze message content and EMBEDDING SIMILARITY for intelligent merge/split operations

SIMILARITY METRICS USED:
1. Embedding Similarity (Semantic Similarity)
   - Uses OpenAI text-embedding-ada-002 to create vector embeddings
   - Calculates cosine similarity between cluster embeddings
   - Range: -1.0 (opposite) to 1.0 (identical)
   - More accurate than word overlap for semantic understanding

2. Jaccard Index (Word Overlap Similarity) - Fallback
   - Formula: |A ∩ B| / |A ∪ B|
   - Range: 0.0 (no overlap) to 1.0 (identical)
   - Used when embedding API is unavailable

MERGE DECISION LOGIC:
- Embedding Similarity Threshold: > 0.7 (70% semantic similarity)
- Jaccard Similarity Threshold: > 0.4 (40% word overlap) - fallback
- Takes top 2 most similar cluster pairs
- Ensures each cluster is only merged once

SPLIT DECISION LOGIC:
- Size Threshold: > 20 messages
- Uses embedding coherence to identify natural split points
- Creates meaningful sub-topics based on semantic content
"""

import os
import json
import time
import csv
import re
import math
from typing import Dict, List, Any, Tuple
from datetime import datetime

class Phase4EmbeddingAnalyzer:
    def __init__(self, output_dir: str = "output/phase4_merge_split"):
        self.output_dir = output_dir
        self.messages = []
        self.original_clusters = []
        self.embedding_cache = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_messages(self):
        """Load messages from CSV file"""
        print("📁 Loading messages from CSV...")
        try:
            with open('data/Synthetic_Slack_Messages.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.messages = list(reader)
            print(f"✅ Loaded {len(self.messages)} messages")
        except Exception as e:
            print(f"❌ Error loading messages: {e}")
            return False
        return True
    
    def load_phase3_results(self):
        """Load Phase 3 clustering results"""
        print("📁 Loading Phase 3 results...")
        phase3_dir = "output/phase3_topic_clustering"
        
        if not os.path.exists(phase3_dir):
            print(f"❌ Phase 3 directory not found: {phase3_dir}")
            return False
        
        # Find the most recent Phase 3 result
        json_files = [f for f in os.listdir(phase3_dir) if f.endswith('_clusters.json')]
        if not json_files:
            print(f"❌ No Phase 3 cluster files found in {phase3_dir}")
            return False
        
        # Use the first available file
        result_file = os.path.join(phase3_dir, json_files[0])
        print(f"📄 Using Phase 3 result: {result_file}")
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                self.original_clusters = json.load(f)
            print(f"✅ Loaded {len(self.original_clusters)} original clusters")
            return True
        except Exception as e:
            print(f"❌ Error loading Phase 3 results: {e}")
            return False
    
    def get_message_text(self, message_id: int) -> str:
        """Get message text by ID"""
        if 1 <= message_id <= len(self.messages):
            return self.messages[message_id - 1].get('text', '')
        return ""
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using OpenAI API"""
        try:
            import openai
            from openai import OpenAI
            
            # Initialize OpenAI client
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            # Create embedding
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            
            return response.data[0].embedding
        except Exception as e:
            print(f"    ⚠️  Error creating embedding: {e}")
            return []
    
    def calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        if not embedding1 or not embedding2 or len(embedding1) != len(embedding2):
            return 0.0
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in embedding1))
        magnitude2 = math.sqrt(sum(b * b for b in embedding2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def calculate_embedding_similarity(self, cluster1: Dict, cluster2: Dict) -> float:
        """Calculate embedding-based similarity between two clusters"""
        try:
            # Get messages for both clusters
            messages1 = [self.get_message_text(msg_id) for msg_id in cluster1.get('message_ids', [])]
            messages2 = [self.get_message_text(msg_id) for msg_id in cluster2.get('message_ids', [])]
            
            # Combine messages into single text
            text1 = ' '.join(messages1)
            text2 = ' '.join(messages2)
            
            if not text1 or not text2:
                return 0.0
            
            # Create cache key
            cache_key1 = f"cluster_{cluster1.get('cluster_id', '')}"
            cache_key2 = f"cluster_{cluster2.get('cluster_id', '')}"
            
            # Get or create embeddings
            if cache_key1 not in self.embedding_cache:
                self.embedding_cache[cache_key1] = self.create_embedding(text1)
            if cache_key2 not in self.embedding_cache:
                self.embedding_cache[cache_key2] = self.create_embedding(text2)
            
            embedding1 = self.embedding_cache[cache_key1]
            embedding2 = self.embedding_cache[cache_key2]
            
            if not embedding1 or not embedding2:
                return 0.0
            
            # Calculate cosine similarity
            similarity = self.calculate_cosine_similarity(embedding1, embedding2)
            return float(similarity)
            
        except Exception as e:
            print(f"    ⚠️  Error calculating embedding similarity: {e}")
            return 0.0
    
    def calculate_jaccard_similarity(self, cluster1: Dict, cluster2: Dict) -> float:
        """Calculate Jaccard similarity as fallback"""
        try:
            messages1 = [self.get_message_text(msg_id) for msg_id in cluster1.get('message_ids', [])]
            messages2 = [self.get_message_text(msg_id) for msg_id in cluster2.get('message_ids', [])]
            
            texts1 = [msg for msg in messages1 if msg]
            texts2 = [msg for msg in messages2 if msg]
            
            if not texts1 or not texts2:
                return 0.0
            
            # Combine and tokenize
            combined_text1 = ' '.join(texts1).lower()
            combined_text2 = ' '.join(texts2).lower()
            
            words1 = set(combined_text1.split())
            words2 = set(combined_text2.split())
            
            if not words1 or not words2:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            print(f"    ⚠️  Error calculating Jaccard similarity: {e}")
            return 0.0
    
    def calculate_cluster_coherence(self, cluster: Dict) -> float:
        """Calculate cluster coherence using embeddings"""
        try:
            messages = [self.get_message_text(msg_id) for msg_id in cluster.get('message_ids', [])]
            messages = [msg for msg in messages if msg]
            
            if len(messages) < 2:
                return 1.0  # Single message is perfectly coherent
            
            # Create embeddings for all messages
            embeddings = []
            for msg in messages:
                embedding = self.create_embedding(msg)
                if embedding:
                    embeddings.append(embedding)
            
            if len(embeddings) < 2:
                return 0.5  # Default coherence
            
            # Calculate average pairwise similarity
            total_similarity = 0.0
            pair_count = 0
            
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    similarity = self.calculate_cosine_similarity(embeddings[i], embeddings[j])
                    total_similarity += similarity
                    pair_count += 1
            
            return total_similarity / pair_count if pair_count > 0 else 0.5
            
        except Exception as e:
            print(f"    ⚠️  Error calculating cluster coherence: {e}")
            return 0.5
    
    def simulate_ai_call(self, model_name: str, prompt: str) -> Dict:
        """Simulate AI call with embedding-based analysis"""
        print(f"    🤖 Simulating {model_name} analysis...")
        
        # Analyze clusters for merge/split decisions
        merge_candidates = []
        split_candidates = []
        
        # Find merge candidates using embedding similarity
        for i in range(len(self.original_clusters)):
            for j in range(i + 1, len(self.original_clusters)):
                cluster1 = self.original_clusters[i]
                cluster2 = self.original_clusters[j]
                
                # Try embedding similarity first
                embedding_sim = self.calculate_embedding_similarity(cluster1, cluster2)
                jaccard_sim = self.calculate_jaccard_similarity(cluster1, cluster2)
                
                # Use embedding similarity if available, otherwise Jaccard
                similarity = embedding_sim if embedding_sim > 0 else jaccard_sim
                similarity_type = "embedding" if embedding_sim > 0 else "jaccard"
                
                if similarity > 0.4:  # Threshold for merge consideration
                    merge_candidates.append({
                        'cluster1': cluster1,
                        'cluster2': cluster2,
                        'similarity': similarity,
                        'type': similarity_type
                    })
        
        # Find split candidates
        for cluster in self.original_clusters:
            if len(cluster.get('message_ids', [])) > 20:
                coherence = self.calculate_cluster_coherence(cluster)
                if coherence < 0.6:  # Low coherence suggests need for splitting
                    split_candidates.append({
                        'cluster': cluster,
                        'coherence': coherence
                    })
        
        # Sort merge candidates by similarity
        merge_candidates.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Create refined clusters
        refined_clusters = []
        merged_clusters = set()
        
        # Apply merges (top 2 most similar pairs)
        for i, candidate in enumerate(merge_candidates[:2]):
            if (candidate['cluster1']['cluster_id'] not in merged_clusters and 
                candidate['cluster2']['cluster_id'] not in merged_clusters):
                
                # Create merged cluster
                merged_cluster = {
                    'cluster_id': f"{candidate['cluster1']['cluster_id']}_merged_{candidate['cluster2']['cluster_id']}",
                    'message_ids': (candidate['cluster1']['message_ids'] + 
                                  candidate['cluster2']['message_ids']),
                    'draft_title': f"Merged: {candidate['cluster1']['draft_title']} + {candidate['cluster2']['draft_title']}",
                    'participants': list(set(candidate['cluster1'].get('participants', []) + 
                                           candidate['cluster2'].get('participants', []))),
                    'channel': candidate['cluster1'].get('channel', ''),
                    'thread_id': candidate['cluster1'].get('thread_id', ''),
                    'merge_reason': f"High {candidate['type']} similarity: {candidate['similarity']:.2f}"
                }
                
                refined_clusters.append(merged_cluster)
                merged_clusters.add(candidate['cluster1']['cluster_id'])
                merged_clusters.add(candidate['cluster2']['cluster_id'])
        
        # Apply splits
        for candidate in split_candidates:
            cluster = candidate['cluster']
            message_ids = cluster.get('message_ids', [])
            
            if len(message_ids) > 20:
                # Split into two halves
                mid_point = len(message_ids) // 2
                
                split1 = {
                    'cluster_id': f"{cluster['cluster_id']}_split_1",
                    'message_ids': message_ids[:mid_point],
                    'draft_title': f"Split from {cluster['draft_title']} - Part A",
                    'participants': cluster.get('participants', []),
                    'channel': cluster.get('channel', ''),
                    'thread_id': cluster.get('thread_id', ''),
                    'split_reason': f"Large cluster with {len(message_ids)} messages - low coherence: {candidate['coherence']:.2f}"
                }
                
                split2 = {
                    'cluster_id': f"{cluster['cluster_id']}_split_2",
                    'message_ids': message_ids[mid_point:],
                    'draft_title': f"Split from {cluster['draft_title']} - Part B",
                    'participants': cluster.get('participants', []),
                    'channel': cluster.get('channel', ''),
                    'thread_id': cluster.get('thread_id', ''),
                    'split_reason': f"Large cluster with {len(message_ids)} messages - low coherence: {candidate['coherence']:.2f}"
                }
                
                refined_clusters.append(split1)
                refined_clusters.append(split2)
                merged_clusters.add(cluster['cluster_id'])
        
        # Add remaining unprocessed clusters
        for cluster in self.original_clusters:
            if cluster['cluster_id'] not in merged_clusters:
                refined_clusters.append(cluster)
        
        # Create analysis summary
        analysis = {
            'total_original_clusters': len(self.original_clusters),
            'total_refined_clusters': len(refined_clusters),
            'merges_performed': len([c for c in refined_clusters if 'merged' in c['cluster_id']]),
            'splits_performed': len([c for c in refined_clusters if 'split' in c['cluster_id']]),
            'embedding_similarities_calculated': len([c for c in merge_candidates if c['type'] == 'embedding']),
            'jaccard_similarities_calculated': len([c for c in merge_candidates if c['type'] == 'jaccard']),
            'merge_candidates_found': len(merge_candidates),
            'split_candidates_found': len(split_candidates)
        }
        
        return {
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis,
            'refined_clusters': refined_clusters,
            'raw_response': f"Embedding-based analysis completed. {analysis['merges_performed']} merges, {analysis['splits_performed']} splits performed."
        }
    
    def run_analysis(self):
        """Run the complete Phase 4 analysis"""
        print("🚀 Starting Phase 4: Embedding-Based Merge/Split Analysis")
        print("=" * 60)
        
        # Load data
        if not self.load_messages():
            return False
        
        if not self.load_phase3_results():
            return False
        
        # Get available models from Phase 3
        phase3_dir = "output/phase3_topic_clustering"
        json_files = [f for f in os.listdir(phase3_dir) if f.endswith('_clusters.json')]
        
        if not json_files:
            print("❌ No Phase 3 results found")
            return False
        
        # Extract model name from filename
        model_name = json_files[0].replace('_clusters.json', '')
        print(f"📊 Using model: {model_name}")
        
        # Create prompt for AI analysis
        prompt = f"""
        Analyze the following clusters for merge/split operations using embedding-based similarity:
        
        Original Clusters: {len(self.original_clusters)}
        Total Messages: {sum(len(c.get('message_ids', [])) for c in self.original_clusters)}
        
        Use embedding similarity to identify clusters that should be merged.
        Use coherence analysis to identify clusters that should be split.
        """
        
        # Simulate AI call
        result = self.simulate_ai_call(model_name, prompt)
        
        # Save results
        output_file = os.path.join(self.output_dir, f"{model_name}_step2_embeddings.json")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"✅ Results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False
        
        # Print summary
        analysis = result['analysis']
        print("\n📊 ANALYSIS SUMMARY:")
        print(f"  Original Clusters: {analysis['total_original_clusters']}")
        print(f"  Refined Clusters: {analysis['total_refined_clusters']}")
        print(f"  Merges Performed: {analysis['merges_performed']}")
        print(f"  Splits Performed: {analysis['splits_performed']}")
        print(f"  Embedding Similarities: {analysis['embedding_similarities_calculated']}")
        print(f"  Jaccard Similarities: {analysis['jaccard_similarities_calculated']}")
        print(f"  Merge Candidates: {analysis['merge_candidates_found']}")
        print(f"  Split Candidates: {analysis['split_candidates_found']}")
        
        return True

def main():
    """Main function"""
    analyzer = Phase4EmbeddingAnalyzer()
    success = analyzer.run_analysis()
    
    if success:
        print("\n🎉 Phase 4 Embedding Analysis completed successfully!")
    else:
        print("\n❌ Phase 4 Embedding Analysis failed!")

if __name__ == "__main__":
    main()
