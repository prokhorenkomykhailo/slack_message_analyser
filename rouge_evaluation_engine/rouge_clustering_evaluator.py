#!/usr/bin/env python3
"""
ROUGE-based Clustering Evaluator for Phase 3 Topic Clustering
Evaluates AI-generated clusters against human reference clusters using ROUGE metrics
"""

import json
import os
import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from sklearn.metrics import homogeneity_score, completeness_score
import pandas as pd

class RougeClusteringEvaluator:
    """Evaluates clustering results using ROUGE metrics and clustering indices"""
    
    def __init__(self, reference_path: str, output_dir: str = "rouge_results"):
        """
        Initialize the evaluator
        
        Args:
            reference_path: Path to reference clusters JSON file
            output_dir: Directory to save evaluation results
        """
        self.reference_path = reference_path
        self.output_dir = output_dir
        self.reference_clusters = self.load_reference_clusters()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"✅ Loaded {len(self.reference_clusters)} reference clusters")
        print(f"📁 Results will be saved to: {output_dir}")
    
    def load_reference_clusters(self) -> List[Dict]:
        """Load reference clusters from JSON file"""
        try:
            with open(self.reference_path, 'r', encoding='utf-8') as f:
                clusters = json.load(f)
            return clusters
        except Exception as e:
            print(f"❌ Error loading reference clusters: {e}")
            return []
    
    def extract_cluster_text(self, cluster: Dict) -> str:
        """Extract meaningful text from a cluster for ROUGE comparison"""
        # Combine title and any other text fields
        text_parts = []
        
        if 'draft_title' in cluster:
            text_parts.append(cluster['draft_title'])
        
        if 'summary' in cluster:
            text_parts.append(cluster['summary'])
        
        if 'description' in cluster:
            text_parts.append(cluster['description'])
        
        # If no text fields, create a basic description from message IDs
        if not text_parts:
            message_count = len(cluster.get('message_ids', []))
            text_parts.append(f"Cluster with {message_count} messages")
        
        return " | ".join(text_parts)
    
    def calculate_rouge_scores(self, reference_text: str, predicted_text: str) -> Dict[str, float]:
        """Calculate ROUGE scores between reference and predicted text"""
        try:
            scores = self.rouge_scorer.score(reference_text, predicted_text)
            
            # Extract F1 scores (harmonic mean of precision and recall)
            rouge_scores = {}
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                rouge_scores[f'{metric}_precision'] = scores[metric].precision
                rouge_scores[f'{metric}_recall'] = scores[metric].recall
                rouge_scores[f'{metric}_fmeasure'] = scores[metric].fmeasure
            
            return rouge_scores
        except Exception as e:
            print(f"⚠️  Error calculating ROUGE scores: {e}")
            return {
                'rouge1_precision': 0.0, 'rouge1_recall': 0.0, 'rouge1_fmeasure': 0.0,
                'rouge2_precision': 0.0, 'rouge2_recall': 0.0, 'rouge2_fmeasure': 0.0,
                'rougeL_precision': 0.0, 'rougeL_recall': 0.0, 'rougeL_fmeasure': 0.0
            }
    
    def calculate_clustering_metrics(self, predicted_clusters: List[Dict]) -> Dict[str, Any]:
        """Calculate clustering structure metrics"""
        if not self.reference_clusters or not predicted_clusters:
            return {}
        
        # Create label arrays for clustering metrics
        all_message_ids = set()
        for cluster in self.reference_clusters:
            all_message_ids.update(cluster.get('message_ids', []))
        for cluster in predicted_clusters:
            all_message_ids.update(cluster.get('message_ids', []))
        
        all_message_ids = sorted(all_message_ids)
        
        # Create label mappings
        ref_labels = {}
        for i, cluster in enumerate(self.reference_clusters):
            for msg_id in cluster.get('message_ids', []):
                ref_labels[msg_id] = i
        
        pred_labels = {}
        for i, cluster in enumerate(predicted_clusters):
            for msg_id in cluster.get('message_ids', []):
                pred_labels[msg_id] = i
        
        # Create label arrays
        y_true = [ref_labels.get(msg_id, -1) for msg_id in all_message_ids]
        y_pred = [pred_labels.get(msg_id, -1) for msg_id in all_message_ids]
        
        # Calculate clustering metrics
        try:
            ari = adjusted_rand_score(y_true, y_pred)
            nmi = normalized_mutual_info_score(y_true, y_pred)
            v_measure = v_measure_score(y_true, y_pred)
            homogeneity = homogeneity_score(y_true, y_pred)
            completeness = completeness_score(y_true, y_pred)
        except Exception as e:
            print(f"⚠️  Error calculating clustering metrics: {e}")
            ari = nmi = v_measure = homogeneity = completeness = 0.0
        
        return {
            'adjusted_rand_index': ari,
            'normalized_mutual_info': nmi,
            'v_measure': v_measure,
            'homogeneity': homogeneity,
            'completeness': completeness,
            'reference_clusters': len(self.reference_clusters),
            'predicted_clusters': len(predicted_clusters),
            'total_messages': len(all_message_ids)
        }
    
    def evaluate_clusters(self, predicted_clusters: List[Dict], model_name: str) -> Dict[str, Any]:
        """Evaluate predicted clusters against reference clusters"""
        print(f"\n🔍 Evaluating {model_name}...")
        print(f"   Reference clusters: {len(self.reference_clusters)}")
        print(f"   Predicted clusters: {len(predicted_clusters)}")
        
        # Calculate clustering structure metrics
        clustering_metrics = self.calculate_clustering_metrics(predicted_clusters)
        
        # Calculate ROUGE scores for each cluster
        rouge_results = []
        cluster_matches = []
        
        for pred_idx, pred_cluster in enumerate(predicted_clusters):
            pred_text = self.extract_cluster_text(pred_cluster)
            best_rouge_score = 0.0
            best_ref_cluster = None
            best_ref_idx = -1
            
            # Find best matching reference cluster
            for ref_idx, ref_cluster in enumerate(self.reference_clusters):
                ref_text = self.extract_cluster_text(ref_cluster)
                rouge_scores = self.calculate_rouge_scores(ref_text, pred_text)
                
                # Use ROUGE-L F1 as the primary similarity score
                rouge_l_f1 = rouge_scores['rougeL_fmeasure']
                
                if rouge_l_f1 > best_rouge_score:
                    best_rouge_score = rouge_l_f1
                    best_ref_cluster = ref_cluster
                    best_ref_idx = ref_idx
            
            if best_ref_cluster:
                # Calculate message-level overlap
                pred_messages = set(pred_cluster.get('message_ids', []))
                ref_messages = set(best_ref_cluster.get('message_ids', []))
                intersection = len(pred_messages & ref_messages)
                
                cluster_match = {
                    'predicted_cluster': pred_cluster.get('draft_title', f'Cluster_{pred_idx}'),
                    'reference_cluster': best_ref_cluster.get('draft_title', f'Ref_{best_ref_idx}'),
                    'rouge_l_f1': best_rouge_score,
                    'message_overlap': intersection,
                    'predicted_messages': len(pred_messages),
                    'reference_messages': len(ref_messages),
                    'overlap_ratio': intersection / len(pred_messages) if pred_messages else 0.0
                }
                
                cluster_matches.append(cluster_match)
                
                # Store ROUGE scores for the best match
                ref_text = self.extract_cluster_text(best_ref_cluster)
                rouge_scores = self.calculate_rouge_scores(ref_text, pred_text)
                rouge_scores['cluster_title'] = pred_cluster.get('draft_title', f'Cluster_{pred_idx}')
                rouge_scores['reference_title'] = best_ref_cluster.get('draft_title', f'Ref_{best_ref_idx}')
                rouge_results.append(rouge_scores)
        
        # Calculate aggregate ROUGE scores
        if rouge_results:
            aggregate_rouge = {}
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                for score_type in ['precision', 'recall', 'fmeasure']:
                    key = f'{metric}_{score_type}'
                    values = [r[key] for r in rouge_results if key in r]
                    aggregate_rouge[f'avg_{key}'] = np.mean(values) if values else 0.0
                    aggregate_rouge[f'std_{key}'] = np.std(values) if values else 0.0
        else:
            aggregate_rouge = {}
        
        # Calculate overall message overlap metrics
        total_pred_messages = sum(len(c.get('message_ids', [])) for c in predicted_clusters)
        total_ref_messages = sum(len(c.get('message_ids', [])) for c in self.reference_clusters)
        total_overlap = sum(match['message_overlap'] for match in cluster_matches)
        
        message_metrics = {
            'total_predicted_messages': total_pred_messages,
            'total_reference_messages': total_ref_messages,
            'total_overlap': total_overlap,
            'overall_precision': total_overlap / total_pred_messages if total_pred_messages > 0 else 0.0,
            'overall_recall': total_overlap / total_ref_messages if total_ref_messages > 0 else 0.0
        }
        
        # Calculate overall F1
        precision = message_metrics['overall_precision']
        recall = message_metrics['overall_recall']
        message_metrics['overall_f1'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Compile final results
        results = {
            'model_name': model_name,
            'clustering_metrics': clustering_metrics,
            'rouge_metrics': aggregate_rouge,
            'message_metrics': message_metrics,
            'cluster_matches': cluster_matches,
            'detailed_rouge_scores': rouge_results
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], model_name: str):
        """Save evaluation results to files"""
        # Save detailed results
        results_file = os.path.join(self.output_dir, f"{model_name}_rouge_evaluation.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save summary CSV
        summary_data = []
        for match in results['cluster_matches']:
            summary_data.append({
                'predicted_cluster': match['predicted_cluster'],
                'reference_cluster': match['reference_cluster'],
                'rouge_l_f1': match['rouge_l_f1'],
                'message_overlap': match['message_overlap'],
                'overlap_ratio': match['overlap_ratio']
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_file = os.path.join(self.output_dir, f"{model_name}_cluster_matches.csv")
            df.to_csv(csv_file, index=False)
        
        print(f"💾 Results saved to: {results_file}")
        if summary_data:
            print(f"📊 Summary CSV saved to: {csv_file}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary"""
        print(f"\n📊 EVALUATION SUMMARY FOR {results['model_name']}")
        print("=" * 60)
        
        # Clustering metrics
        clustering = results['clustering_metrics']
        print(f"🏗️  CLUSTERING STRUCTURE:")
        print(f"   Reference clusters: {clustering['reference_clusters']}")
        print(f"   Predicted clusters: {clustering['predicted_clusters']}")
        print(f"   Adjusted Rand Index: {clustering['adjusted_rand_index']:.4f}")
        print(f"   V-Measure: {clustering['v_measure']:.4f}")
        print(f"   Homogeneity: {clustering['homogeneity']:.4f}")
        print(f"   Completeness: {clustering['completeness']:.4f}")
        
        # Message metrics
        message = results['message_metrics']
        print(f"\n📨 MESSAGE-LEVEL METRICS:")
        print(f"   Overall Precision: {message['overall_precision']:.4f}")
        print(f"   Overall Recall: {message['overall_recall']:.4f}")
        print(f"   Overall F1: {message['overall_f1']:.4f}")
        print(f"   Message Overlap: {message['total_overlap']}/{message['total_predicted_messages']}")
        
        # ROUGE metrics
        rouge = results['rouge_metrics']
        print(f"\n🔍 ROUGE TEXT SIMILARITY:")
        print(f"   ROUGE-1 F1: {rouge.get('avg_rouge1_fmeasure', 0):.4f}")
        print(f"   ROUGE-2 F1: {rouge.get('avg_rouge2_fmeasure', 0):.4f}")
        print(f"   ROUGE-L F1: {rouge.get('avg_rougeL_fmeasure', 0):.4f}")
        
        # Top cluster matches
        print(f"\n🏆 TOP CLUSTER MATCHES:")
        top_matches = sorted(results['cluster_matches'], key=lambda x: x['rouge_l_f1'], reverse=True)[:5]
        for i, match in enumerate(top_matches, 1):
            print(f"   {i}. {match['predicted_cluster'][:40]}...")
            print(f"      → {match['reference_cluster'][:40]}...")
            print(f"      ROUGE-L F1: {match['rouge_l_f1']:.4f}, Overlap: {match['message_overlap']}/{match['predicted_messages']}")

def main():
    """Example usage"""
    print("🎯 ROUGE-based Clustering Evaluator")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = RougeClusteringEvaluator(
        reference_path="../phases/phase3_clusters.json",
        output_dir="rouge_results"
    )
    
    # Example evaluation (you would load your actual predicted clusters)
    print("\n📝 To evaluate a model, use:")
    print("evaluator.evaluate_clusters(predicted_clusters, 'model_name')")
    print("evaluator.save_results(results, 'model_name')")
    print("evaluator.print_summary(results)")

if __name__ == "__main__":
    main()
