#!/usr/bin/env python3
"""
Proper Clustering Evaluator for Phase 3 Topic Clustering
Evaluates AI-generated clusters against human reference clusters using proper clustering metrics
"""

import json
import os
import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np
from rouge_score import rouge_scorer
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from sklearn.metrics import homogeneity_score, completeness_score

class ProperClusteringEvaluator:
    """Evaluates clustering results using proper clustering quality metrics"""
    
    def __init__(self, reference_path: str, output_dir: str = "proper_clustering_results"):
        self.reference_path = reference_path
        self.output_dir = output_dir
        self.reference_clusters = self.load_reference_clusters()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"✅ Proper clustering evaluator initialized")
        print(f"📁 Results will be saved to: {output_dir}")
        print(f"📊 Reference clusters: {len(self.reference_clusters)}")
    
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
        """Extract text for comparison"""
        text_parts = []
        
        if 'draft_title' in cluster:
            text_parts.append(cluster['draft_title'])
        
        if 'summary' in cluster:
            text_parts.append(cluster['summary'])
        
        if 'description' in cluster:
            text_parts.append(cluster['description'])
        
        if not text_parts:
            message_count = len(cluster.get('message_ids', []))
            text_parts.append(f"Cluster with {message_count} messages")
        
        return " | ".join(text_parts)
    
    def calculate_message_coverage(self, predicted_clusters: List[Dict]) -> Dict[str, Any]:
        """Calculate message coverage metrics"""
        if not predicted_clusters:
            return {}
        
        # Get all message IDs from reference clusters
        ref_message_ids = set()
        for cluster in self.reference_clusters:
            ref_message_ids.update(cluster.get('message_ids', []))
        
        total_ref_messages = len(ref_message_ids)
        
        # Get all message IDs from predicted clusters
        pred_message_ids = set()
        for cluster in predicted_clusters:
            pred_message_ids.update(cluster.get('message_ids', []))
        
        total_pred_messages = len(pred_message_ids)
        
        # Calculate coverage metrics
        coverage = len(pred_message_ids & ref_message_ids) / total_ref_messages if total_ref_messages > 0 else 0
        missing_messages = total_ref_messages - len(pred_message_ids & ref_message_ids)
        extra_messages = len(pred_message_ids - ref_message_ids)
        
        return {
            'total_reference_messages': total_ref_messages,
            'total_predicted_messages': total_pred_messages,
            'covered_messages': len(pred_message_ids & ref_message_ids),
            'coverage': coverage,
            'missing_messages': missing_messages,
            'extra_messages': extra_messages,
            'coverage_penalty': 1.0 - coverage
        }
    
    def calculate_cluster_size_similarity(self, predicted_clusters: List[Dict]) -> Dict[str, Any]:
        """Calculate how similar cluster sizes are to reference clusters"""
        if not predicted_clusters or not self.reference_clusters:
            return {}
        
        # Get cluster sizes
        ref_sizes = [len(cluster.get('message_ids', [])) for cluster in self.reference_clusters]
        pred_sizes = [len(cluster.get('message_ids', [])) for cluster in predicted_clusters]
        
        # Calculate size statistics
        ref_mean = np.mean(ref_sizes)
        ref_std = np.std(ref_sizes)
        pred_mean = np.mean(pred_sizes)
        pred_std = np.std(pred_sizes)
        
        # Size similarity score (closer sizes = higher score)
        size_similarity = 1.0 / (1.0 + abs(ref_mean - pred_mean) / max(ref_mean, 1))
        
        # Size distribution similarity
        if len(ref_sizes) > 1 and len(pred_sizes) > 1:
            # Use coefficient of variation for distribution similarity
            ref_cv = ref_std / ref_mean if ref_mean > 0 else 0
            pred_cv = pred_std / pred_mean if pred_mean > 0 else 0
            distribution_similarity = 1.0 / (1.0 + abs(ref_cv - pred_cv))
        else:
            distribution_similarity = 0.0
        
        return {
            'reference_mean_size': ref_mean,
            'reference_std_size': ref_std,
            'predicted_mean_size': pred_mean,
            'predicted_std_size': pred_std,
            'size_similarity': size_similarity,
            'distribution_similarity': distribution_similarity,
            'overall_size_score': (size_similarity + distribution_similarity) / 2
        }
    
    def calculate_cluster_matching(self, predicted_clusters: List[Dict]) -> Dict[str, Any]:
        """Calculate how well predicted clusters match reference clusters"""
        if not predicted_clusters or not self.reference_clusters:
            return {}
        
        matches = []
        total_rouge_score = 0.0
        total_message_overlap = 0.0
        
        for pred_cluster in predicted_clusters:
            try:
                pred_text = self.extract_cluster_text(pred_cluster)
                pred_messages = set(pred_cluster.get('message_ids', []))
                
                if not pred_messages:
                    print(f"   ⚠️  Warning: Cluster has no message IDs")
                    continue
                
                best_match = None
                best_rouge = 0.0
                best_message_overlap = 0.0
                best_combined_score = -1.0
                
                for ref_cluster in self.reference_clusters:
                    try:
                        ref_text = self.extract_cluster_text(ref_cluster)
                        ref_messages = set(ref_cluster.get('message_ids', []))
                        
                        # Calculate ROUGE similarity
                        rouge_scores = self.rouge_scorer.score(ref_text, pred_text)
                        rouge_l_f1 = rouge_scores['rougeL'].fmeasure
                        
                        # Calculate message overlap
                        message_overlap = len(pred_messages & ref_messages)
                        overlap_ratio = message_overlap / len(pred_messages) if pred_messages else 0
                        
                        # Combined score
                        combined_score = (rouge_l_f1 * 0.6) + (overlap_ratio * 0.4)
                        
                        if combined_score > best_combined_score:
                            best_match = {
                                'reference_cluster': ref_cluster.get('draft_title', 'Unknown'),
                                'rouge_l_f1': rouge_l_f1,
                                'message_overlap': message_overlap,
                                'overlap_ratio': overlap_ratio,
                                'combined_score': combined_score
                            }
                            best_rouge = rouge_l_f1
                            best_message_overlap = overlap_ratio
                            best_combined_score = combined_score
                    except Exception as e:
                        print(f"   ⚠️  Error processing reference cluster: {e}")
                        continue
                
                if best_match:
                    matches.append(best_match)
                    total_rouge_score += best_rouge
                    total_message_overlap += best_message_overlap
                else:
                    print(f"   ⚠️  Warning: No best match found for cluster")
                    
            except Exception as e:
                print(f"   ⚠️  Error processing predicted cluster: {e}")
                continue
        
        # Calculate aggregate metrics
        avg_rouge = total_rouge_score / len(matches) if matches else 0
        avg_message_overlap = total_message_overlap / len(matches) if matches else 0
        
        return {
            'cluster_matches': matches,
            'avg_rouge_l_f1': avg_rouge,
            'avg_message_overlap': avg_message_overlap,
            'matching_quality': (avg_rouge + avg_message_overlap) / 2
        }
    
    def calculate_clustering_structure_metrics(self, predicted_clusters: List[Dict]) -> Dict[str, Any]:
        """Calculate traditional clustering structure metrics"""
        if not predicted_clusters or not self.reference_clusters:
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
            'completeness': completeness
        }
    
    def comprehensive_evaluation(self, predicted_clusters: List[Dict], model_name: str) -> Dict[str, Any]:
        """Comprehensive evaluation using proper clustering metrics"""
        print(f"\n🔍 Proper clustering evaluation of {model_name}...")
        print(f"   Reference clusters: {len(self.reference_clusters)}")
        print(f"   Predicted clusters: {len(predicted_clusters)}")
        
        try:
            # 1. Message coverage (most important)
            print("   📊 Calculating message coverage...")
            coverage_metrics = self.calculate_message_coverage(predicted_clusters)
            print(f"      Coverage: {coverage_metrics.get('coverage', 0):.4f}")
            
            # 2. Cluster size similarity
            print("   📏 Calculating cluster size similarity...")
            size_metrics = self.calculate_cluster_size_similarity(predicted_clusters)
            print(f"      Size similarity: {size_metrics.get('overall_size_score', 0):.4f}")
            
            # 3. Cluster matching quality
            print("   🔍 Calculating cluster matching...")
            matching_metrics = self.calculate_cluster_matching(predicted_clusters)
            print(f"      Matching quality: {matching_metrics.get('matching_quality', 0):.4f}")
            
            # 4. Traditional clustering structure
            print("   🏗️  Calculating clustering structure...")
            structure_metrics = self.calculate_clustering_structure_metrics(predicted_clusters)
            print(f"      Structure score: {np.mean([structure_metrics.get('adjusted_rand_index', 0), structure_metrics.get('v_measure', 0), structure_metrics.get('homogeneity', 0), structure_metrics.get('completeness', 0)]):.4f}")
            
            # 5. Calculate overall score
            print("   🏆 Calculating overall score...")
            overall_score = self.calculate_overall_score(
                coverage_metrics, size_metrics, matching_metrics, structure_metrics
            )
            
            results = {
                'model_name': model_name,
                'coverage_metrics': coverage_metrics,
                'size_metrics': size_metrics,
                'matching_metrics': matching_metrics,
                'structure_metrics': structure_metrics,
                'overall_score': overall_score,
                'recommendation': self.generate_recommendation(overall_score)
            }
            
            return results
            
        except Exception as e:
            print(f"   ❌ Error in comprehensive evaluation: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def calculate_overall_score(self, coverage: Dict, size: Dict, matching: Dict, structure: Dict) -> Dict[str, float]:
        """Calculate overall clustering quality score"""
        # Weights for different aspects
        coverage_weight = 0.35      # Most important - must cluster most messages
        size_weight = 0.20          # Cluster sizes should be similar
        matching_weight = 0.25      # Clusters should match reference well
        structure_weight = 0.20     # Traditional clustering quality
        
        # Coverage score (critical)
        coverage_score = coverage.get('coverage', 0)
        
        # Size similarity score
        size_score = size.get('overall_size_score', 0)
        
        # Matching quality score
        matching_score = matching.get('matching_quality', 0)
        
        # Structure quality score (average of traditional metrics)
        structure_metrics = [
            structure.get('adjusted_rand_index', 0),
            structure.get('v_measure', 0),
            structure.get('homogeneity', 0),
            structure.get('completeness', 0)
        ]
        structure_score = np.mean(structure_metrics)
        
        # Calculate weighted overall score
        overall_score = (
            coverage_score * coverage_weight +
            size_score * size_weight +
            matching_score * matching_weight +
            structure_score * structure_weight
        )
        
        return {
            'overall_score': overall_score,
            'coverage_score': coverage_score,
            'size_score': size_score,
            'matching_score': matching_score,
            'structure_score': structure_score,
            'breakdown': {
                'coverage': coverage_score,
                'size_similarity': size_score,
                'cluster_matching': matching_score,
                'clustering_structure': structure_score
            }
        }
    
    def generate_recommendation(self, overall_score: Dict) -> str:
        """Generate recommendation based on scores"""
        overall = overall_score['overall_score']
        coverage = overall_score['coverage_score']
        
        if coverage < 0.5:
            return "POOR - Very low message coverage, clustering is incomplete"
        elif overall >= 0.8:
            return "EXCELLENT - High quality clustering with good coverage and matching"
        elif overall >= 0.6:
            return "GOOD - Solid clustering quality with room for improvement"
        elif overall >= 0.4:
            return "FAIR - Some good aspects but significant improvements needed"
        else:
            return "POOR - Major improvements required in clustering quality"
    
    def save_results(self, results: Dict[str, Any], model_name: str):
        """Save evaluation results"""
        results_file = os.path.join(self.output_dir, f"{model_name}_proper_evaluation.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Proper evaluation results saved to: {results_file}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print comprehensive evaluation summary"""
        print(f"\n📊 PROPER CLUSTERING EVALUATION SUMMARY FOR {results['model_name']}")
        print("=" * 80)
        
        overall = results['overall_score']
        print(f"🏆 OVERALL SCORE: {overall['overall_score']:.4f}")
        print(f"💡 RECOMMENDATION: {results['recommendation']}")
        
        print(f"\n📊 MESSAGE COVERAGE (35% weight):")
        coverage = results['coverage_metrics']
        print(f"   Coverage: {coverage['coverage']:.4f} ({coverage['covered_messages']}/{coverage['total_reference_messages']})")
        print(f"   Missing Messages: {coverage['missing_messages']}")
        print(f"   Extra Messages: {coverage['extra_messages']}")
        
        print(f"\n📏 CLUSTER SIZE SIMILARITY (20% weight):")
        size = results['size_metrics']
        print(f"   Reference: {size['reference_mean_size']:.1f} ± {size['reference_std_size']:.1f}")
        print(f"   Predicted: {size['predicted_mean_size']:.1f} ± {size['predicted_std_size']:.1f}")
        print(f"   Size Similarity: {size['overall_size_score']:.4f}")
        
        print(f"\n🔍 CLUSTER MATCHING QUALITY (25% weight):")
        matching = results['matching_metrics']
        print(f"   ROUGE-L F1: {matching['avg_rouge_l_f1']:.4f}")
        print(f"   Message Overlap: {matching['avg_message_overlap']:.4f}")
        print(f"   Overall Matching: {matching['matching_quality']:.4f}")
        
        print(f"\n🏗️  CLUSTERING STRUCTURE (20% weight):")
        structure = results['structure_metrics']
        print(f"   Adjusted Rand Index: {structure['adjusted_rand_index']:.4f}")
        print(f"   V-Measure: {structure['v_measure']:.4f}")
        print(f"   Homogeneity: {structure['homogeneity']:.4f}")
        print(f"   Completeness: {structure['completeness']:.4f}")
        
        print(f"\n📈 SCORE BREAKDOWN:")
        for metric, score in overall['breakdown'].items():
            print(f"   {metric.replace('_', ' ').title()}: {score:.4f}")

def main():
    """Example usage"""
    print("🎯 Proper Clustering Evaluator")
    print("=" * 60)
    print("Focuses on actual clustering quality metrics:")
    print("- Message coverage (most important)")
    print("- Cluster size similarity")
    print("- Cluster matching quality")
    print("- Traditional clustering structure")
    
    # Example usage
    print("\n📝 To evaluate a model:")
    print("evaluator = ProperClusteringEvaluator('phases/phase3_clusters.json')")
    print("results = evaluator.comprehensive_evaluation(clusters, 'model_name')")
    print("evaluator.save_results(results, 'model_name')")
    print("evaluator.print_summary(results)")

if __name__ == "__main__":
    main()
