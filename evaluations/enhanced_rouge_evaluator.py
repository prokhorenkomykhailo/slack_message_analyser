#!/usr/bin/env python3
"""
Enhanced ROUGE-based evaluation for unlimited topics and summary types
Focuses on semantic quality rather than rigid structural matching
"""

import json
import os
import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np
from rouge_score import rouge_scorer
import pandas as pd

class EnhancedRougeEvaluator:
    """Enhanced evaluator for unlimited topic scenarios"""
    
    def __init__(self, reference_path: str, output_dir: str = "enhanced_rouge_results"):
        self.reference_path = reference_path
        self.output_dir = output_dir
        self.reference_clusters = self.load_reference_clusters()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"✅ Enhanced evaluator initialized")
        print(f"📁 Results will be saved to: {output_dir}")
    
    def load_reference_clusters(self) -> List[Dict]:
        """Load reference clusters (but don't treat as absolute truth)"""
        try:
            with open(self.reference_path, 'r', encoding='utf-8') as f:
                clusters = json.load(f)
            return clusters
        except Exception as e:
            print(f"⚠️  Reference clusters not available: {e}")
            return []
    
    def evaluate_cluster_quality(self, clusters: List[Dict], total_messages: int = None) -> Dict[str, Any]:
        """Evaluate cluster quality without rigid reference matching"""
        
        if not clusters:
            return {}
        
        results = {
            'total_clusters': len(clusters),
            'topic_coherence': [],
            'topic_diversity': [],
            'summary_quality': [],
            'cluster_sizes': [],
            'semantic_richness': []
        }
        
        # Analyze each cluster
        for cluster in clusters:
            # Cluster size analysis
            message_count = len(cluster.get('message_ids', []))
            results['cluster_sizes'].append(message_count)
            
            # Summary quality (if available)
            summary = cluster.get('summary', '') or cluster.get('draft_title', '')
            if summary:
                results['summary_quality'].append(self.analyze_summary_quality(summary))
            
            # Semantic richness (message diversity within cluster)
            if message_count > 1:
                results['semantic_richness'].append(self.calculate_semantic_richness(cluster))
        
        # Calculate aggregate metrics
        results['avg_cluster_size'] = np.mean(results['cluster_sizes']) if results['cluster_sizes'] else 0
        results['cluster_size_std'] = np.std(results['cluster_sizes']) if results['cluster_sizes'] else 0
        results['avg_summary_quality'] = np.mean(results['summary_quality']) if results['summary_quality'] else 0
        results['avg_semantic_richness'] = np.mean(results['semantic_richness']) if results['semantic_richness'] else 0
        
        # Calculate coverage metrics
        total_clustered_messages = sum(results['cluster_sizes'])
        results['total_clustered_messages'] = total_clustered_messages
        
        if total_messages:
            results['coverage'] = total_clustered_messages / total_messages
            results['missing_messages'] = total_messages - total_clustered_messages
            results['coverage_penalty'] = 1.0 - results['coverage']  # Higher penalty for lower coverage
        else:
            results['coverage'] = 0.0
            results['missing_messages'] = 0
            results['coverage_penalty'] = 1.0
        
        return results
    
    def analyze_summary_quality(self, summary: str) -> float:
        """Analyze summary quality without reference comparison"""
        if not summary:
            return 0.0
        
        # Length appropriateness (not too short, not too long)
        words = summary.split()
        if len(words) < 3:
            length_score = 0.3  # Too short
        elif len(words) < 15:
            length_score = 1.0  # Good length
        elif len(words) < 25:
            length_score = 0.8  # Slightly long
        else:
            length_score = 0.5  # Too long
        
        # Clarity indicators
        clarity_indicators = [
            'clear', 'specific', 'focused', 'comprehensive', 'detailed',
            'vague', 'unclear', 'confusing', 'general', 'broad'
        ]
        
        clarity_score = 0.5  # Base score
        for indicator in clarity_indicators:
            if indicator in summary.lower():
                if indicator in ['clear', 'specific', 'focused', 'comprehensive', 'detailed']:
                    clarity_score += 0.1
                else:
                    clarity_score -= 0.1
        
        clarity_score = max(0.0, min(1.0, clarity_score))
        
        # Actionability (does it suggest next steps?)
        action_words = ['implement', 'develop', 'create', 'improve', 'optimize', 'analyze', 'review']
        action_score = 0.5
        for word in action_words:
            if word in summary.lower():
                action_score += 0.1
        
        action_score = min(1.0, action_score)
        
        # Combined score
        return (length_score + clarity_score + action_score) / 3
    
    def calculate_semantic_richness(self, cluster: Dict) -> float:
        """Calculate semantic richness of a cluster"""
        # This is a simplified version - in practice you'd use embeddings
        # For now, we'll use message count and variety as proxy
        
        message_count = len(cluster.get('message_ids', []))
        if message_count <= 1:
            return 0.0
        
        # More messages = more semantic content (up to a point)
        if message_count <= 5:
            richness = message_count / 5
        elif message_count <= 15:
            richness = 1.0
        else:
            richness = 15 / message_count  # Diminishing returns
        
        return richness
    
    def evaluate_with_reference(self, predicted_clusters: List[Dict], model_name: str) -> Dict[str, Any]:
        """Evaluate against reference clusters when available"""
        if not self.reference_clusters:
            return {}
        
        results = {}
        
        # ROUGE-based similarity (but not as primary metric)
        rouge_scores = []
        for pred_cluster in predicted_clusters:
            pred_text = self.extract_cluster_text(pred_cluster)
            best_rouge = 0.0
            
            for ref_cluster in self.reference_clusters:
                ref_text = self.extract_cluster_text(ref_cluster)
                scores = self.rouge_scorer.score(ref_text, pred_text)
                best_rouge = max(best_rouge, scores['rougeL'].fmeasure)
            
            rouge_scores.append(best_rouge)
        
        if rouge_scores:
            results['avg_rouge_l'] = np.mean(rouge_scores)
            results['max_rouge_l'] = np.max(rouge_scores)
            results['rouge_consistency'] = 1.0 - np.std(rouge_scores)  # Higher is better
        
        return results
    
    def extract_cluster_text(self, cluster: Dict) -> str:
        """Extract text for ROUGE comparison"""
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
    
    def comprehensive_evaluation(self, predicted_clusters: List[Dict], model_name: str, total_messages: int = None) -> Dict[str, Any]:
        """Comprehensive evaluation combining multiple approaches"""
        print(f"\n🔍 Enhanced evaluation of {model_name}...")
        
        # 1. Intrinsic quality evaluation (no reference needed)
        intrinsic_quality = self.evaluate_cluster_quality(predicted_clusters, total_messages)
        
        # 2. Reference-based evaluation (if available)
        reference_evaluation = self.evaluate_with_reference(predicted_clusters, model_name)
        
        # 3. Combined scoring
        combined_score = self.calculate_combined_score(intrinsic_quality, reference_evaluation)
        
        results = {
            'model_name': model_name,
            'intrinsic_quality': intrinsic_quality,
            'reference_evaluation': reference_evaluation,
            'combined_score': combined_score,
            'recommendation': self.generate_recommendation(combined_score)
        }
        
        return results
    
    def calculate_combined_score(self, intrinsic: Dict, reference: Dict) -> Dict[str, float]:
        """Calculate combined quality score"""
        # Weight intrinsic quality higher than reference matching
        intrinsic_weight = 0.6
        reference_weight = 0.2
        coverage_weight = 0.2  # New: coverage penalty
        
        # Intrinsic quality components
        intrinsic_score = (
            intrinsic.get('avg_summary_quality', 0) * 0.4 +
            intrinsic.get('avg_semantic_richness', 0) * 0.3 +
            (1.0 - intrinsic.get('cluster_size_std', 0) / max(intrinsic.get('avg_cluster_size', 1), 1)) * 0.3
        )
        
        # Coverage penalty (critical for clustering quality)
        coverage_score = intrinsic.get('coverage', 0)  # Higher is better
        
        # Reference matching (if available)
        reference_score = 0.0
        if reference:
            reference_score = (
                reference.get('avg_rouge_l', 0) * 0.6 +
                reference.get('rouge_consistency', 0) * 0.4
            )
        
        # Combined score with coverage penalty
        combined = (
            intrinsic_score * intrinsic_weight + 
            reference_score * reference_weight + 
            coverage_score * coverage_weight
        )
        
        return {
            'overall_score': combined,
            'intrinsic_score': intrinsic_score,
            'reference_score': reference_score,
            'coverage_score': coverage_score,
            'breakdown': {
                'summary_quality': intrinsic.get('avg_summary_quality', 0),
                'semantic_richness': intrinsic.get('avg_semantic_richness', 0),
                'cluster_balance': 1.0 - intrinsic.get('cluster_size_std', 0) / max(intrinsic.get('avg_cluster_size', 1), 1),
                'coverage': coverage_score,
                'rouge_similarity': reference.get('avg_rouge_l', 0),
                'rouge_consistency': reference.get('rouge_consistency', 0)
            }
        }
    
    def generate_recommendation(self, combined_score: Dict) -> str:
        """Generate recommendation based on scores"""
        overall = combined_score['overall_score']
        
        if overall >= 0.8:
            return "EXCELLENT - High quality clustering with good topic discovery"
        elif overall >= 0.6:
            return "GOOD - Solid clustering quality with room for improvement"
        elif overall >= 0.4:
            return "FAIR - Some good aspects but significant improvements needed"
        else:
            return "POOR - Major improvements required in clustering quality"
    
    def save_results(self, results: Dict[str, Any], model_name: str):
        """Save enhanced evaluation results"""
        results_file = os.path.join(self.output_dir, f"{model_name}_enhanced_evaluation.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Enhanced results saved to: {results_file}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print enhanced evaluation summary"""
        print(f"\n📊 ENHANCED EVALUATION SUMMARY FOR {results['model_name']}")
        print("=" * 70)
        
        combined = results['combined_score']
        print(f"🏆 OVERALL SCORE: {combined['overall_score']:.4f}")
        print(f"💡 RECOMMENDATION: {combined['recommendation']}")
        
        print(f"\n🔍 INTRINSIC QUALITY:")
        intrinsic = results['intrinsic_quality']
        print(f"   Summary Quality: {intrinsic.get('avg_summary_quality', 0):.4f}")
        print(f"   Semantic Richness: {intrinsic.get('avg_semantic_richness', 0):.4f}")
        print(f"   Cluster Balance: {combined['breakdown']['cluster_balance']:.4f}")
        print(f"   Average Cluster Size: {intrinsic.get('avg_cluster_size', 0):.1f}")
        print(f"   Coverage: {intrinsic.get('coverage', 0):.4f} ({intrinsic.get('total_clustered_messages', 0)}/{intrinsic.get('total_clustered_messages', 0) + intrinsic.get('missing_messages', 0)})")
        
        if results['reference_evaluation']:
            print(f"\n📚 REFERENCE COMPARISON:")
            ref = results['reference_evaluation']
            print(f"   ROUGE-L Similarity: {ref.get('avg_rouge_l', 0):.4f}")
            print(f"   ROUGE Consistency: {ref.get('rouge_consistency', 0):.4f}")
        
        print(f"\n📈 SCORE BREAKDOWN:")
        for metric, score in combined['breakdown'].items():
            print(f"   {metric.replace('_', ' ').title()}: {score:.4f}")

def main():
    """Example usage"""
    print("🎯 Enhanced ROUGE-based Clustering Evaluator")
    print("=" * 60)
    print("Focuses on semantic quality rather than rigid structural matching")
    print("Better suited for unlimited topics and summary types")
    
    # Example usage
    print("\n📝 To evaluate a model:")
    print("evaluator = EnhancedRougeEvaluator('phases/phase3_clusters.json')")
    print("results = evaluator.comprehensive_evaluation(clusters, 'model_name')")
    print("evaluator.save_results(results, 'model_name')")
    print("evaluator.print_summary(results)")

if __name__ == "__main__":
    main()
