#!/usr/bin/env python3
"""
Create Individual Model Analysis Files for Phase 4
Generates separate Excel analysis files for each model, similar to Step 1 format
"""

import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class IndividualModelAnalyzer:
    """Creates individual Excel analysis files for each model"""
    
    def __init__(self):
        self.output_dir = "output/individual_model_analysis"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load comprehensive evaluation results
        self.evaluation_results = self.load_evaluation_results()
        print(f"✅ Loaded {len(self.evaluation_results)} evaluation results")
    
    def load_evaluation_results(self) -> List[Dict]:
        """Load Phase 4 comprehensive evaluation results"""
        results_file = "output/phase4_comprehensive_evaluation/detailed_evaluation_results.json"
        
        if not os.path.exists(results_file):
            print(f"❌ Evaluation results not found: {results_file}")
            print("Please run phase4_comprehensive_evaluation.py first")
            return []
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            return results
        except Exception as e:
            print(f"❌ Error loading evaluation results: {e}")
            return []
    
    def create_model_analysis(self, result: Dict[str, Any]) -> pd.DataFrame:
        """Create detailed analysis for a single model"""
        
        if not result.get('success', False):
            return pd.DataFrame()
        
        # Create comprehensive analysis data
        analysis_data = []
        
        # Basic model information
        model_info = {
            'MODEL_NAME': result['model'],
            'PROVIDER': result['provider'],
            'EXECUTION_TIME_SECONDS': result['duration'],
            'SUCCESS_STATUS': 'SUCCESS',
            'EVALUATION_TIMESTAMP': result.get('timestamp', ''),
            
            # Cluster metrics
            'NUM_REFINED_CLUSTERS': result['num_refined_clusters'],
            'NUM_STEP1_CLUSTERS': result['num_step1_clusters'],
            'CLUSTER_COUNT_REDUCTION': result['cluster_count_reduction'],
            'AVERAGE_CLUSTER_SIZE': result['avg_cluster_size'],
            'AVERAGE_STEP1_SIZE': result['avg_step1_size'],
            'SIZE_REDUCTION': result['size_reduction'],
            'SIZE_STD_REDUCTION': result['size_std_reduction'],
            'TOTAL_PARTICIPANTS': result['total_participants'],
            
            # Operations analysis
            'TOTAL_OPERATIONS': result['total_operations'],
            'MERGE_OPERATIONS': result['merge_operations'],
            'SPLIT_OPERATIONS': result['split_operations'],
            'REFINE_OPERATIONS': result['refine_operations'],
            'OPERATIONS_PER_CLUSTER': result['total_operations'] / result['num_refined_clusters'] if result['num_refined_clusters'] > 0 else 0,
            
            # Benchmark evaluation
            'AVG_TITLE_SIMILARITY': result['avg_title_similarity'],
            'AVG_COMBINED_SIMILARITY': result['avg_combined_similarity'],
            'HIGH_QUALITY_MATCHES': result['high_quality_matches'],
            'QUALITY_RATE': result['quality_rate'],
            
            # Performance scores
            'SIMILARITY_SCORE': result['avg_combined_similarity'] * 100,
            'QUALITY_SCORE': result['quality_rate'] * 100,
            'OPERATION_EFFICIENCY': max(0, min(100, 100 - (result['total_operations'] / result['num_refined_clusters'] * 10))) if result['num_refined_clusters'] > 0 else 100,
            'SIZE_OPTIMIZATION': max(0, min(100, 100 - abs(result['size_reduction']))),
        }
        
        # Calculate combined score
        model_info['COMBINED_SCORE'] = (
            model_info['SIMILARITY_SCORE'] * 0.35 +        # 35% - similarity to benchmark
            model_info['QUALITY_SCORE'] * 0.25 +           # 25% - quality rate
            model_info['SIZE_OPTIMIZATION'] * 0.20 +       # 20% - size optimization
            model_info['OPERATION_EFFICIENCY'] * 0.20      # 20% - operation efficiency
        )
        
        analysis_data.append(model_info)
        
        # Create DataFrame
        df = pd.DataFrame(analysis_data)
        
        return df
    
    def create_detailed_cluster_analysis(self, result: Dict[str, Any]) -> pd.DataFrame:
        """Create detailed cluster-by-cluster analysis"""
        
        if not result.get('success', False):
            return pd.DataFrame()
        
        cluster_data = []
        benchmark_eval = result.get('benchmark_evaluation', {})
        best_matches = benchmark_eval.get('best_matches', [])
        
        # Get cluster titles
        cluster_titles = result.get('cluster_titles', [])
        
        for i, cluster_title in enumerate(cluster_titles):
            # Find corresponding benchmark match
            match_info = best_matches[i] if i < len(best_matches) else {}
            
            cluster_row = {
                'CLUSTER_INDEX': i + 1,
                'CLUSTER_TITLE': cluster_title,
                'BENCHMARK_MATCH': match_info.get('best_benchmark_match', ''),
                'SIMILARITY_SCORE': match_info.get('title_similarity', 0) * 100,
                'IS_HIGH_QUALITY': 'YES' if match_info.get('combined_similarity', 0) > 0.7 else 'NO',
                'COMBINED_SIMILARITY': match_info.get('combined_similarity', 0) * 100
            }
            
            cluster_data.append(cluster_row)
        
        return pd.DataFrame(cluster_data)
    
    def create_benchmark_comparison(self, result: Dict[str, Any]) -> pd.DataFrame:
        """Create benchmark comparison analysis"""
        
        if not result.get('success', False):
            return pd.DataFrame()
        
        # Load benchmark topics
        try:
            with open('phases/phase4_clusters_refined.json', 'r') as f:
                benchmark_topics = json.load(f)
        except Exception as e:
            print(f"❌ Error loading benchmark: {e}")
            return pd.DataFrame()
        
        comparison_data = []
        
        for i, benchmark_topic in enumerate(benchmark_topics):
            benchmark_row = {
                'BENCHMARK_INDEX': i + 1,
                'BENCHMARK_TITLE': benchmark_topic.get('draft_title', ''),
                'BENCHMARK_MESSAGE_COUNT': len(benchmark_topic.get('message_ids', [])),
                'BENCHMARK_PARTICIPANTS': len(benchmark_topic.get('participants', [])),
                'BENCHMARK_MESSAGE_IDS': ', '.join(map(str, benchmark_topic.get('message_ids', []))),
                'BENCHMARK_PARTICIPANT_NAMES': ', '.join(benchmark_topic.get('participants', []))
            }
            
            comparison_data.append(benchmark_row)
        
        return pd.DataFrame(comparison_data)
    
    def create_performance_metrics(self, result: Dict[str, Any]) -> pd.DataFrame:
        """Create performance metrics analysis"""
        
        if not result.get('success', False):
            return pd.DataFrame()
        
        metrics_data = [{
            'METRIC_NAME': 'Overall Combined Score',
            'METRIC_VALUE': result['avg_combined_similarity'] * 100,
            'METRIC_UNIT': 'Percentage',
            'DESCRIPTION': 'Overall performance score combining similarity, quality, and efficiency'
        }, {
            'METRIC_NAME': 'Benchmark Similarity',
            'METRIC_VALUE': result['avg_title_similarity'] * 100,
            'METRIC_UNIT': 'Percentage',
            'DESCRIPTION': 'Average similarity to benchmark topic titles'
        }, {
            'METRIC_NAME': 'Quality Rate',
            'METRIC_VALUE': result['quality_rate'] * 100,
            'METRIC_UNIT': 'Percentage',
            'DESCRIPTION': 'Percentage of high-quality matches (>70% similarity)'
        }, {
            'METRIC_NAME': 'Execution Time',
            'METRIC_VALUE': result['duration'],
            'METRIC_UNIT': 'Seconds',
            'DESCRIPTION': 'Time taken to complete the refinement task'
        }, {
            'METRIC_NAME': 'Clusters Generated',
            'METRIC_VALUE': result['num_refined_clusters'],
            'METRIC_UNIT': 'Count',
            'DESCRIPTION': 'Number of refined clusters produced'
        }, {
            'METRIC_NAME': 'Cluster Reduction',
            'METRIC_VALUE': result['cluster_count_reduction'],
            'METRIC_UNIT': 'Count',
            'DESCRIPTION': 'Number of clusters reduced from Step 1'
        }, {
            'METRIC_NAME': 'Operations Performed',
            'METRIC_VALUE': result['total_operations'],
            'METRIC_UNIT': 'Count',
            'DESCRIPTION': 'Total refinement operations (merge/split/refine)'
        }, {
            'METRIC_NAME': 'Size Optimization',
            'METRIC_VALUE': max(0, min(100, 100 - abs(result['size_reduction']))),
            'METRIC_UNIT': 'Percentage',
            'DESCRIPTION': 'How well cluster sizes were optimized'
        }]
        
        return pd.DataFrame(metrics_data)
    
    def create_model_excel_file(self, result: Dict[str, Any]):
        """Create individual Excel file for a single model"""
        
        if not result.get('success', False):
            print(f"⚠️ Skipping {result['model']} - evaluation failed")
            return
        
        model_name = result['model'].replace('/', '_').replace('-', '_')
        filename = f"{model_name}_phase4_analysis.xlsx"
        filepath = os.path.join(self.output_dir, filename)
        
        print(f"📊 Creating analysis for {result['model']}...")
        
        # Create all analysis sheets
        model_analysis_df = self.create_model_analysis(result)
        cluster_analysis_df = self.create_detailed_cluster_analysis(result)
        benchmark_comparison_df = self.create_benchmark_comparison(result)
        performance_metrics_df = self.create_performance_metrics(result)
        
        # Create Excel file with multiple sheets
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main analysis sheet
            model_analysis_df.to_excel(writer, sheet_name='Model Analysis', index=False)
            
            # Cluster-by-cluster analysis
            if not cluster_analysis_df.empty:
                cluster_analysis_df.to_excel(writer, sheet_name='Cluster Analysis', index=False)
            
            # Benchmark comparison
            benchmark_comparison_df.to_excel(writer, sheet_name='Benchmark Topics', index=False)
            
            # Performance metrics
            performance_metrics_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Model Name',
                    'Provider',
                    'Combined Score',
                    'Similarity Score',
                    'Quality Rate',
                    'Execution Time',
                    'Clusters Generated',
                    'Operations Performed',
                    'Evaluation Date'
                ],
                'Value': [
                    result['model'],
                    result['provider'],
                    f"{result['avg_combined_similarity'] * 100:.2f}%",
                    f"{result['avg_title_similarity'] * 100:.2f}%",
                    f"{result['quality_rate'] * 100:.1f}%",
                    f"{result['duration']:.2f} seconds",
                    result['num_refined_clusters'],
                    result['total_operations'],
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"✅ Created: {filename}")
    
    def run_individual_analysis(self):
        """Run individual analysis for all models"""
        
        print("🚀 Creating Individual Model Analysis Files")
        print("=" * 60)
        
        if not self.evaluation_results:
            print("❌ No evaluation results available")
            return
        
        successful_results = [r for r in self.evaluation_results if r.get('success', False)]
        
        if not successful_results:
            print("❌ No successful model results found")
            return
        
        print(f"📊 Creating individual analysis files for {len(successful_results)} models...")
        
        # Create individual Excel files for each model
        for result in successful_results:
            self.create_model_excel_file(result)
        
        # Create summary file with all models
        self.create_summary_file(successful_results)
        
        print(f"\n✅ Individual model analysis complete!")
        print(f"📁 Results saved to: {self.output_dir}")
        
        # List generated files
        files = os.listdir(self.output_dir)
        print(f"\n📋 Generated {len(files)} files:")
        for file in sorted(files):
            print(f"  - {file}")
    
    def create_summary_file(self, successful_results: List[Dict]):
        """Create a summary file with all models"""
        
        summary_data = []
        
        for result in successful_results:
            summary_row = {
                'MODEL_NAME': result['model'],
                'PROVIDER': result['provider'],
                'COMBINED_SCORE': result['avg_combined_similarity'] * 100,
                'SIMILARITY_SCORE': result['avg_title_similarity'] * 100,
                'QUALITY_RATE': result['quality_rate'] * 100,
                'EXECUTION_TIME': result['duration'],
                'CLUSTERS_GENERATED': result['num_refined_clusters'],
                'OPERATIONS_PERFORMED': result['total_operations'],
                'HIGH_QUALITY_MATCHES': result['high_quality_matches']
            }
            summary_data.append(summary_row)
        
        # Sort by combined score
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('COMBINED_SCORE', ascending=False)
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "all_models_summary.xlsx")
        summary_df.to_excel(summary_file, index=False)
        
        print(f"📊 Created summary file: all_models_summary.xlsx")

def main():
    """Main function"""
    
    analyzer = IndividualModelAnalyzer()
    analyzer.run_individual_analysis()

if __name__ == "__main__":
    main()
