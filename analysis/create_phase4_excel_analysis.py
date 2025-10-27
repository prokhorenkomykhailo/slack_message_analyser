#!/usr/bin/env python3
"""
Create Phase 4 Excel Analysis
Generates comprehensive Excel analysis for Phase 4 results similar to Phase 3 format
"""

import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class Phase4ExcelAnalyzer:
    """Creates comprehensive Excel analysis for Phase 4 results"""
    
    def __init__(self):
        self.output_dir = "output/phase4_comprehensive_evaluation"
        self.excel_output_dir = "output/phase4_excel_analysis"
        os.makedirs(self.excel_output_dir, exist_ok=True)
        
        # Load comprehensive evaluation results
        self.evaluation_results = self.load_evaluation_results()
        print(f"✅ Loaded {len(self.evaluation_results)} evaluation results")
    
    def load_evaluation_results(self) -> List[Dict]:
        """Load Phase 4 comprehensive evaluation results"""
        results_file = os.path.join(self.output_dir, "detailed_evaluation_results.json")
        
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
    
    def create_comprehensive_analysis(self):
        """Create comprehensive analysis similar to Phase 3 format"""
        
        if not self.evaluation_results:
            print("❌ No evaluation results available")
            return
        
        # Filter successful results
        successful_results = [r for r in self.evaluation_results if r.get('success', False)]
        
        if not successful_results:
            print("❌ No successful model results found")
            return
        
        print(f"📊 Creating comprehensive analysis for {len(successful_results)} successful models...")
        
        # Create comprehensive DataFrame
        analysis_data = []
        
        for result in successful_results:
            # Basic model information
            base_row = {
                'MODEL': result['model'],
                'PROVIDER': result['provider'],
                'SUCCESS': result['success'],
                'DURATION_SECONDS': result['duration'],
                'TIMESTAMP': result['timestamp'],
                
                # Cluster metrics
                'NUM_REFINED_CLUSTERS': result['num_refined_clusters'],
                'NUM_STEP1_CLUSTERS': result['num_step1_clusters'],
                'CLUSTER_COUNT_REDUCTION': result['cluster_count_reduction'],
                'AVG_CLUSTER_SIZE': result['avg_cluster_size'],
                'AVG_STEP1_SIZE': result['avg_step1_size'],
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
                'AVG_SUMMARY_SIMILARITY': result['avg_summary_similarity'],
                'AVG_COMBINED_SIMILARITY': result['avg_combined_similarity'],
                'HIGH_QUALITY_MATCHES': result['high_quality_matches'],
                'QUALITY_RATE': result['quality_rate'],
                
                # Cluster titles (comma-separated)
                'CLUSTER_TITLES': '; '.join(result['cluster_titles']) if result['cluster_titles'] else ''
            }
            
            analysis_data.append(base_row)
        
        # Create DataFrame
        df = pd.DataFrame(analysis_data)
        
        # Calculate performance scores
        df['SIMILARITY_SCORE'] = df['AVG_COMBINED_SIMILARITY'] * 100
        df['QUALITY_SCORE'] = df['QUALITY_RATE'] * 100
        df['OPERATION_EFFICIENCY'] = (100 - (df['OPERATIONS_PER_CLUSTER'] * 10)).clip(0, 100)
        df['SIZE_OPTIMIZATION'] = (100 - abs(df['SIZE_REDUCTION'])).clip(0, 100)
        
        # Combined performance score (weighted)
        df['COMBINED_SCORE'] = (
            df['SIMILARITY_SCORE'] * 0.35 +        # 35% - similarity to benchmark
            df['QUALITY_SCORE'] * 0.25 +           # 25% - quality rate
            df['SIZE_OPTIMIZATION'] * 0.20 +       # 20% - size optimization
            df['OPERATION_EFFICIENCY'] * 0.20      # 20% - operation efficiency
        )
        
        # Sort by combined score
        df_sorted = df.sort_values('COMBINED_SCORE', ascending=False)
        
        # Save comprehensive analysis
        comprehensive_file = os.path.join(self.excel_output_dir, "phase4_comprehensive_analysis.xlsx")
        
        with pd.ExcelWriter(comprehensive_file, engine='openpyxl') as writer:
            # Main analysis sheet
            df_sorted.to_excel(writer, sheet_name='Comprehensive Analysis', index=False)
            
            # Top performers sheet
            top_10 = df_sorted.head(10)
            top_10.to_excel(writer, sheet_name='Top 10 Models', index=False)
            
            # Provider comparison sheet
            provider_stats = df_sorted.groupby('PROVIDER').agg({
                'COMBINED_SCORE': ['mean', 'max', 'min', 'count'],
                'SIMILARITY_SCORE': 'mean',
                'QUALITY_SCORE': 'mean',
                'DURATION_SECONDS': 'mean',
                'TOTAL_OPERATIONS': 'mean'
            }).round(2)
            provider_stats.to_excel(writer, sheet_name='Provider Comparison')
            
            # Performance metrics sheet
            metrics_data = []
            for _, row in df_sorted.iterrows():
                metrics_data.append({
                    'MODEL': row['MODEL'],
                    'PROVIDER': row['PROVIDER'],
                    'COMBINED_SCORE': row['COMBINED_SCORE'],
                    'SIMILARITY_SCORE': row['SIMILARITY_SCORE'],
                    'QUALITY_SCORE': row['QUALITY_SCORE'],
                    'OPERATION_EFFICIENCY': row['OPERATION_EFFICIENCY'],
                    'SIZE_OPTIMIZATION': row['SIZE_OPTIMIZATION'],
                    'DURATION_SECONDS': row['DURATION_SECONDS'],
                    'TOTAL_OPERATIONS': row['TOTAL_OPERATIONS'],
                    'NUM_REFINED_CLUSTERS': row['NUM_REFINED_CLUSTERS']
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
        
        print(f"📊 Comprehensive analysis saved to: {comprehensive_file}")
        
        # Generate summary report
        self.generate_summary_report(df_sorted)
        
        return df_sorted
    
    def generate_summary_report(self, df: pd.DataFrame):
        """Generate summary report with key insights"""
        
        print(f"\n📋 PHASE 4 ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Overall statistics
        print(f"📊 Total Models Analyzed: {len(df)}")
        print(f"🏆 Best Combined Score: {df.iloc[0]['COMBINED_SCORE']:.2f} ({df.iloc[0]['MODEL']})")
        print(f"📈 Average Combined Score: {df['COMBINED_SCORE'].mean():.2f}")
        print(f"⚡ Fastest Model: {df.loc[df['DURATION_SECONDS'].idxmin(), 'MODEL']} ({df['DURATION_SECONDS'].min():.2f}s)")
        print(f"🔧 Most Operations: {df.loc[df['TOTAL_OPERATIONS'].idxmax(), 'MODEL']} ({df['TOTAL_OPERATIONS'].max()})")
        
        # Provider analysis
        print(f"\n🏢 PROVIDER PERFORMANCE:")
        provider_performance = df.groupby('PROVIDER').agg({
            'COMBINED_SCORE': ['mean', 'count'],
            'SIMILARITY_SCORE': 'mean',
            'QUALITY_SCORE': 'mean'
        }).round(2)
        
        for provider in provider_performance.index:
            mean_score = provider_performance.loc[provider, ('COMBINED_SCORE', 'mean')]
            count = provider_performance.loc[provider, ('COMBINED_SCORE', 'count')]
            print(f"  {provider.upper()}: {mean_score:.2f} avg score ({count} models)")
        
        # Top 10 models
        print(f"\n🏆 TOP 10 MODELS:")
        print("=" * 80)
        for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['MODEL']}")
            print(f"    Combined Score: {row['COMBINED_SCORE']:.2f}")
            print(f"    Similarity: {row['SIMILARITY_SCORE']:.2f}")
            print(f"    Quality: {row['QUALITY_SCORE']:.2f}")
            print(f"    Operations: {row['TOTAL_OPERATIONS']}")
            print(f"    Duration: {row['DURATION_SECONDS']:.2f}s")
            print()
        
        # Save summary to file
        summary_file = os.path.join(self.excel_output_dir, "analysis_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("PHASE 4 ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Models Analyzed: {len(df)}\n")
            f.write(f"Best Combined Score: {df.iloc[0]['COMBINED_SCORE']:.2f} ({df.iloc[0]['MODEL']})\n")
            f.write(f"Average Combined Score: {df['COMBINED_SCORE'].mean():.2f}\n\n")
            
            f.write("TOP 10 MODELS:\n")
            f.write("=" * 80 + "\n")
            for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
                f.write(f"{i:2d}. {row['MODEL']} - Score: {row['COMBINED_SCORE']:.2f}\n")
        
        print(f"📄 Summary report saved to: {summary_file}")
    
    def create_client_format_analysis(self):
        """Create client-friendly analysis format"""
        
        if not self.evaluation_results:
            print("❌ No evaluation results available")
            return
        
        successful_results = [r for r in self.evaluation_results if r.get('success', False)]
        
        # Create client-friendly format
        client_data = []
        
        for result in successful_results:
            client_row = {
                'MODEL_NAME': result['model'],
                'PROVIDER': result['provider'],
                'EXECUTION_TIME_SECONDS': result['duration'],
                'SUCCESS_STATUS': 'SUCCESS' if result['success'] else 'FAILED',
                
                # Refinement metrics
                'INITIAL_CLUSTERS': result['num_step1_clusters'],
                'FINAL_CLUSTERS': result['num_refined_clusters'],
                'CLUSTERS_REDUCED': result['cluster_count_reduction'],
                'AVERAGE_CLUSTER_SIZE': result['avg_cluster_size'],
                
                # Operations performed
                'TOTAL_OPERATIONS': result['total_operations'],
                'MERGE_OPERATIONS': result['merge_operations'],
                'SPLIT_OPERATIONS': result['split_operations'],
                'REFINE_OPERATIONS': result['refine_operations'],
                
                # Quality metrics
                'BENCHMARK_SIMILARITY': result['avg_combined_similarity'],
                'HIGH_QUALITY_MATCHES': result['high_quality_matches'],
                'QUALITY_PERCENTAGE': result['quality_rate'],
                
                # Performance score
                'OVERALL_SCORE': 0  # Will be calculated
            }
            
            client_data.append(client_row)
        
        # Create DataFrame and calculate scores
        client_df = pd.DataFrame(client_data)
        
        # Calculate overall score (client-friendly)
        client_df['OVERALL_SCORE'] = (
            client_df['BENCHMARK_SIMILARITY'] * 100 * 0.4 +
            client_df['QUALITY_PERCENTAGE'] * 100 * 0.3 +
            (100 - abs(client_df['CLUSTERS_REDUCED'])) * 0.2 +
            (100 - (client_df['TOTAL_OPERATIONS'] / client_df['FINAL_CLUSTERS'] * 10)).clip(0, 100) * 0.1
        )
        
        # Sort by overall score
        client_df_sorted = client_df.sort_values('OVERALL_SCORE', ascending=False)
        
        # Save client format
        client_file = os.path.join(self.excel_output_dir, "phase4_client_analysis.xlsx")
        
        with pd.ExcelWriter(client_file, engine='openpyxl') as writer:
            client_df_sorted.to_excel(writer, sheet_name='Model Performance', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Total Models Analyzed',
                    'Best Performing Model',
                    'Average Overall Score',
                    'Fastest Execution Time',
                    'Most Operations Performed'
                ],
                'Value': [
                    len(client_df_sorted),
                    client_df_sorted.iloc[0]['MODEL_NAME'],
                    f"{client_df_sorted['OVERALL_SCORE'].mean():.2f}",
                    f"{client_df_sorted['EXECUTION_TIME_SECONDS'].min():.2f} seconds",
                    f"{client_df_sorted['TOTAL_OPERATIONS'].max()}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"📊 Client analysis saved to: {client_file}")
        
        return client_df_sorted
    
    def run_full_analysis(self):
        """Run complete Phase 4 Excel analysis"""
        
        print("🚀 Starting Phase 4 Excel Analysis")
        print("=" * 60)
        
        # Create comprehensive analysis
        comprehensive_df = self.create_comprehensive_analysis()
        
        # Create client format
        client_df = self.create_client_format_analysis()
        
        print(f"\n✅ Phase 4 Excel analysis complete!")
        print(f"📁 Results saved to: {self.excel_output_dir}")
        
        return comprehensive_df, client_df

def main():
    """Main function"""
    
    analyzer = Phase4ExcelAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
