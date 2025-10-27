#!/usr/bin/env python3
"""
Analyze the updated phase3_clusters.json against the existing Gemini 1.5 Flash analysis
to identify any changes or discrepancies.
"""

import json
import pandas as pd
from pathlib import Path

def load_updated_benchmark_clusters():
    """Load the updated benchmark clusters from phase3_clusters.json"""
    clusters_file = Path("phases/phase3_clusters.json")
    
    with open(clusters_file, 'r') as f:
        clusters_data = json.load(f)
    
    # Convert to a more convenient format
    benchmark_clusters = {}
    for cluster in clusters_data:
        cluster_id = cluster["cluster_id"]
        benchmark_clusters[cluster_id] = {
            "title": cluster["draft_title"],
            "message_ids": set(cluster["message_ids"]),
            "message_count": len(cluster["message_ids"]),
            "participants": cluster["participants"]
        }
    
    return benchmark_clusters

def load_existing_analysis():
    """Load the existing Gemini 1.5 Flash analysis"""
    analysis_file = Path("llm_analysis_with_improved_scores.csv")
    
    df = pd.read_csv(analysis_file)
    # Filter for Gemini 1.5 Flash only
    gemini_data = df[df['MODEL'] == 'gemini-1.5-flash'].copy()
    
    return gemini_data

def parse_message_ids(message_ids_str):
    """Parse semicolon-separated message IDs into a set of integers"""
    if pd.isna(message_ids_str) or message_ids_str == '':
        return set()
    return set(int(id.strip()) for id in str(message_ids_str).split(';'))

def compare_clusters():
    """Compare the updated benchmark clusters with existing analysis"""
    
    print("=== ANALYZING UPDATED PHASE3_CLUSTERS.JSON ===\n")
    
    # Load data
    benchmark_clusters = load_updated_benchmark_clusters()
    existing_analysis = load_existing_analysis()
    
    print(f"Updated benchmark clusters: {len(benchmark_clusters)}")
    print(f"Existing analysis rows: {len(existing_analysis)}")
    
    print("\n=== UPDATED BENCHMARK CLUSTER DETAILS ===")
    total_benchmark_messages = 0
    for cluster_id, data in benchmark_clusters.items():
        print(f"\nCluster: {cluster_id}")
        print(f"  Title: {data['title']}")
        print(f"  Message Count: {data['message_count']}")
        print(f"  Message IDs: {sorted(list(data['message_ids']))}")
        print(f"  Participants: {data['participants']}")
        total_benchmark_messages += data['message_count']
    
    print(f"\n📊 TOTAL BENCHMARK MESSAGES: {total_benchmark_messages}")
    print(f"📊 CLUSTER BREAKDOWN:")
    print(f"   - eco_bloom_campaign: {len(benchmark_clusters['eco_bloom_campaign']['message_ids'])} messages")
    print(f"   - fitfusion_rebrand: {len(benchmark_clusters['fitfusion_rebrand']['message_ids'])} messages")  
    print(f"   - technova_launch: {len(benchmark_clusters['technova_launch']['message_ids'])} messages")
    print(f"   - greenscape_report: {len(benchmark_clusters['greenscape_report']['message_ids'])} messages")
    print(f"   - urbanedge_strategy: {len(benchmark_clusters['urbanedge_strategy']['message_ids'])} messages")
    print(f"   - q3_content_calendar: {len(benchmark_clusters['q3_content_calendar']['message_ids'])} messages")
    
    print("\n=== COMPARISON WITH EXISTING ANALYSIS ===")
    
    # Group existing analysis by benchmark cluster
    analysis_by_cluster = {}
    for _, row in existing_analysis.iterrows():
        benchmark_id = row['BENCHMARK_CLUSTER_ID']
        if benchmark_id not in analysis_by_cluster:
            analysis_by_cluster[benchmark_id] = []
        analysis_by_cluster[benchmark_id].append(row)
    
    discrepancies_found = False
    
    for cluster_id, benchmark_data in benchmark_clusters.items():
        print(f"\n--- Analyzing {cluster_id} ---")
        
        if cluster_id in analysis_by_cluster:
            analysis_rows = analysis_by_cluster[cluster_id]
            print(f"Found {len(analysis_rows)} analysis rows for this cluster")
            
            for i, row in enumerate(analysis_rows):
                print(f"\n  Analysis Row {i+1}:")
                print(f"    LLM Cluster ID: {row['CLUSTER_ID']}")
                print(f"    LLM Title: {row['CLUSTER_TITLE']}")
                print(f"    Benchmark Title: {row['BENCHMARK_TITLE']}")
                print(f"    Benchmark Message Count: {row['BENCHMARK_MESSAGE_COUNT']}")
                print(f"    LLM Message Count: {row['LLM_MESSAGE_COUNT']}")
                
                # Parse LLM message IDs
                llm_message_ids = parse_message_ids(row['MESSAGE_IDS'])
                
                # Compare with updated benchmark
                expected_count = benchmark_data['message_count']
                actual_benchmark_count = row['BENCHMARK_MESSAGE_COUNT']
                
                if expected_count != actual_benchmark_count:
                    print(f"    ⚠️  DISCREPANCY: Expected {expected_count} messages, analysis shows {actual_benchmark_count}")
                    discrepancies_found = True
                else:
                    print(f"    ✅ Message count matches: {expected_count}")
                
                # Check title match
                if benchmark_data['title'] != row['BENCHMARK_TITLE']:
                    print(f"    ⚠️  TITLE DISCREPANCY:")
                    print(f"       Expected: {benchmark_data['title']}")
                    print(f"       Analysis: {row['BENCHMARK_TITLE']}")
                    discrepancies_found = True
                else:
                    print(f"    ✅ Title matches")
                
                # Calculate actual overlap with updated benchmark
                overlap = len(benchmark_data['message_ids'].intersection(llm_message_ids))
                missing = len(benchmark_data['message_ids'] - llm_message_ids)
                extra = len(llm_message_ids - benchmark_data['message_ids'])
                
                print(f"    Actual Overlap: {overlap}/{expected_count} messages")
                print(f"    Missing: {missing} messages")
                print(f"    Extra: {extra} messages")
                
                # Compare with analysis metrics
                analysis_matched = row['MATCHED_MESSAGES']
                analysis_missing = row['MISSING_MESSAGES']
                analysis_extra = row['EXTRA_MESSAGES']
                
                if overlap != analysis_matched:
                    print(f"    ⚠️  MATCHED DISCREPANCY: Actual {overlap} vs Analysis {analysis_matched}")
                    discrepancies_found = True
                
                if missing != analysis_missing:
                    print(f"    ⚠️  MISSING DISCREPANCY: Actual {missing} vs Analysis {analysis_missing}")
                    discrepancies_found = True
                
                if extra != analysis_extra:
                    print(f"    ⚠️  EXTRA DISCREPANCY: Actual {extra} vs Analysis {analysis_extra}")
                    discrepancies_found = True
        else:
            print(f"  ⚠️  MISSING: No analysis found for cluster {cluster_id}")
            discrepancies_found = True
    
    print("\n=== SUMMARY ===")
    if discrepancies_found:
        print("⚠️  DISCREPANCIES FOUND: The analysis needs to be updated with the new benchmark data")
        print("\nRecommended actions:")
        print("1. Run the enhanced clustering analysis script with updated benchmark")
        print("2. Regenerate the Excel file with corrected data")
        print("3. Verify all metrics are recalculated correctly")
    else:
        print("✅ NO DISCREPANCIES: The existing analysis is consistent with updated benchmark")
    
    return discrepancies_found

def generate_updated_analysis():
    """Generate updated analysis based on the new benchmark data"""
    
    print("\n=== GENERATING UPDATED ANALYSIS ===")
    
    benchmark_clusters = load_updated_benchmark_clusters()
    existing_analysis = load_existing_analysis()
    
    # Create corrected analysis
    corrected_rows = []
    
    for _, row in existing_analysis.iterrows():
        benchmark_id = row['BENCHMARK_CLUSTER_ID']
        
        if benchmark_id in benchmark_clusters:
            benchmark_data = benchmark_clusters[benchmark_id]
            llm_message_ids = parse_message_ids(row['MESSAGE_IDS'])
            
            # Recalculate metrics with updated benchmark
            overlap = len(benchmark_data['message_ids'].intersection(llm_message_ids))
            missing = len(benchmark_data['message_ids'] - llm_message_ids)
            extra = len(llm_message_ids - benchmark_data['message_ids'])
            
            expected_count = benchmark_data['message_count']
            llm_count = len(llm_message_ids)
            
            # Calculate corrected percentages
            coverage_pct = (overlap / expected_count * 100) if expected_count > 0 else 0
            precision_pct = (overlap / llm_count * 100) if llm_count > 0 else 0
            recall_pct = coverage_pct  # Same as coverage
            deviation_pct = ((llm_count - expected_count) / expected_count * 100) if expected_count > 0 else 0
            
            # Create corrected row
            corrected_row = row.copy()
            corrected_row['BENCHMARK_TITLE'] = benchmark_data['title']
            corrected_row['BENCHMARK_MESSAGE_COUNT'] = expected_count
            corrected_row['MATCHED_MESSAGES'] = overlap
            corrected_row['MISSING_MESSAGES'] = missing
            corrected_row['EXTRA_MESSAGES'] = extra
            corrected_row['COVERAGE_PERCENTAGE'] = round(coverage_pct, 2)
            corrected_row['PRECISION_PERCENT'] = round(precision_pct, 2)
            corrected_row['RECALL_PERCENT'] = round(recall_pct, 2)
            corrected_row['MESSAGE_COUNT_DEVIATION_PERCENT'] = round(deviation_pct, 2)
            
            corrected_rows.append(corrected_row)
    
    # Create corrected DataFrame
    corrected_df = pd.DataFrame(corrected_rows)
    
    # Save corrected analysis
    output_file = Path("gemini_1.5_flash_corrected_analysis.csv")
    corrected_df.to_csv(output_file, index=False)
    
    print(f"✅ Corrected analysis saved to: {output_file}")
    
    # Print summary of corrections
    print("\n=== CORRECTION SUMMARY ===")
    for i, (_, row) in enumerate(corrected_df.iterrows()):
        print(f"\nTopic {row['CLUSTER_ID']}: {row['BENCHMARK_TITLE']}")
        print(f"  Coverage: {row['COVERAGE_PERCENTAGE']:.1f}%")
        print(f"  Precision: {row['PRECISION_PERCENT']:.1f}%")
        print(f"  Matched/Missing/Extra: {row['MATCHED_MESSAGES']}/{row['MISSING_MESSAGES']}/{row['EXTRA_MESSAGES']}")
        print(f"  Deviation: {row['MESSAGE_COUNT_DEVIATION_PERCENT']:.1f}%")
    
    return corrected_df

def main():
    """Main analysis function"""
    
    print("GEMINI 1.5 FLASH ANALYSIS VERIFICATION")
    print("="*50)
    
    # Compare existing analysis with updated benchmark
    discrepancies_found = compare_clusters()
    
    if discrepancies_found:
        print("\n" + "="*50)
        print("GENERATING CORRECTED ANALYSIS")
        print("="*50)
        
        corrected_df = generate_updated_analysis()
        
        print(f"\n✅ Analysis complete!")
        print(f"   - Corrected analysis saved")
        print(f"   - Ready for Excel update")
    else:
        print(f"\n✅ No corrections needed - existing analysis is accurate")

if __name__ == "__main__":
    main()
