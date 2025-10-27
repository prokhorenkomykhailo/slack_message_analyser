#!/usr/bin/env python3
"""
Extract benchmark metadata from gemini-2.5-flash output
Creates a clean benchmark file for Step 3 evaluation
"""

import json

def extract_benchmark():
    """Extract core metadata from gemini-2.5-flash output"""
    
    # Load the gemini output
    input_file = "output/phase5_metadata_generation/google_gemini-2.5-flash.json"
    output_file = "phases/phase5_metadata_benchmark.json"
    
    print("📖 Loading gemini-2.5-flash output...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract just the metadata from each cluster
    benchmark = []
    
    for result in data['metadata_results']:
        if result['success']:
            benchmark_entry = {
                "cluster_id": result['cluster_id'],
                "metadata": result['metadata']
            }
            benchmark.append(benchmark_entry)
    
    print(f"✅ Extracted {len(benchmark)} clusters")
    
    # Save as clean benchmark
    with open(output_file, 'w') as f:
        json.dump(benchmark, f, indent=2)
    
    print(f"💾 Benchmark saved to: {output_file}")
    
    # Print summary
    print("\n📊 BENCHMARK SUMMARY:")
    print(f"Total clusters: {len(benchmark)}")
    
    for entry in benchmark:
        cluster_id = entry['cluster_id']
        metadata = entry['metadata']
        title = metadata.get('title', 'N/A')
        action_items = len(metadata.get('action_items', []))
        participants = len(metadata.get('participants', []))
        tags = len(metadata.get('tags', []))
        
        print(f"\n{cluster_id}:")
        print(f"  Title: {title}")
        print(f"  Action Items: {action_items}")
        print(f"  Participants: {participants}")
        print(f"  Tags: {tags}")
        print(f"  Urgency: {metadata.get('urgency', 'N/A')}")
        print(f"  Status: {metadata.get('status', 'N/A')}")

if __name__ == "__main__":
    extract_benchmark()

