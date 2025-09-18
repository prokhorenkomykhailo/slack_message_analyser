#!/usr/bin/env python3
"""
Script to update gemini_1.5_flash_new_format.xlsx with the latest data from phase3_clusters.json
"""

import json
import pandas as pd
from pathlib import Path

def update_excel_from_clusters():
    # Read the updated clusters data
    clusters_file = Path("phases/phase3_clusters.json")
    excel_file = Path("gemini_1.5_flash_new_format.xlsx")
    
    # Load clusters data
    with open(clusters_file, 'r') as f:
        clusters_data = json.load(f)
    
    # Create a list to store all message data
    all_messages = []
    
    # Process each cluster
    for cluster in clusters_data:
        cluster_id = cluster["cluster_id"]
        draft_title = cluster["draft_title"]
        participants = ", ".join(cluster["participants"])
        
        # Add each message ID with cluster information
        for msg_id in cluster["message_ids"]:
            all_messages.append({
                "message_id": msg_id,
                "cluster_id": cluster_id,
                "draft_title": draft_title,
                "participants": participants
            })
    
    # Sort by message_id for better organization
    all_messages.sort(key=lambda x: x["message_id"])
    
    # Create DataFrame
    df = pd.DataFrame(all_messages)
    
    # Try to read existing Excel file to preserve other sheets/data
    try:
        # Read existing Excel file
        existing_df = pd.read_excel(excel_file)
        print(f"Found existing Excel file with {len(existing_df)} rows")
        
        # Check if we need to update or create new structure
        if set(df.columns).issubset(set(existing_df.columns)):
            print("Updating existing structure...")
            # Update the existing data
            with pd.ExcelWriter(excel_file, mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name='clusters_data', index=False)
        else:
            print("Creating new structure...")
            # Create new Excel file with our data
            df.to_excel(excel_file, sheet_name='clusters_data', index=False)
            
    except FileNotFoundError:
        print("Excel file not found, creating new one...")
        # Create new Excel file
        df.to_excel(excel_file, sheet_name='clusters_data', index=False)
    except Exception as e:
        print(f"Error reading existing Excel file: {e}")
        print("Creating new Excel file...")
        df.to_excel(excel_file, sheet_name='clusters_data', index=False)
    
    print(f"Successfully updated {excel_file}")
    print(f"Total messages processed: {len(all_messages)}")
    print(f"Clusters: {len(clusters_data)}")
    
    # Print summary by cluster
    print("\nCluster summary:")
    for cluster in clusters_data:
        print(f"  {cluster['cluster_id']}: {len(cluster['message_ids'])} messages")

if __name__ == "__main__":
    update_excel_from_clusters()
