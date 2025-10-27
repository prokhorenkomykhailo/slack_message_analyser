#!/usr/bin/env python3
"""
Calculate input token count for Gemini model in Step 1 (Phase 3 Topic Clustering)
This script reconstructs the exact prompt sent to the model and calculates token count.
"""

import csv
import os
import json
from typing import List, Dict

def load_messages_from_csv() -> List[Dict]:
    """Load messages from Synthetic_Slack_Messages.csv"""
    csv_path = "data/Synthetic_Slack_Messages.csv"
    messages = []
    
    try:
        with open(csv_path, "r", encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                message = {
                    "id": i + 1,
                    "channel": row["channel"],
                    "user": row["user_name"],
                    "user_id": row["user_id"],
                    "timestamp": row["timestamp"],
                    "text": row["text"],
                    "thread_ts": row["thread_id"] if row["thread_id"] != "None" else None
                }
                messages.append(message)
        print(f"✅ Loaded {len(messages)} messages from {csv_path}")
        return messages
    except FileNotFoundError:
        print(f"❌ {csv_path} not found")
        return []
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return []

def get_clustering_prompt_template() -> str:
    """Get the exact prompt template used in Phase 3"""
    return """You are an expert at analyzing Slack conversations and grouping messages into coherent topics.

Given a set of Slack messages, your task is to:

1. **Group messages into topic clusters** based on:
   - Shared thread relationships (same thread_id)
   - Same participants
   - Semantic similarity (same subject matter)
   - Temporal proximity
   - Channel context

2. **For each cluster, provide:**
   - cluster_id: unique identifier (e.g., "cluster_001")
   - message_ids: list of message IDs in this cluster
   - draft_title: brief descriptive title
   - participants: list of users involved
   - channel: primary channel for this topic
   - thread_id: if messages belong to a specific thread

**Messages to analyze:**
{messages}

**Instructions:**
- Group related messages together based on meaningful relationships
- Each message should belong to exactly one cluster
- Consider both explicit relationships (threads) and implicit ones (same topic)
- Merge related threads and topics into cohesive clusters
- Provide clear, descriptive titles for each cluster
- Pay attention to project names (EcoBloom, FitFusion, TechNova, GreenScape, UrbanEdge)
- Consider temporal relationships and deadlines mentioned
- Focus on creating logical, meaningful groupings rather than arbitrary cluster counts

**Output Format (JSON):**
{{
  "clusters": [
    {{
      "cluster_id": "cluster_001",
      "message_ids": [1, 2, 3, 4, 5],
      "draft_title": "EcoBloom Summer Campaign Planning",
      "participants": ["Devon", "Sam", "Leah", "Jordan"],
      "channel": "#campaign-briefs",
      "thread_id": "thread_001"
    }}
  ]
}}

Analyze the messages and provide the clustering results in the specified JSON format.
"""

def format_messages_for_prompt(messages: List[Dict], max_tokens: int = 80000) -> str:
    """Format messages exactly as done in Phase 3"""
    formatted = []
    
    for msg in messages:
        # Format each message - truncate text to 300 chars as per original code
        msg_text = (
            f"ID: {msg['id']} | Channel: {msg['channel']} | User: {msg['user']} | "
            f"Thread: {msg.get('thread_ts', 'None')} | Text: {msg['text'][:300]}..."
        )
        formatted.append(msg_text)
    
    return "\n".join(formatted)

def estimate_tokens_simple(text: str) -> int:
    """Simple token estimation (4 chars per token)"""
    return len(text) // 4

def estimate_tokens_accurate(text: str) -> int:
    """More accurate token estimation for Gemini models"""
    # Gemini uses similar tokenization to GPT models
    # Average: ~4 characters per token for English text
    # But we'll be more precise:
    # - Words: ~1.3 tokens per word
    # - Special chars and formatting: counted separately
    
    words = len(text.split())
    chars = len(text)
    
    # Weighted estimate
    word_based = words * 1.3
    char_based = chars / 4
    
    # Use average of both methods
    return int((word_based + char_based) / 2)

def main():
    """Calculate input tokens for Gemini model in Step 1"""
    
    print("=" * 80)
    print("STEP 1 (PHASE 3) INPUT TOKEN CALCULATION FOR GEMINI MODEL")
    print("=" * 80)
    print()
    
    # Load messages
    print("📂 Loading Synthetic_Slack_Messages.csv...")
    messages = load_messages_from_csv()
    
    if not messages:
        print("❌ Failed to load messages")
        return
    
    print(f"   Total messages: {len(messages)}")
    print()
    
    # Get prompt template
    print("📝 Building prompt template...")
    prompt_template = get_clustering_prompt_template()
    
    # Format messages
    print("🔧 Formatting messages for prompt...")
    formatted_messages = format_messages_for_prompt(messages)
    
    # Create full prompt
    full_prompt = prompt_template.replace("{messages}", formatted_messages)
    
    print(f"   Formatted {len(messages)} messages")
    print()
    
    # Calculate token counts
    print("🧮 Calculating token counts...")
    print()
    
    # Simple estimation (original method)
    simple_tokens = estimate_tokens_simple(full_prompt)
    print(f"   Simple Estimation (4 chars/token): {simple_tokens:,} tokens")
    
    # More accurate estimation
    accurate_tokens = estimate_tokens_accurate(full_prompt)
    print(f"   Accurate Estimation (weighted):    {accurate_tokens:,} tokens")
    
    # Character and word counts
    char_count = len(full_prompt)
    word_count = len(full_prompt.split())
    print()
    print(f"   Character count: {char_count:,}")
    print(f"   Word count:      {word_count:,}")
    print()
    
    # Breakdown
    print("📊 PROMPT BREAKDOWN:")
    print("-" * 80)
    
    template_only = prompt_template.replace("{messages}", "")
    template_tokens = estimate_tokens_accurate(template_only)
    messages_tokens = estimate_tokens_accurate(formatted_messages)
    
    print(f"   Prompt template (instructions): {template_tokens:,} tokens")
    print(f"   Formatted messages (data):      {messages_tokens:,} tokens")
    print(f"   Total:                          {accurate_tokens:,} tokens")
    print()
    
    # Compare with actual Gemini output
    print("🔍 COMPARING WITH ACTUAL GEMINI OUTPUT:")
    print("-" * 80)
    
    gemini_output_file = "output/phase3_topic_clustering/google_gemini-2.0-flash-001.json"
    
    if os.path.exists(gemini_output_file):
        with open(gemini_output_file, 'r') as f:
            gemini_data = json.load(f)
        
        actual_prompt_tokens = gemini_data.get('usage', {}).get('prompt_tokens', 0)
        actual_completion_tokens = gemini_data.get('usage', {}).get('completion_tokens', 0)
        actual_total_tokens = gemini_data.get('usage', {}).get('total_tokens', 0)
        
        print(f"   Actual prompt_tokens (from API):      {actual_prompt_tokens:,}")
        print(f"   Actual completion_tokens (from API):  {actual_completion_tokens:,}")
        print(f"   Actual total_tokens (from API):       {actual_total_tokens:,}")
        print()
        
        # Calculate difference
        difference = actual_prompt_tokens - accurate_tokens
        percentage = (difference / actual_prompt_tokens * 100) if actual_prompt_tokens > 0 else 0
        
        print(f"   Estimated vs Actual:")
        print(f"   - Difference: {difference:,} tokens ({percentage:+.1f}%)")
        print(f"   - Estimation accuracy: {100 - abs(percentage):.1f}%")
    else:
        print(f"   ⚠️  Gemini output file not found: {gemini_output_file}")
        print(f"   Cannot compare with actual values")
    
    print()
    print("=" * 80)
    print("✅ CALCULATION COMPLETE")
    print("=" * 80)
    print()
    print("📌 SUMMARY:")
    print(f"   Input file: data/Synthetic_Slack_Messages.csv")
    print(f"   Messages processed: {len(messages)}")
    print(f"   Estimated input tokens: {accurate_tokens:,}")
    print()
    
    # Save detailed report
    report = {
        "input_file": "data/Synthetic_Slack_Messages.csv",
        "total_messages": len(messages),
        "character_count": char_count,
        "word_count": word_count,
        "estimated_tokens_simple": simple_tokens,
        "estimated_tokens_accurate": accurate_tokens,
        "prompt_template_tokens": template_tokens,
        "formatted_messages_tokens": messages_tokens,
    }
    
    if os.path.exists(gemini_output_file):
        report["actual_prompt_tokens"] = actual_prompt_tokens
        report["actual_completion_tokens"] = actual_completion_tokens
        report["actual_total_tokens"] = actual_total_tokens
        report["estimation_difference"] = difference
        report["estimation_accuracy_percent"] = 100 - abs(percentage)
    
    report_file = "step1_input_token_calculation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Detailed report saved: {report_file}")
    print()

if __name__ == "__main__":
    main()

