#!/usr/bin/env python3
"""
Step 5: User Topic Filtering
Manual filtering based on action items and channel access
NO AI - Pure rule-based logic
"""

import os
import json
from typing import Dict, List, Any
from collections import defaultdict
from datetime import datetime

class Step5UserFiltering:
    """Step 5: Filter topics per user based on action items and channel access"""
    
    def __init__(self):
        self.step_name = "step5_user_filtering"
        self.output_dir = os.path.join("output", self.step_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load topics from Step 3
        self.topics = self.load_topics_from_step3()
        
        # Derive channel memberships from message data (not hardcoded)
        self.user_channels = self.load_user_channel_memberships()
        
        print(f"✅ Step 5: User Filtering initialized")
        print(f"✅ Loaded {len(self.topics)} topics from Step 3")
        print(f"✅ Derived channel memberships for {len(self.user_channels)} users from Step 3 metadata")
    
    def load_topics_from_step3(self) -> List[Dict]:
        """Load topics from Step 3 JSON file"""
        try:
            step3_file = "output/phase5_metadata_generation/google_gemini-2.0-flash.json"
            
            if not os.path.exists(step3_file):
                print(f"❌ Step 3 file not found: {step3_file}")
                return []
            
            with open(step3_file, "r") as f:
                step3_data = json.load(f)
            
            if "metadata_results" in step3_data:
                topics = step3_data["metadata_results"]
                
                # Step 3 now includes channel info directly in metadata
                # No need to load from Step 1 anymore!
                print(f"✅ Loaded {len(topics)} topics from Step 3 (with channel info)")
                return topics
            else:
                print(f"❌ No 'metadata_results' found in Step 3 file")
                return []
            
        except Exception as e:
            print(f"❌ Error loading topics: {e}")
            return []
    
    def load_user_channel_memberships(self) -> Dict[str, List[str]]:
        """
        Derive channel memberships from Step 3 metadata.
        
        Logic: If a user is a participant in a topic, and the topic is in channel X,
               then that user is a member of channel X.
        
        This is more accurate than CSV because:
        - Step 3 already has topic-channel-participant relationships
        - Participants in a topic must be members of that topic's channel
        - Aligns with Step 5's topic-based filtering logic
        """
        user_channels = defaultdict(set)
        
        if not self.topics:
            print(f"⚠️  No topics loaded from Step 3")
            return {}
        
        for topic in self.topics:
            # Check if topic has valid metadata
            if not topic.get("success") or not topic.get("metadata"):
                continue
            
            metadata = topic["metadata"]
            channel = metadata.get("channel", "").lower()
            participants = metadata.get("participants", [])
            
            # Skip if channel is invalid
            if not channel or channel in ["n/a", "unspecified", ""]:
                continue
            
            # For each participant, add this channel to their membership list
            for participant in participants:
                # Remove @ prefix and convert to lowercase
                user = participant.replace("@", "").lower().strip()
                
                if user:  # Only add if user name is valid
                    user_channels[user].add(channel)
        
        # Convert sets to sorted lists
        result = {user: sorted(list(channels)) for user, channels in user_channels.items()}
        
        print(f"✅ Derived channel memberships from Step 3 metadata")
        print(f"   Logic: Participants in a topic → members of that topic's channel")
        for user, channels in sorted(result.items()):
            print(f"   {user}: {len(channels)} channels")
        
        return result
    
    def filter_topics_for_user(self, user_name: str) -> List[str]:
        """
        Filter topics for a specific user based on two rules:
        1. User must have at least one open action item in the topic
        2. User must be a member of the topic's channel
        """
        user_channels = self.user_channels.get(user_name.lower(), [])
        visible_topics = []
        
        for topic in self.topics:
            # Check if topic has valid metadata
            if not topic.get("success") or not topic.get("metadata"):
                continue
            
            metadata = topic["metadata"]
            channel = metadata.get("channel", "").lower()
            
            # Rule 1: Check if user is in the topic's channel
            user_in_channel = False
            for user_channel in user_channels:
                if user_channel.lower() == channel:
                    user_in_channel = True
                    break
            
            if not user_in_channel:
                continue  # Skip if user not in channel
            
            # Rule 2: Check if user has open action items in the topic
            action_items = metadata.get("action_items", [])
            user_has_action_items = False
            
            for action_item in action_items:
                owner = action_item.get("owner", "").lower()
                status = action_item.get("status", "").lower()
                
                # Check if user is the owner (handle multiple owners)
                if f"@{user_name.lower()}" in owner or f"@{user_name}" in owner:
                    # Only count open action items
                    if status in ["pending", "in_progress"]:
                        user_has_action_items = True
                        break
            
            if user_has_action_items:
                visible_topics.append(topic["cluster_id"])
        
        return visible_topics
    
    def run_filtering(self) -> Dict[str, List[str]]:
        """
        Run filtering for all users
        Returns: {"username": [topic_id1, topic_id2, ...]}
        """
        print(f"\n🎯 STEP 5: USER FILTERING")
        print("=" * 60)
        
        all_results = {}
        
        for user_name in self.user_channels.keys():
            visible_topics = self.filter_topics_for_user(user_name)
            all_results[user_name] = visible_topics
            
            print(f"✅ {user_name}: {len(visible_topics)} visible topics")
            if visible_topics:
                print(f"   Topics: {', '.join(visible_topics)}")
        
        # Save results
        output_file = os.path.join(self.output_dir, "filtered_topics.json")
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n📁 Results saved to: {output_file}")
        
        # Generate summary
        summary = {
            "step": "step5_user_filtering",
            "timestamp": datetime.now().isoformat(),
            "total_topics": len(self.topics),
            "users_processed": len(self.user_channels),
            "filtering_rules": {
                "rule1": "User must have at least one open action item in the topic",
                "rule2": "User must be a member of the topic's channel"
            },
            "user_channel_memberships": {user: channels for user, channels in self.user_channels.items()},
            "results": all_results
        }
        
        summary_file = os.path.join(self.output_dir, "filtering_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"📁 Summary saved to: {summary_file}")
        
        return all_results
    
    def verify_filtering(self, results: Dict[str, List[str]]):
        """
        Verify that filtering rules are working correctly
        """
        print(f"\n🔍 Verifying filtering rules...")
        
        all_verified = True
        
        for user_name, visible_topics in results.items():
            user_channels = self.user_channels.get(user_name.lower(), [])
            
            for topic_id in visible_topics:
                # Find the topic
                topic = next((t for t in self.topics if t["cluster_id"] == topic_id), None)
                if not topic:
                    print(f"❌ {user_name}: Topic {topic_id} not found!")
                    all_verified = False
                    continue
                
                metadata = topic["metadata"]
                channel = metadata.get("channel", "").lower()
                
                # Verify Rule 1: Channel membership
                user_in_channel = any(user_ch.lower() == channel for user_ch in user_channels)
                if not user_in_channel:
                    print(f"❌ {user_name}: Topic {topic_id} from wrong channel {channel}")
                    all_verified = False
                    continue
                
                # Verify Rule 2: Action items
                action_items = metadata.get("action_items", [])
                has_open_action = False
                for action in action_items:
                    owner = action.get("owner", "").lower()
                    status = action.get("status", "").lower()
                    if f"@{user_name.lower()}" in owner and status in ["pending", "in_progress"]:
                        has_open_action = True
                        break
                
                if not has_open_action:
                    print(f"❌ {user_name}: Topic {topic_id} has no open action items for user")
                    all_verified = False
        
        if all_verified:
            print("✅ All filtering rules verified correctly!")
        else:
            print("⚠️  Some filtering rules failed verification")

def main():
    """Main execution function"""
    print("🚀 Step 5: User Topic Filtering")
    print("Manual rule-based filtering (NO AI)")
    
    # Initialize Step 5
    step5 = Step5UserFiltering()
    
    # Run filtering for all users
    results = step5.run_filtering()
    
    # Verify the filtering
    step5.verify_filtering(results)
    
    print(f"\n✅ Step 5 completed successfully!")
    print(f"📁 Results saved to: output/{step5.step_name}/")
    print(f"🔒 {len(results)} users' topic lists created")

if __name__ == "__main__":
    main()

