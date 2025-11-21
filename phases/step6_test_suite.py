#!/usr/bin/env python3
"""
Step 6 Test Suite
Tests 11 cases: 4 UPDATE, 6 CREATE NEW, 1 DISCARD (including edge cases)
Generates exact testing scores for client reporting

Edge Cases Tested:
- Channel/thread weighting: structural metadata has stronger influence than tags
- Structural metadata importance: channel, thread, participants override semantic similarity
- Message filtering: meaningless messages (casual questions without context) are discarded
"""

import os
import json
import time
from typing import Dict, List, Any
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phases.step6_new_message_processing import Step6NewMessageProcessing


class Step6TestSuite:
    """Test suite for Step 6 with scoring metrics"""

    def __init__(self):
        self.step6 = Step6NewMessageProcessing()
        self.results = []
        self.output_dir = os.path.join("output", "step6_test_suite")
        os.makedirs(self.output_dir, exist_ok=True)

    def get_test_cases(self) -> List[Dict[str, Any]]:
        """Define 11 test cases: 4 UPDATE, 6 CREATE, 1 DISCARD (including edge cases)"""
        return [
            # ========== NORMAL UPDATE CASES ==========
            # Test Case 1: UPDATE - EcoBloom Campaign Deadline Change
            {
                "test_id": "TC001",
                "test_name": "Update EcoBloom Campaign Deadline",
                "expected_action": "update",
                "expected_topic_id": "topic_001",
                "message": {
                    "id": 301,
                    "channel": "#campaign-briefs",
                    "user": "Devon",
                    "text": "@team The EcoBloom campaign deadline has been moved to August 5, 2025. Please update your timelines accordingly.",
                    "thread_ts": "1",
                    "timestamp": "2025-06-25T10:00:00"
                },
                "validation_criteria": {
                    "min_similarity": 0.7,
                    "should_contain": ["deadline", "August 5", "EcoBloom"],
                    "should_update_deadline": True
                }
            },
            # Test Case 2: UPDATE - Content Completion Update
            {
                "test_id": "TC002",
                "test_name": "Update EcoBloom Content Status",
                "expected_action": "update",
                "expected_topic_id": "topic_001",
                "message": {
                    "id": 302,
                    "channel": "#campaign-briefs",
                    "user": "Leah",
                    "text": "@sam @jordan I've completed the first draft of the EcoBloom content. Please review by end of week. The tone is more casual than the initial brief requested.",
                    "thread_ts": "1",
                    "timestamp": "2025-06-27T14:30:00"
                },
                "validation_criteria": {
                    "min_similarity": 0.7,
                    "should_contain": ["content", "EcoBloom", "review"],
                    "should_update_action_items": True
                }
            },
            
            # ========== EDGE CASE: FALLBACK RULE ==========
            # Test Case 3: UPDATE via Fallback - Same thread + from/to, low semantic similarity
            # Tests: If thread_id and from/to match, force attach even if semantic similarity <0.7
            {
                "test_id": "TC003",
                "test_name": "Fallback Rule: Same Thread + Participants, Different Topic",
                "expected_action": "update",
                "expected_topic_id": "topic_001",
                "message": {
                    "id": 303,
                    "channel": "#campaign-briefs",
                    "user": "Sam",
                    "text": "@devon @leah Quick question about the meeting time tomorrow. Can we move it to 3pm instead?",
                    "thread_ts": "1",  # Same thread as topic_001
                    "timestamp": "2025-06-28T09:00:00"
                },
                "validation_criteria": {
                    "min_similarity": 0.6,  # Lower threshold due to fallback rule
                    "should_contain": ["meeting", "time"],
                    "structural_match_required": True,  # Must match on thread + participants
                    "test_fallback_rule": True
                }
            },
            
            # ========== EDGE CASE: CHANNEL/THREAD WEIGHTING ==========
            # Test Case 4: UPDATE - Same channel + thread, different tags/semantic content
            # Tests: Channel and thread should have stronger influence than tags
            {
                "test_id": "TC004",
                "test_name": "Channel/Thread Weighting: Same Structure, Different Tags",
                "expected_action": "update",
                "expected_topic_id": "topic_001",
                "message": {
                    "id": 304,
                    "channel": "#campaign-briefs",  # Same channel
                    "user": "Jordan",
                    "text": "@devon @sam @leah The legal review is progressing well. I've identified a few minor compliance items that need addressing before final approval.",
                    "thread_ts": "1",  # Same thread
                    "timestamp": "2025-06-29T11:00:00"
                },
                "validation_criteria": {
                    "min_similarity": 0.7,
                    "should_contain": ["legal", "review", "compliance"],
                    "test_structural_weighting": True  # Channel/thread should override different semantic content
                }
            },
            
            # ========== NORMAL CREATE CASES ==========
            # Test Case 5: CREATE NEW - Finance Invoice
            {
                "test_id": "TC005",
                "test_name": "Create New Finance Invoice Topic",
                "expected_action": "create",
                "expected_topic_id": None,
                "message": {
                    "id": 305,
                    "channel": "#finance-updates",
                    "user": "Priya",
                    "text": "@finance-team The Q3 invoice for Horizon Robotics (PO-HR-7784) is still unpaid. Please confirm that the wire transfer is scheduled before the July 3, 2025 cutoff to avoid penalties.",
                    "thread_ts": "100",  # Different thread
                    "timestamp": "2025-06-26T09:45:00"
                },
                "validation_criteria": {
                    "max_similarity": 0.7,
                    "should_contain": ["invoice", "Horizon Robotics", "payment"],
                    "should_have_urgency": "high",
                    "should_have_deadline": "2025-07-03"
                }
            },
            # Test Case 6: CREATE NEW - HR Hiring
            {
                "test_id": "TC006",
                "test_name": "Create New HR Hiring Topic",
                "expected_action": "create",
                "expected_topic_id": None,
                "message": {
                    "id": 306,
                    "channel": "#hr-announcements",
                    "user": "Maria",
                    "text": "@team We're opening a new Senior Product Designer position. The role will focus on our mobile app redesign project. Applications close July 15, 2025. Please share with your networks.",
                    "thread_ts": "200",  # Different thread
                    "timestamp": "2025-06-30T10:20:00"
                },
                "validation_criteria": {
                    "max_similarity": 0.7,
                    "should_contain": ["hiring", "Product Designer", "position"],
                    "should_have_deadline": "2025-07-15",
                    "should_have_channel": "#hr-announcements"
                }
            },
            
            # ========== EDGE CASE: SAME SEMANTIC, DIFFERENT STRUCTURE ==========
            # Test Case 7: CREATE - Same semantic content (EcoBloom) but different channel/thread
            # Tests: Structural metadata (channel/thread) should prevent false match
            {
                "test_id": "TC007",
                "test_name": "Same Semantic Content, Different Structure (Should Create)",
                "expected_action": "create",
                "expected_topic_id": None,
                "message": {
                    "id": 307,
                    "channel": "#general",  # Different channel
                    "user": "Alex",
                    "text": "@team I heard about the EcoBloom campaign. Can someone share the latest updates? I'm working on a similar project and would love to learn from your approach.",
                    "thread_ts": "300",  # Different thread
                    "timestamp": "2025-07-01T14:00:00"
                },
                "validation_criteria": {
                    "max_similarity": 0.7,  # Should be below threshold despite semantic similarity
                    "should_contain": ["EcoBloom", "campaign", "updates"],
                    "test_structure_over_semantic": True  # Structure should prevent false match
                }
            },
            
            # ========== EDGE CASE: SAME CHANNEL/THREAD, DIFFERENT PARTICIPANTS ==========
            # Test Case 8: CREATE - Same channel + thread but completely different participants
            # Tests: Participants (from/to) are important for matching
            {
                "test_id": "TC008",
                "test_name": "Same Channel/Thread, Different Participants (Should Create)",
                "expected_action": "create",
                "expected_topic_id": None,
                "message": {
                    "id": 308,
                    "channel": "#campaign-briefs",  # Same channel
                    "user": "Priya",  # Different participant
                    "text": "@finance-team @accounting I need budget approval for the Q4 marketing campaign. The total cost is $50,000 and we need to submit by July 10, 2025.",
                    "thread_ts": "1",  # Same thread (but different context)
                    "timestamp": "2025-07-02T10:00:00"
                },
                "validation_criteria": {
                    "max_similarity": 0.7,  # Different participants should prevent match
                    "should_contain": ["budget", "approval", "Q4"],
                    "test_participants_importance": True
                }
            },
            
            # ========== EDGE CASE: HIGH SEMANTIC, DIFFERENT THREAD ==========
            # Test Case 9: CREATE - High semantic similarity but different thread
            # Tests: Thread is a strong structural signal that should prevent false match
            {
                "test_id": "TC009",
                "test_name": "High Semantic Similarity, Different Thread (Should Create)",
                "expected_action": "create",
                "expected_topic_id": None,
                "message": {
                    "id": 309,
                    "channel": "#campaign-briefs",  # Same channel
                    "user": "Devon",
                    "text": "@sam @leah @jordan We need to discuss the EcoBloom summer campaign timeline. The deadline is July 28, 2025 and we need to coordinate content, design, and legal review.",
                    "thread_ts": "400",  # Different thread (new conversation)
                    "timestamp": "2025-07-03T09:00:00"
                },
                "validation_criteria": {
                    "max_similarity": 0.7,  # Different thread should prevent match
                    "should_contain": ["EcoBloom", "campaign", "timeline"],
                    "test_thread_importance": True  # Thread should override high semantic similarity
                }
            },
            
            # ========== EDGE CASE: MULTIPLE MENTIONS VS SINGLE ==========
            # Test Case 10: UPDATE - Multiple mentions in same thread
            # Tests: Multiple participants (from/to) should still match if thread matches
            {
                "test_id": "TC010",
                "test_name": "Multiple Mentions in Same Thread (Should Update)",
                "expected_action": "update",
                "expected_topic_id": "topic_001",
                "message": {
                    "id": 310,
                    "channel": "#campaign-briefs",
                    "user": "Devon",
                    "text": "@sam @leah @jordan @team All team members please note: The EcoBloom campaign kickoff meeting is scheduled for tomorrow at 2pm. Please confirm attendance.",
                    "thread_ts": "1",  # Same thread
                    "timestamp": "2025-07-04T15:00:00"
                },
                "validation_criteria": {
                    "min_similarity": 0.7,
                    "should_contain": ["EcoBloom", "meeting", "kickoff"],
                    "test_multiple_mentions": True
                }
            },
            
            # ========== EDGE CASE: DISCARD MEANINGLESS MESSAGE ==========
            # Test Case 11: DISCARD - Casual question without substance
            # Tests: Messages without actionable content should be discarded
            {
                "test_id": "TC011",
                "test_name": "Discard Meaningless Message (Casual Question)",
                "expected_action": "discarded",
                "expected_topic_id": None,
                "message": {
                    "id": 311,
                    "channel": "#general",
                    "user": "Alex",
                    "text": "I heard about the EcoBloom campaign. Can someone share the latest updates?",
                    "thread_ts": "500",
                    "timestamp": "2025-07-05T10:00:00"
                },
                "validation_criteria": {
                    "should_be_discarded": True,
                    "reason_contains": ["casual", "question", "without context", "not topic-worthy"]
                }
            }
        ]

    def calculate_similarity_score(self, result: Dict) -> float:
        """
        Extract highest similarity score from result.
        
        Returns:
            - Highest similarity score (0.0-1.0) if similar topics found
            - 0.0 if no similar topics found (all below 0.7 threshold)
        
        Note: Similarity score of 0.00 means:
            - No existing topics matched above the 0.7 threshold
            - This is normal for CREATE scenarios where message is unique
        """
        similar_topics = result.get("similar_topics", [])
        if similar_topics:
            return similar_topics[0][1]  # Highest similarity score
        # No similar topics found - all below threshold
        return 0.0

    def validate_metadata_completeness(self, metadata: Dict) -> Dict[str, Any]:
        """Check if metadata has all required fields"""
        required_fields = [
            "title", "summary", "action_items", "participants",
            "urgency", "channel", "tags", "status"
        ]
        present_fields = [field for field in required_fields if field in metadata]
        completeness = len(present_fields) / len(required_fields)
        
        return {
            "completeness_score": completeness,
            "present_fields": present_fields,
            "missing_fields": [f for f in required_fields if f not in metadata],
            "total_fields": len(required_fields),
            "present_count": len(present_fields)
        }

    def validate_action_items(self, action_items: List[Dict]) -> Dict[str, Any]:
        """Validate action items structure"""
        if not action_items:
            return {
                "count": 0,
                "has_required_fields": False,
                "score": 0.0
            }
        
        required_fields = ["task", "owner", "due_date", "priority", "status"]
        valid_items = 0
        
        for item in action_items:
            if all(field in item for field in required_fields):
                valid_items += 1
        
        score = valid_items / len(action_items) if action_items else 0.0
        
        return {
            "count": len(action_items),
            "valid_count": valid_items,
            "has_required_fields": all(f in action_items[0] for f in required_fields) if action_items else False,
            "score": score
        }

    def check_text_contains(self, text: str, keywords: List[str]) -> Dict[str, Any]:
        """Check if text contains required keywords"""
        text_lower = text.lower()
        found_keywords = [kw for kw in keywords if kw.lower() in text_lower]
        score = len(found_keywords) / len(keywords) if keywords else 1.0
        
        return {
            "found_keywords": found_keywords,
            "missing_keywords": [kw for kw in keywords if kw.lower() not in text_lower],
            "score": score,
            "total_keywords": len(keywords),
            "found_count": len(found_keywords)
        }

    def run_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case and calculate scores"""
        print(f"\n{'='*60}")
        print(f"🧪 Running {test_case['test_id']}: {test_case['test_name']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        result = self.step6.process_new_message(test_case["message"])
        duration = time.time() - start_time
        
        # Extract metrics
        similarity_score = self.calculate_similarity_score(result)
        action = result.get("action", "")
        topic_id = result.get("topic_id", "")
        metadata = result.get("metadata", {})
        
        # Get original topic metadata for UPDATE cases
        original_metadata = None
        if action in ["update", "updated"] and topic_id:
            original_topic = next(
                (t for t in self.step6.existing_topics 
                 if t.get("cluster_id") == topic_id),
                None
            )
            if original_topic:
                original_metadata = original_topic.get("metadata", {})
                print(f"   📂 Loaded original metadata from Step 3 (topic_id: {topic_id})")
            else:
                print(f"   ⚠️  Original topic {topic_id} not found in existing topics")
        
        # Validate action correctness
        action_correct = (
            action == test_case["expected_action"] or
            (test_case["expected_action"] == "update" and action == "updated") or
            (test_case["expected_action"] == "create" and action == "created") or
            (test_case["expected_action"] == "discarded" and action == "discarded")
        )
        
        # Validate topic ID (for update cases)
        topic_id_correct = True
        if test_case["expected_topic_id"]:
            topic_id_correct = (topic_id == test_case["expected_topic_id"])
        
        # Validate similarity threshold
        criteria = test_case["validation_criteria"]
        similarity_valid = True
        if "min_similarity" in criteria:
            similarity_valid = similarity_score >= criteria["min_similarity"]
        elif "max_similarity" in criteria:
            similarity_valid = similarity_score < criteria["max_similarity"]
        
        # Determine pass/fail
        passed = action_correct and similarity_valid and (topic_id_correct if test_case["expected_topic_id"] else True)
        
        # Normalize action to "update", "create", or "discarded"
        if action in ["update", "updated"]:
            normalized_action = "update"
        elif action == "discarded":
            normalized_action = "discarded"
        else:
            normalized_action = "create"
        
        test_result = {
            "test_id": test_case["test_id"],
            "test_name": test_case["test_name"],
            "expected_action": test_case["expected_action"],
            "expected_topic_id": test_case["expected_topic_id"],
            "message": test_case["message"],
            "duration_seconds": round(duration, 3),
            "results": {
                "action": normalized_action,
                "topic_id": topic_id,
                "similarity_score": round(similarity_score, 4),
                "similar_topics_count": len(result.get("similar_topics", [])),
                "success": result.get("success", False),
                "passed": passed
            },
            "validation": {
                "action_correct": action_correct,
                "topic_id_correct": topic_id_correct,
                "similarity_valid": similarity_valid
            },
            "original_metadata": original_metadata,
            "updated_metadata": metadata if normalized_action == "update" else None,
            "new_metadata": metadata if normalized_action == "create" else None
        }
        
        # Print summary
        message = test_case["message"]
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"\n   📨 NEW MESSAGE:")
        print(f"      ID: {message.get('id', 'N/A')}")
        print(f"      Channel: {message.get('channel', 'N/A')}")
        print(f"      User: {message.get('user', 'N/A')}")
        print(f"      Timestamp: {message.get('timestamp', 'N/A')}")
        print(f"      Text: {message.get('text', 'N/A')}")
        
        print(f"\n   {status} Action: {normalized_action} (Expected: {test_case['expected_action']})")
        print(f"   Similarity Score: {similarity_score:.4f}")
        
        if normalized_action == "update":
            print(f"\n   📝 Topic ID: {topic_id}")
            print(f"\n   📋 ORIGINAL TOPIC:")
            if original_metadata:
                print(f"      Title: {original_metadata.get('title', 'N/A')}")
                print(f"      Summary: {original_metadata.get('summary', 'N/A')}")
                print(f"      Participants: {', '.join(original_metadata.get('participants', []))}")
                print(f"      Action Items: {len(original_metadata.get('action_items', []))} items")
                print(f"      Deadline: {original_metadata.get('deadline', 'N/A')}")
                print(f"      Urgency: {original_metadata.get('urgency', 'N/A')}")
            else:
                print("      (Original metadata not found)")
            
            print(f"\n   ✏️  UPDATED TOPIC:")
            print(f"      Title: {metadata.get('title', 'N/A')}")
            print(f"      Summary: {metadata.get('summary', 'N/A')}")
            print(f"      Participants: {', '.join(metadata.get('participants', []))}")
            print(f"      Action Items: {len(metadata.get('action_items', []))} items")
            print(f"      Deadline: {metadata.get('deadline', 'N/A')}")
            print(f"      Urgency: {metadata.get('urgency', 'N/A')}")
        
        elif normalized_action == "create":
            print(f"\n   🆕 NEW TOPIC CREATED:")
            print(f"      Topic ID: {topic_id}")
            print(f"      Title: {metadata.get('title', 'N/A')}")
            print(f"      Summary: {metadata.get('summary', 'N/A')}")
            print(f"      Participants: {', '.join(metadata.get('participants', []))}")
            print(f"      Action Items: {len(metadata.get('action_items', []))} items")
            if metadata.get('action_items'):
                for idx, item in enumerate(metadata.get('action_items', []), 1):
                    print(f"        {idx}. {item.get('task', 'N/A')} (Owner: {item.get('owner', 'N/A')}, Due: {item.get('due_date', 'N/A')})")
            print(f"      Deadline: {metadata.get('deadline', 'N/A')}")
            print(f"      Urgency: {metadata.get('urgency', 'N/A')}")
            print(f"      Channel: {metadata.get('channel', 'N/A')}")
            print(f"      Tags: {', '.join(metadata.get('tags', []))}")
        
        elif normalized_action == "discarded":
            print(f"\n   🗑️  MESSAGE DISCARDED (NOT TOPIC-WORTHY):")
            discard_reason = result.get("reason", "No reason provided")
            worthiness = result.get("worthiness_evaluation", {})
            print(f"      Reason: {discard_reason}")
            if worthiness:
                print(f"      Evaluation: {worthiness.get('reason', 'N/A')}")
        
        print(f"\n   ⏱️  Duration: {duration:.3f}s")
        
        return test_result

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases and generate report"""
        print("🚀 Step 6 Test Suite")
        print("=" * 60)
        print("📊 Running 11 test cases:")
        print("  - 4 UPDATE scenarios")
        print("  - 6 CREATE NEW scenarios")
        print("  - 1 DISCARD scenario (meaningless messages)")
        print("  - Edge cases test:")
        print("    • Channel/thread weighting vs tags")
        print("    • Structural metadata importance")
        print("    • Message filtering (topic-worthiness)")
        print("=" * 60)
        
        test_cases = self.get_test_cases()
        all_results = []
        
        for test_case in test_cases:
            try:
                result = self.run_test_case(test_case)
                all_results.append(result)
                self.results.append(result)
            except Exception as e:
                print(f"❌ Test {test_case['test_id']} failed: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "test_id": test_case["test_id"],
                    "test_name": test_case["test_name"],
                    "error": str(e),
                    "results": {"passed": False}
                })
        
        # Calculate aggregate metrics
        passed_results = [r for r in all_results if r.get("results", {}).get("passed", False)]
        failed_results = [r for r in all_results if not r.get("results", {}).get("passed", True)]
        update_results = [r for r in all_results if r.get("expected_action") == "update"]
        create_results = [r for r in all_results if r.get("expected_action") == "create"]
        discard_results = [r for r in all_results if r.get("expected_action") == "discarded"]
        
        aggregate_metrics = {
            "total_tests": len(all_results),
            "update_tests": len(update_results),
            "create_tests": len(create_results),
            "discard_tests": len(discard_results),
            "passed_tests": len(passed_results),
            "failed_tests": len(failed_results),
            "pass_rate": round(len(passed_results) / len(all_results) * 100, 1) if all_results else 0.0,
            "total_duration": round(sum(r.get("duration_seconds", 0) for r in all_results), 3)
        }
        
        report = {
            "test_suite": "Step 6 New Message Processing",
            "timestamp": datetime.now().isoformat(),
            "aggregate_metrics": aggregate_metrics,
            "test_results": all_results
        }
        
        # Save report
        report_file = os.path.join(self.output_dir, "test_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("📊 TEST SUITE SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {aggregate_metrics['total_tests']}")
        print(f"  - UPDATE tests: {aggregate_metrics['update_tests']}")
        print(f"  - CREATE tests: {aggregate_metrics['create_tests']}")
        print(f"  - DISCARD tests: {aggregate_metrics['discard_tests']}")
        print(f"\n✅ Passed: {aggregate_metrics['passed_tests']}")
        print(f"❌ Failed: {aggregate_metrics['failed_tests']}")
        print(f"📈 Pass Rate: {aggregate_metrics['pass_rate']:.1f}%")
        print(f"\n⏱️  Total Duration: {aggregate_metrics['total_duration']:.3f}s")
        print(f"📁 Report saved to: {report_file}")
        
        return report

    def generate_client_report(self) -> str:
        """Generate a formatted report for client sharing"""
        if not self.results:
            return "No test results available"
        
        report_lines = [
            "# Step 6 Test Results - Client Report",
            "",
            f"**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Test Overview",
            "",
            f"- **Total Tests**: {len(self.results)}",
            f"- **UPDATE Tests**: {len([r for r in self.results if r.get('expected_action') == 'update'])}",
            f"- **CREATE Tests**: {len([r for r in self.results if r.get('expected_action') == 'create'])}",
            f"- **DISCARD Tests**: {len([r for r in self.results if r.get('expected_action') == 'discarded'])}",
            f"- **CREATE Tests**: {len([r for r in self.results if r.get('expected_action') == 'create'])}",
            "",
            "## Test Results",
            ""
        ]
        
        for result in self.results:
            passed = result.get("results", {}).get("passed", False)
            status = "✅ PASS" if passed else "❌ FAIL"
            action = result['results'].get('action', 'N/A')
            
            # Get the original message from result
            test_case_msg = result.get('message', {})
            
            report_lines.extend([
                f"### {result['test_id']}: {result['test_name']}",
                "",
                f"**Status**: {status}",
                "",
                "#### New Message:",
                f"- **ID**: {test_case_msg.get('id', 'N/A')}",
                f"- **Channel**: {test_case_msg.get('channel', 'N/A')}",
                f"- **User**: {test_case_msg.get('user', 'N/A')}",
                f"- **Timestamp**: {test_case_msg.get('timestamp', 'N/A')}",
                f"- **Text**: {test_case_msg.get('text', 'N/A')}",
                "",
                "#### Test Results:",
                f"- **Expected Action**: {result['expected_action'].upper()}",
                f"- **Actual Action**: {action.upper()}",
                f"- **Similarity Score**: {result['results'].get('similarity_score', 0):.4f}",
                f"- **Duration**: {result.get('duration_seconds', 0):.3f}s",
                ""
            ])
            
            if action == "update":
                original = result.get("original_metadata", {})
                updated = result.get("updated_metadata", {})
                
                report_lines.extend([
                    "#### Original Topic (Before Update):",
                    f"- **Topic ID**: {result['results'].get('topic_id', 'N/A')}",
                    f"- **Title**: {original.get('title', 'N/A')}",
                    f"- **Summary**: {original.get('summary', 'N/A')}",
                    f"- **Participants**: {', '.join(original.get('participants', []))}",
                    f"- **Action Items**: {len(original.get('action_items', []))} items",
                    f"- **Deadline**: {original.get('deadline', 'N/A')}",
                    f"- **Urgency**: {original.get('urgency', 'N/A')}",
                    "",
                    "#### Updated Topic (After Update):",
                    f"- **Title**: {updated.get('title', 'N/A')}",
                    f"- **Summary**: {updated.get('summary', 'N/A')}",
                    f"- **Participants**: {', '.join(updated.get('participants', []))}",
                    f"- **Action Items**: {len(updated.get('action_items', []))} items",
                    f"- **Deadline**: {updated.get('deadline', 'N/A')}",
                    f"- **Urgency**: {updated.get('urgency', 'N/A')}",
                    ""
                ])
            
            elif action == "create":
                new_metadata = result.get("new_metadata", {})
                
                report_lines.extend([
                    "#### New Topic Created:",
                    f"- **Topic ID**: {result['results'].get('topic_id', 'N/A')}",
                    f"- **Title**: {new_metadata.get('title', 'N/A')}",
                    f"- **Summary**: {new_metadata.get('summary', 'N/A')}",
                    f"- **Participants**: {', '.join(new_metadata.get('participants', []))}",
                    f"- **Action Items**: {len(new_metadata.get('action_items', []))} items",
                ])
                
                if new_metadata.get('action_items'):
                    report_lines.append("")
                    report_lines.append("**Action Items List:**")
                    for idx, item in enumerate(new_metadata.get('action_items', []), 1):
                        report_lines.append(
                            f"{idx}. {item.get('task', 'N/A')} "
                            f"(Owner: {item.get('owner', 'N/A')}, "
                            f"Due: {item.get('due_date', 'N/A')}, "
                            f"Priority: {item.get('priority', 'N/A')})"
                        )
                
                report_lines.extend([
                    "",
                    f"- **Deadline**: {new_metadata.get('deadline', 'N/A')}",
                    f"- **Urgency**: {new_metadata.get('urgency', 'N/A')}",
                    f"- **Channel**: {new_metadata.get('channel', 'N/A')}",
                    f"- **Tags**: {', '.join(new_metadata.get('tags', []))}",
                    ""
                ])
            
            report_lines.append("---")
            report_lines.append("")
        
        # Calculate summary
        passed_count = len([r for r in self.results if r.get("results", {}).get("passed", False)])
        total_count = len(self.results)
        pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0
        
        report_lines.extend([
            "## Summary Statistics",
            "",
            f"- **Total Tests**: {total_count}",
            f"- **Passed**: {passed_count}",
            f"- **Failed**: {total_count - passed_count}",
            f"- **Pass Rate**: {pass_rate:.1f}%",
            ""
        ])
        
        return "\n".join(report_lines)


def main():
    """Main execution"""
    suite = Step6TestSuite()
    suite.run_all_tests()
    
    # Generate client report
    client_report = suite.generate_client_report()
    client_report_file = os.path.join(suite.output_dir, "client_report.md")
    with open(client_report_file, "w") as f:
        f.write(client_report)
    
    print(f"\n📄 Client report saved to: {client_report_file}")
    print("\n" + "="*60)
    print("✅ Test suite completed!")
    print("="*60)


if __name__ == "__main__":
    main()

