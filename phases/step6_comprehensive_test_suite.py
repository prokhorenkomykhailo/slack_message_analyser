#!/usr/bin/env python3
"""
Step 6 Comprehensive Test Suite - Realistic Technical Scenarios
Tests 20+ cases covering ALL scenarios:
- UPDATE (thread match, similar content)
- CREATE (new topic, different structure)
- DISCARD (meaningless messages)
- EDGE CASES (borderline similarity, mixed metadata)
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


class ComprehensiveTestSuite:
    """Comprehensive test suite with realistic technical scenarios"""

    def __init__(self):
        self.step6 = Step6NewMessageProcessing()
        self.results = []
        self.output_dir = os.path.join("output", "step6_comprehensive_tests")
        os.makedirs(self.output_dir, exist_ok=True)

    def get_test_cases(self) -> List[Dict[str, Any]]:
        """
        Define 28 comprehensive test cases covering:
        - 8 UPDATE scenarios (same thread/channel/participants with actionable info)
        - 10 CREATE scenarios (new topics, different structure)
        - 5 DISCARD scenarios (meaningless messages, new threads)
        - 3 UPDATE_NON_ACTIONABLE scenarios (CRITICAL: same thread but no actionable info)
        - 2 EDGE CASES (borderline similarity)
        """
        return [
            # ========== UPDATE SCENARIOS ==========
            
            # TC001: UPDATE - Status update in same thread
            {
                "test_id": "TC001",
                "category": "UPDATE",
                "test_name": "Status Update - Bug Fix Progress",
                "expected_action": "update",
                "expected_topic_id": "topic_001",
                "message": {
                    "id": 1001,
                    "channel": "#campaign-briefs",
                    "user": "Sam",
                    "text": "@Devon I've completed the design mockups for EcoBloom. Ready for your review. Files are in Figma: https://figma.com/ecobloom-designs",
                    "thread_ts": "1",
                    "timestamp": "2025-06-26T14:30:00"
                },
                "expected_behavior": "Should update topic_001 (EcoBloom Campaign) and mark Sam's design task as completed"
            },
            
            # TC002: UPDATE - Deadline extension in same thread
            {
                "test_id": "TC002",
                "category": "UPDATE",
                "test_name": "Deadline Extension - Same Thread",
                "expected_action": "update",
                "expected_topic_id": "topic_001",
                "message": {
                    "id": 1002,
                    "channel": "#campaign-briefs",
                    "user": "Devon",
                    "text": "@team Update: The EcoBloom campaign deadline has been extended to August 15, 2025 due to client request. Please adjust your timelines.",
                    "thread_ts": "1",
                    "timestamp": "2025-06-27T09:00:00"
                },
                "expected_behavior": "Should update topic_001 deadline field and summary"
            },
            
            # TC003: UPDATE - New action item in same thread
            {
                "test_id": "TC003",
                "category": "UPDATE",
                "test_name": "New Action Item - Same Thread",
                "expected_action": "update",
                "expected_topic_id": "topic_001",
                "message": {
                    "id": 1003,
                    "channel": "#campaign-briefs",
                    "user": "Leah",
                    "text": "@Jordan Can you also review the legal compliance for EU GDPR requirements? We need this by July 20, 2025 before launch.",
                    "thread_ts": "1",
                    "timestamp": "2025-06-28T10:15:00"
                },
                "expected_behavior": "Should add new action item for Jordan (GDPR review) without removing existing items"
            },
            
            # TC004: UPDATE - Task completion notification
            {
                "test_id": "TC004",
                "category": "UPDATE",
                "test_name": "Task Completion - Same Thread",
                "expected_action": "update",
                "expected_topic_id": "topic_001",
                "message": {
                    "id": 1004,
                    "channel": "#campaign-briefs",
                    "user": "Leah",
                    "text": "@Devon Content strategy is done! I've uploaded all materials to Google Drive. Ready for review.",
                    "thread_ts": "1",
                    "timestamp": "2025-06-29T16:45:00"
                },
                "expected_behavior": "Should mark Leah's content task as 'completed' status"
            },
            
            # TC005: UPDATE - Multiple participants mentioned
            {
                "test_id": "TC005",
                "category": "UPDATE",
                "test_name": "Multiple Participants - Same Thread",
                "expected_action": "update",
                "expected_topic_id": "topic_001",
                "message": {
                    "id": 1005,
                    "channel": "#campaign-briefs",
                    "user": "Devon",
                    "text": "@sam @leah @jordan Team sync tomorrow at 2pm to review EcoBloom progress. Please bring your status updates.",
                    "thread_ts": "1",
                    "timestamp": "2025-06-30T11:00:00"
                },
                "expected_behavior": "Should add meeting action item for all three participants"
            },
            
            # TC006: UPDATE - Question in same thread (topic-worthy)
            {
                "test_id": "TC006",
                "category": "UPDATE",
                "test_name": "Technical Question - Same Thread",
                "expected_action": "update",
                "expected_topic_id": "topic_001",
                "message": {
                    "id": 1006,
                    "channel": "#campaign-briefs",
                    "user": "Sam",
                    "text": "@Devon Should we use the old EcoBloom brand colors (#2D5016, #8BC34A) or the new rebrand palette for this campaign? This affects all my designs.",
                    "thread_ts": "1",
                    "timestamp": "2025-07-01T09:30:00"
                },
                "expected_behavior": "Should update topic with decision needed (brand colors) as action item"
            },
            
            # TC007: UPDATE - Resource request in same thread
            {
                "test_id": "TC007",
                "category": "UPDATE",
                "test_name": "Resource Request - Same Thread",
                "expected_action": "update",
                "expected_topic_id": "topic_001",
                "message": {
                    "id": 1007,
                    "channel": "#campaign-briefs",
                    "user": "Jordan",
                    "text": "@Devon I need budget approval for $5,000 to hire an external copywriter for the EcoBloom landing page. Our internal team is at capacity.",
                    "thread_ts": "1",
                    "timestamp": "2025-07-02T14:00:00"
                },
                "expected_behavior": "Should add budget approval as new action item with $5,000 amount"
            },
            
            # TC008: UPDATE - Risk/blocker notification
            {
                "test_id": "TC008",
                "category": "UPDATE",
                "test_name": "Blocker Notification - Same Thread",
                "expected_action": "update",
                "expected_topic_id": "topic_001",
                "message": {
                    "id": 1008,
                    "channel": "#campaign-briefs",
                    "user": "Sam",
                    "text": "@Devon URGENT: EcoBloom client hasn't approved the design concepts yet. This is blocking development. Can you follow up with them ASAP?",
                    "thread_ts": "1",
                    "timestamp": "2025-07-03T08:15:00"
                },
                "expected_behavior": "Should increase urgency to 'critical' and add blocker as action item"
            },
            
            # ========== CREATE SCENARIOS ==========
            
            # TC009: CREATE - Different channel, similar content
            {
                "test_id": "TC009",
                "category": "CREATE",
                "test_name": "Different Channel - New Topic",
                "expected_action": "create",
                "expected_topic_id": None,
                "message": {
                    "id": 2001,
                    "channel": "#engineering",
                    "user": "Alex",
                    "text": "@backend-team We have a critical API performance issue in production. Response times are 5-8 seconds (should be <500ms). User ID 47291 reported timeouts. Need root cause analysis by EOD.",
                    "thread_ts": "100",
                    "timestamp": "2025-06-26T10:00:00"
                },
                "expected_behavior": "Should create new topic about API performance (different channel from EcoBloom)"
            },
            
            # TC010: CREATE - Same channel, different thread, different topic
            {
                "test_id": "TC010",
                "category": "CREATE",
                "test_name": "Same Channel, Different Thread - New Topic",
                "expected_action": "create",
                "expected_topic_id": None,
                "message": {
                    "id": 2002,
                    "channel": "#campaign-briefs",
                    "user": "Maria",
                    "text": "@creative-team Launching new Q3 campaign for TechNova Software (tech startup client). Need logo design, website mockups, and social media assets by July 30, 2025. Budget: $15,000.",
                    "thread_ts": "200",
                    "timestamp": "2025-06-26T11:30:00"
                },
                "expected_behavior": "Should create new topic (TechNova campaign, different from EcoBloom)"
            },
            
            # TC011: CREATE - Bug report
            {
                "test_id": "TC011",
                "category": "CREATE",
                "test_name": "Bug Report - New Topic",
                "expected_action": "create",
                "expected_topic_id": None,
                "message": {
                    "id": 2003,
                    "channel": "#bug-reports",
                    "user": "Priya",
                    "text": "🐛 BUG: User authentication fails intermittently on Safari 17.2 (macOS Sonoma). Error: 'Invalid CSRF token'. Affects ~500 users/day. Reproduction steps: 1) Open app in Safari 2) Clear cookies 3) Try to log in. Priority: HIGH",
                    "thread_ts": "300",
                    "timestamp": "2025-06-26T13:00:00"
                },
                "expected_behavior": "Should create new bug topic with reproduction steps and priority"
            },
            
            # TC012: CREATE - Feature request
            {
                "test_id": "TC012",
                "category": "CREATE",
                "test_name": "Feature Request - New Topic",
                "expected_action": "create",
                "expected_topic_id": None,
                "message": {
                    "id": 2004,
                    "channel": "#product-ideas",
                    "user": "Kevin",
                    "text": "@product-team Feature request from 15 enterprise clients: Bulk user import via CSV. Requirements: Support 10k+ users, role assignment, custom fields mapping, validation errors report. Est. dev time: 3 sprints. ROI: $50k ARR.",
                    "thread_ts": "400",
                    "timestamp": "2025-06-26T14:30:00"
                },
                "expected_behavior": "Should create feature request topic with requirements and ROI"
            },
            
            # TC013: CREATE - Deployment notification
            {
                "test_id": "TC013",
                "category": "CREATE",
                "test_name": "Deployment Notification - New Topic",
                "expected_action": "create",
                "expected_topic_id": None,
                "message": {
                    "id": 2005,
                    "channel": "#deployments",
                    "user": "DevOps",
                    "text": "🚀 DEPLOYMENT SCHEDULED: v2.5.0 to production on July 5, 2025 at 2am UTC (maintenance window). Changes: new payment gateway, database migration, API v3 rollout. Estimated downtime: 30min. Rollback plan ready.",
                    "thread_ts": "500",
                    "timestamp": "2025-06-26T16:00:00"
                },
                "expected_behavior": "Should create deployment topic with version, date, changes, downtime"
            },
            
            # TC014: CREATE - Hiring announcement
            {
                "test_id": "TC014",
                "category": "CREATE",
                "test_name": "Hiring Announcement - New Topic",
                "expected_action": "create",
                "expected_topic_id": None,
                "message": {
                    "id": 2006,
                    "channel": "#hr-announcements",
                    "user": "HR",
                    "text": "@team Opening: Senior Backend Engineer (Python/Django). Location: Remote US. Salary: $140k-$180k. Requirements: 5+ years Python, microservices, Kubernetes. Applications due July 20, 2025. Refer qualified candidates for $3k bonus.",
                    "thread_ts": "600",
                    "timestamp": "2025-06-27T09:00:00"
                },
                "expected_behavior": "Should create hiring topic with role, requirements, salary, deadline"
            },
            
            # TC015: CREATE - Security incident
            {
                "test_id": "TC015",
                "category": "CREATE",
                "test_name": "Security Incident - New Topic",
                "expected_action": "create",
                "expected_topic_id": None,
                "message": {
                    "id": 2007,
                    "channel": "#security-alerts",
                    "user": "Security",
                    "text": "⚠️ SECURITY: Detected unusual API traffic from IP 185.220.101.x (TOR exit node). 10k requests/min to /api/users endpoint. Possible data scraping attempt. Actions: 1) Rate limiting enabled 2) IP blocked 3) Investigating data exposure. Incident ID: SEC-2025-047",
                    "thread_ts": "700",
                    "timestamp": "2025-06-27T11:45:00"
                },
                "expected_behavior": "Should create security incident topic with incident ID and actions taken"
            },
            
            # TC016: CREATE - Customer escalation
            {
                "test_id": "TC016",
                "category": "CREATE",
                "test_name": "Customer Escalation - New Topic",
                "expected_action": "create",
                "expected_topic_id": None,
                "message": {
                    "id": 2008,
                    "channel": "#customer-escalations",
                    "user": "Support",
                    "text": "🔥 ESCALATION: Enterprise customer AcmeCorp (account ID: 12847, MRR: $25k/mo) threatening to churn. Issue: Data export feature broken for 3 weeks, no response from eng team. CEO demanding call with our CEO. Need immediate fix + exec response.",
                    "thread_ts": "800",
                    "timestamp": "2025-06-27T15:00:00"
                },
                "expected_behavior": "Should create high-urgency escalation topic with account details and impact"
            },
            
            # TC017: CREATE - Vendor/invoice issue
            {
                "test_id": "TC017",
                "category": "CREATE",
                "test_name": "Vendor Invoice - New Topic",
                "expected_action": "create",
                "expected_topic_id": None,
                "message": {
                    "id": 2009,
                    "channel": "#finance",
                    "user": "Finance",
                    "text": "@accounting Invoice from AWS for June 2025 is $47,892.33 (expected: $35k). Investigating: EC2 instances running in eu-west-2 (unauthorized region). Need eng team to audit and terminate unused resources by July 10, 2025 to avoid overage.",
                    "thread_ts": "900",
                    "timestamp": "2025-06-28T10:00:00"
                },
                "expected_behavior": "Should create finance topic with invoice discrepancy and action items"
            },
            
            # TC018: CREATE - Policy/compliance update
            {
                "test_id": "TC018",
                "category": "CREATE",
                "test_name": "Compliance Update - New Topic",
                "expected_action": "create",
                "expected_topic_id": None,
                "message": {
                    "id": 2010,
                    "channel": "#legal-compliance",
                    "user": "Legal",
                    "text": "📋 COMPLIANCE: New GDPR requirement effective August 1, 2025 - must implement 'Right to Data Portability' API. Requirements: JSON/CSV export, all user data, within 30 days of request. Penalty: €20M or 4% revenue. Eng estimate: 6 weeks.",
                    "thread_ts": "1000",
                    "timestamp": "2025-06-28T13:30:00"
                },
                "expected_behavior": "Should create compliance topic with regulation, deadline, penalty, estimate"
            },
            
            # ========== DISCARD SCENARIOS ==========
            
            # TC019: DISCARD - Casual question without context
            {
                "test_id": "TC019",
                "category": "DISCARD",
                "test_name": "Casual Question - Should Discard",
                "expected_action": "discarded",
                "expected_topic_id": None,
                "message": {
                    "id": 3001,
                    "channel": "#general",
                    "user": "Alex",
                    "text": "Hey, I heard about the EcoBloom campaign. Can someone share the latest updates?",
                    "thread_ts": "5000",
                    "timestamp": "2025-06-29T10:00:00"
                },
                "expected_behavior": "Should discard - casual question without actionable content or context"
            },
            
            # TC020: DISCARD - Simple acknowledgment
            {
                "test_id": "TC020",
                "category": "DISCARD",
                "test_name": "Acknowledgment - Should Discard",
                "expected_action": "discarded",
                "expected_topic_id": None,
                "message": {
                    "id": 3002,
                    "channel": "#campaign-briefs",
                    "user": "Sam",
                    "text": "Thanks Devon! Got it 👍",
                    "thread_ts": "5001",
                    "timestamp": "2025-06-29T11:00:00"
                },
                "expected_behavior": "Should discard - simple acknowledgment without substance"
            },
            
            # TC021: DISCARD - Small talk
            {
                "test_id": "TC021",
                "category": "DISCARD",
                "test_name": "Small Talk - Should Discard",
                "expected_action": "discarded",
                "expected_topic_id": None,
                "message": {
                    "id": 3003,
                    "channel": "#general",
                    "user": "Maria",
                    "text": "Good morning everyone! Hope you all have a great Friday! 🎉",
                    "thread_ts": "5002",
                    "timestamp": "2025-06-29T08:00:00"
                },
                "expected_behavior": "Should discard - small talk with no business value"
            },
            
            # TC022: DISCARD - Vague reference
            {
                "test_id": "TC022",
                "category": "DISCARD",
                "test_name": "Vague Reference - Should Discard",
                "expected_action": "discarded",
                "expected_topic_id": None,
                "message": {
                    "id": 3004,
                    "channel": "#engineering",
                    "user": "Kevin",
                    "text": "What's the status on that bug?",
                    "thread_ts": "5003",
                    "timestamp": "2025-06-29T14:30:00"
                },
                "expected_behavior": "Should discard - vague, no context (which bug?)"
            },
            
            # TC023: DISCARD - Emoji reaction only
            {
                "test_id": "TC023",
                "category": "DISCARD",
                "test_name": "Emoji Only - Should Discard",
                "expected_action": "discarded",
                "expected_topic_id": None,
                "message": {
                    "id": 3005,
                    "channel": "#product-ideas",
                    "user": "Priya",
                    "text": "👍👍👍",
                    "thread_ts": "5004",
                    "timestamp": "2025-06-29T16:00:00"
                },
                "expected_behavior": "Should discard - emoji-only message with no content"
            },
            
            # ========== CRITICAL: NON-ACTIONABLE IN SAME THREAD ==========
            
            # TC026: CRITICAL - Acknowledgment in SAME thread (should NOT update)
            {
                "test_id": "TC026",
                "category": "UPDATE_NON_ACTIONABLE",
                "test_name": "CRITICAL: Acknowledgment in Same Thread - Should NOT Update",
                "expected_action": "discarded",  # Should discard even though same thread
                "expected_topic_id": None,
                "message": {
                    "id": 3006,
                    "channel": "#campaign-briefs",
                    "user": "Sam",
                    "text": "Thanks Devon! Got it 👍",
                    "thread_ts": "1",  # SAME THREAD as EcoBloom topic_001
                    "timestamp": "2025-06-29T17:00:00"
                },
                "expected_behavior": "Should DISCARD even though same thread - no actionable information to update topic"
            },
            
            # TC027: CRITICAL - Small talk in SAME thread (should NOT update)
            {
                "test_id": "TC027",
                "category": "UPDATE_NON_ACTIONABLE",
                "test_name": "CRITICAL: Small Talk in Same Thread - Should NOT Update",
                "expected_action": "discarded",
                "expected_topic_id": None,
                "message": {
                    "id": 3007,
                    "channel": "#campaign-briefs",
                    "user": "Leah",
                    "text": "Have a great weekend everyone! 🎉",
                    "thread_ts": "1",  # SAME THREAD as EcoBloom
                    "timestamp": "2025-06-29T18:00:00"
                },
                "expected_behavior": "Should DISCARD - social message adds no business value to topic"
            },
            
            # TC028: CRITICAL - Vague question in SAME thread (should NOT update)
            {
                "test_id": "TC028",
                "category": "UPDATE_NON_ACTIONABLE",
                "test_name": "CRITICAL: Vague Question in Same Thread - Should NOT Update",
                "expected_action": "discarded",
                "expected_topic_id": None,
                "message": {
                    "id": 3008,
                    "channel": "#campaign-briefs",
                    "user": "Jordan",
                    "text": "Any updates?",
                    "thread_ts": "1",  # SAME THREAD as EcoBloom
                    "timestamp": "2025-06-30T09:00:00"
                },
                "expected_behavior": "Should DISCARD - vague question without specific actionable request"
            },
            
            # ========== EDGE CASES ==========
            
            # TC024: EDGE CASE - Borderline topic-worthiness (question with partial context)
            {
                "test_id": "TC024",
                "category": "EDGE_CASE",
                "test_name": "Borderline Topic-Worthiness - Question with Some Context",
                "expected_action": "create",  # Or discarded, depending on LLM evaluation
                "expected_topic_id": None,
                "message": {
                    "id": 4001,
                    "channel": "#engineering",
                    "user": "Alex",
                    "text": "@backend-team Has anyone looked at the database slow query warnings from yesterday? I'm seeing some in the logs but not sure if it's urgent.",
                    "thread_ts": "6000",
                    "timestamp": "2025-06-30T09:00:00"
                },
                "expected_behavior": "Borderline - has some context (database, slow queries, logs) but lacks urgency/action"
            },
            
            # TC025: EDGE CASE - High semantic similarity but different structure
            {
                "test_id": "TC025",
                "category": "EDGE_CASE",
                "test_name": "High Semantic, Different Structure - Should Create",
                "expected_action": "create",
                "expected_topic_id": None,
                "message": {
                    "id": 4002,
                    "channel": "#general",  # Different channel
                    "user": "NewUser",
                    "text": "@team Starting work on EcoBloom summer campaign designs. Need access to brand guidelines and asset library. When is the kickoff meeting?",
                    "thread_ts": "7000",  # Different thread
                    "timestamp": "2025-06-30T10:30:00"
                },
                "expected_behavior": "Should CREATE (different channel/thread despite semantic similarity to topic_001)"
            }
        ]

    def run_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case"""
        print(f"\n{'='*80}")
        print(f"🧪 {test_case['test_id']}: {test_case['test_name']}")
        print(f"   Category: {test_case['category']}")
        print(f"{'='*80}")
        
        start_time = time.time()
        result = self.step6.process_new_message(test_case["message"])
        duration = time.time() - start_time
        
        # Extract metrics
        action = result.get("action", "")
        topic_id = result.get("topic_id", "")
        similar_topics = result.get("similar_topics", [])
        similarity_score = similar_topics[0][1] if similar_topics else 0.0
        
        # Normalize action
        if action in ["update", "updated"]:
            normalized_action = "update"
        elif action == "discarded":
            normalized_action = "discarded"
        else:
            normalized_action = "create"
        
        # Check if passed
        passed = (
            normalized_action == test_case["expected_action"] or
            (test_case["expected_action"] == "update" and normalized_action == "update") or
            (test_case["expected_action"] == "create" and normalized_action == "create") or
            (test_case["expected_action"] == "discarded" and normalized_action == "discarded")
        )
        
        # Print results
        status = "✅ PASS" if passed else "❌ FAIL"
        message = test_case["message"]
        
        print(f"\n   📨 MESSAGE:")
        print(f"      Channel: {message.get('channel', 'N/A')}")
        print(f"      User: {message.get('user', 'N/A')}")
        print(f"      Thread: {message.get('thread_ts', 'N/A')}")
        print(f"      Text: {message.get('text', 'N/A')[:100]}...")
        
        print(f"\n   {status}")
        print(f"      Expected: {test_case['expected_action'].upper()}")
        print(f"      Got: {normalized_action.upper()}")
        print(f"      Similarity: {similarity_score:.4f}")
        
        if normalized_action == "update":
            print(f"      Topic ID: {topic_id}")
        elif normalized_action == "discarded":
            print(f"      Reason: {result.get('reason', 'N/A')}")
        
        print(f"\n   ⏱️  Duration: {duration:.3f}s")
        print(f"   💡 Expected Behavior: {test_case['expected_behavior']}")
        
        test_result = {
            "test_id": test_case["test_id"],
            "category": test_case["category"],
            "test_name": test_case["test_name"],
            "expected_action": test_case["expected_action"],
            "actual_action": normalized_action,
            "passed": passed,
            "similarity_score": round(similarity_score, 4),
            "duration_seconds": round(duration, 3),
            "message": message,
            "expected_behavior": test_case["expected_behavior"],
            "result": result
        }
        
        return test_result

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive test cases"""
        print("🚀 STEP 6 COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print("📊 Running 28 test cases:")
        print("  - 8 UPDATE scenarios (with actionable info)")
        print("  - 10 CREATE scenarios")
        print("  - 5 DISCARD scenarios (new threads)")
        print("  - 3 UPDATE_NON_ACTIONABLE scenarios (CRITICAL TEST)")
        print("  - 2 EDGE CASES")
        print("=" * 80)
        
        test_cases = self.get_test_cases()
        all_results = []
        
        for test_case in test_cases:
            try:
                result = self.run_test_case(test_case)
                all_results.append(result)
                self.results.append(result)
            except Exception as e:
                print(f"❌ Test {test_case['test_id']} FAILED WITH ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "test_id": test_case["test_id"],
                    "category": test_case["category"],
                    "test_name": test_case["test_name"],
                    "error": str(e),
                    "passed": False
                })
        
        # Calculate metrics by category
        update_results = [r for r in all_results if r.get("category") == "UPDATE"]
        create_results = [r for r in all_results if r.get("category") == "CREATE"]
        discard_results = [r for r in all_results if r.get("category") == "DISCARD"]
        update_non_actionable_results = [r for r in all_results if r.get("category") == "UPDATE_NON_ACTIONABLE"]
        edge_results = [r for r in all_results if r.get("category") == "EDGE_CASE"]
        
        passed_update = len([r for r in update_results if r.get("passed")])
        passed_create = len([r for r in create_results if r.get("passed")])
        passed_discard = len([r for r in discard_results if r.get("passed")])
        passed_update_non_actionable = len([r for r in update_non_actionable_results if r.get("passed")])
        passed_edge = len([r for r in edge_results if r.get("passed")])
        
        total_passed = passed_update + passed_create + passed_discard + passed_update_non_actionable + passed_edge
        total_tests = len(all_results)
        
        aggregate_metrics = {
            "total_tests": total_tests,
            "passed_tests": total_passed,
            "failed_tests": total_tests - total_passed,
            "pass_rate": round(total_passed / total_tests * 100, 1) if total_tests else 0.0,
            "by_category": {
                "UPDATE": {"total": len(update_results), "passed": passed_update},
                "CREATE": {"total": len(create_results), "passed": passed_create},
                "DISCARD": {"total": len(discard_results), "passed": passed_discard},
                "UPDATE_NON_ACTIONABLE": {"total": len(update_non_actionable_results), "passed": passed_update_non_actionable},
                "EDGE_CASE": {"total": len(edge_results), "passed": passed_edge}
            },
            "total_duration": round(sum(r.get("duration_seconds", 0) for r in all_results), 3)
        }
        
        report = {
            "test_suite": "Step 6 Comprehensive Test Suite",
            "timestamp": datetime.now().isoformat(),
            "aggregate_metrics": aggregate_metrics,
            "test_results": all_results
        }
        
        # Save report
        report_file = os.path.join(self.output_dir, "comprehensive_test_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n{'='*80}")
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total Tests: {total_tests}")
        print(f"  ✅ Passed: {total_passed}")
        print(f"  ❌ Failed: {total_tests - total_passed}")
        print(f"  📈 Pass Rate: {aggregate_metrics['pass_rate']:.1f}%")
        print(f"\nBy Category:")
        print(f"  UPDATE:               {passed_update}/{len(update_results)} passed")
        print(f"  CREATE:               {passed_create}/{len(create_results)} passed")
        print(f"  DISCARD:              {passed_discard}/{len(discard_results)} passed")
        print(f"  UPDATE_NON_ACTIONABLE: {passed_update_non_actionable}/{len(update_non_actionable_results)} passed ⚠️  CRITICAL")
        print(f"  EDGE:                 {passed_edge}/{len(edge_results)} passed")
        print(f"\n⏱️  Total Duration: {aggregate_metrics['total_duration']:.3f}s")
        print(f"📁 Report saved to: {report_file}")
        
        return report


def main():
    """Run comprehensive test suite"""
    suite = ComprehensiveTestSuite()
    report = suite.run_all_tests()
    
    # Print failed tests details
    failed = [r for r in report["test_results"] if not r.get("passed")]
    if failed:
        print(f"\n{'='*80}")
        print("❌ FAILED TESTS DETAILS:")
        print(f"{'='*80}")
        for test in failed:
            print(f"\n{test['test_id']}: {test['test_name']}")
            print(f"   Expected: {test.get('expected_action', 'N/A')}")
            print(f"   Got: {test.get('actual_action', 'N/A')}")
            if "error" in test:
                print(f"   Error: {test['error']}")


if __name__ == "__main__":
    main()


