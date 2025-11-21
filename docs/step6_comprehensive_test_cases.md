# Step 6 Comprehensive Test Cases - Technical Documentation

## Overview

This document details 25 realistic, technical test cases covering ALL scenarios for Step 6 (New Message Processing). These tests verify the system's ability to correctly UPDATE existing topics, CREATE new topics, and DISCARD meaningless messages.

---

## Test Suite Structure

**Total: 25 Test Cases**
- **8 UPDATE scenarios**: Messages that should update existing topics (same thread/channel)
- **10 CREATE scenarios**: Messages that should create new topics (different structure/context)
- **5 DISCARD scenarios**: Messages that should be discarded (not topic-worthy)
- **2 EDGE CASES**: Borderline scenarios testing system robustness

---

## UPDATE Scenarios (TC001-TC008)

### TC001: Status Update - Bug Fix Progress
**Scenario**: Developer completes a task and provides status update in existing thread  
**Message**: 
```
Channel: #campaign-briefs
User: Sam
Thread: 1 (same as EcoBloom topic_001)
Text: "@Devon I've completed the design mockups for EcoBloom. Ready for your review. 
       Files are in Figma: https://figma.com/ecobloom-designs"
```
**Expected**: UPDATE topic_001  
**What it tests**: 
- Thread matching works correctly
- Status completion is detected
- Action item status changes from 'pending' to 'completed'
- Existing action items are preserved

---

### TC002: Deadline Extension - Same Thread
**Scenario**: Project manager extends deadline in existing project thread  
**Message**:
```
Channel: #campaign-briefs
User: Devon
Thread: 1 (same as EcoBloom)
Text: "@team Update: The EcoBloom campaign deadline has been extended to August 15, 2025 
       due to client request. Please adjust your timelines."
```
**Expected**: UPDATE topic_001  
**What it tests**:
- Deadline changes are detected and updated
- Summary is updated to reflect new information
- All existing metadata is preserved
- Thread weighting (3x) ensures high similarity

---

### TC003: New Action Item - Same Thread
**Scenario**: New task added to existing project thread  
**Message**:
```
Channel: #campaign-briefs
User: Leah
Thread: 1
Text: "@Jordan Can you also review the legal compliance for EU GDPR requirements? 
       We need this by July 20, 2025 before launch."
```
**Expected**: UPDATE topic_001  
**What it tests**:
- New action items are added WITHOUT removing existing ones
- Owner, due date, and task are extracted correctly
- Action item list grows (not replaced)
- Critical: Tests action item preservation logic

---

### TC004: Task Completion Notification
**Scenario**: Team member marks their task as done  
**Message**:
```
Channel: #campaign-briefs
User: Leah
Thread: 1
Text: "@Devon Content strategy is done! I've uploaded all materials to Google Drive. 
       Ready for review."
```
**Expected**: UPDATE topic_001  
**What it tests**:
- Completion signals ("done", "completed", "finished") are detected
- Specific user's task status changes to 'completed'
- Other users' tasks remain unchanged
- LLM prompt for selective action item updates works

---

### TC005: Multiple Participants - Same Thread
**Scenario**: Meeting scheduled with multiple team members  
**Message**:
```
Channel: #campaign-briefs
User: Devon
Thread: 1
Text: "@sam @leah @jordan Team sync tomorrow at 2pm to review EcoBloom progress. 
       Please bring your status updates."
```
**Expected**: UPDATE topic_001  
**What it tests**:
- Multiple @mentions are captured
- New action item with multiple owners OR separate tasks per owner
- Participants field updated with all mentioned users
- From/to weighting (2x) works correctly

---

### TC006: Technical Question - Same Thread
**Scenario**: Question requiring decision in project thread (topic-worthy)  
**Message**:
```
Channel: #campaign-briefs
User: Sam
Thread: 1
Text: "@Devon Should we use the old EcoBloom brand colors (#2D5016, #8BC34A) or the 
       new rebrand palette for this campaign? This affects all my designs."
```
**Expected**: UPDATE topic_001  
**What it tests**:
- Questions WITH context are topic-worthy (not discarded)
- Decision needed is captured as action item
- Technical details (color codes) preserved
- Same thread = UPDATE even if it's a question

---

### TC007: Resource Request - Same Thread
**Scenario**: Budget approval requested in project thread  
**Message**:
```
Channel: #campaign-briefs
User: Jordan
Thread: 1
Text: "@Devon I need budget approval for $5,000 to hire an external copywriter for 
       the EcoBloom landing page. Our internal team is at capacity."
```
**Expected**: UPDATE topic_001  
**What it tests**:
- Financial amounts are preserved
- Budget approval becomes action item
- Justification is captured in summary
- Urgency may increase based on constraint ("at capacity")

---

### TC008: Blocker Notification - Same Thread
**Scenario**: Critical blocker reported with urgency  
**Message**:
```
Channel: #campaign-briefs
User: Sam
Thread: 1
Text: "@Devon URGENT: EcoBloom client hasn't approved the design concepts yet. 
       This is blocking development. Can you follow up with them ASAP?"
```
**Expected**: UPDATE topic_001  
**What it tests**:
- Urgency keywords ("URGENT", "blocking", "ASAP") increase priority
- Blocker becomes high-priority action item
- Urgency level upgraded (medium → high → critical)
- Risk/dependency information captured

---

## CREATE Scenarios (TC009-TC018)

### TC009: Different Channel - API Performance Issue
**Scenario**: Critical bug in different channel than EcoBloom  
**Message**:
```
Channel: #engineering (different)
User: Alex
Thread: 100 (different)
Text: "@backend-team We have a critical API performance issue in production. 
       Response times are 5-8 seconds (should be <500ms). User ID 47291 reported timeouts. 
       Need root cause analysis by EOD."
```
**Expected**: CREATE new topic  
**What it tests**:
- Different channel (3x weight) prevents false match to EcoBloom
- Technical metrics preserved (5-8 sec, <500ms, user ID)
- Urgency correctly set to 'critical'
- Action item with specific deadline (EOD)

---

### TC010: Same Channel, Different Thread - TechNova Campaign
**Scenario**: New campaign in same channel as EcoBloom  
**Message**:
```
Channel: #campaign-briefs (same)
User: Maria
Thread: 200 (different)
Text: "@creative-team Launching new Q3 campaign for TechNova Software (tech startup client). 
       Need logo design, website mockups, and social media assets by July 30, 2025. 
       Budget: $15,000."
```
**Expected**: CREATE new topic  
**What it tests**:
- Different thread (3x weight) prevents match despite same channel
- New client name (TechNova vs EcoBloom) creates semantic difference
- Multiple deliverables captured as separate action items
- Budget amount preserved

---

### TC011: Bug Report - Authentication Issue
**Scenario**: Bug report with reproduction steps  
**Message**:
```
Channel: #bug-reports
User: Priya
Thread: 300
Text: "🐛 BUG: User authentication fails intermittently on Safari 17.2 (macOS Sonoma). 
       Error: 'Invalid CSRF token'. Affects ~500 users/day. 
       Reproduction steps: 1) Open app in Safari 2) Clear cookies 3) Try to log in. 
       Priority: HIGH"
```
**Expected**: CREATE new topic  
**What it tests**:
- Bug metadata extracted (browser, OS, error message)
- Impact quantified (500 users/day)
- Reproduction steps preserved
- Priority explicitly set
- Topic type: bug/incident

---

### TC012: Feature Request - Bulk User Import
**Scenario**: Feature request with requirements and ROI  
**Message**:
```
Channel: #product-ideas
User: Kevin
Thread: 400
Text: "@product-team Feature request from 15 enterprise clients: Bulk user import via CSV. 
       Requirements: Support 10k+ users, role assignment, custom fields mapping, 
       validation errors report. Est. dev time: 3 sprints. ROI: $50k ARR."
```
**Expected**: CREATE new topic  
**What it tests**:
- Feature requirements captured
- Business justification (15 clients, $50k ARR)
- Technical estimates preserved (3 sprints)
- Multiple requirements as separate action items or bullet points

---

### TC013: Deployment Notification
**Scenario**: Production deployment scheduled  
**Message**:
```
Channel: #deployments
User: DevOps
Thread: 500
Text: "🚀 DEPLOYMENT SCHEDULED: v2.5.0 to production on July 5, 2025 at 2am UTC 
       (maintenance window). Changes: new payment gateway, database migration, API v3 rollout. 
       Estimated downtime: 30min. Rollback plan ready."
```
**Expected**: CREATE new topic  
**What it tests**:
- Version number, date, time preserved
- Multiple changes captured
- Downtime estimate included
- Rollback plan mentioned (risk mitigation)

---

### TC014: Hiring Announcement
**Scenario**: Job opening posted  
**Message**:
```
Channel: #hr-announcements
User: HR
Thread: 600
Text: "@team Opening: Senior Backend Engineer (Python/Django). Location: Remote US. 
       Salary: $140k-$180k. Requirements: 5+ years Python, microservices, Kubernetes. 
       Applications due July 20, 2025. Refer qualified candidates for $3k bonus."
```
**Expected**: CREATE new topic  
**What it tests**:
- Job details preserved (title, location, salary range)
- Requirements list captured
- Deadline for applications
- Referral bonus information

---

### TC015: Security Incident
**Scenario**: Security alert with incident ID  
**Message**:
```
Channel: #security-alerts
User: Security
Thread: 700
Text: "⚠️ SECURITY: Detected unusual API traffic from IP 185.220.101.x (TOR exit node). 
       10k requests/min to /api/users endpoint. Possible data scraping attempt. 
       Actions: 1) Rate limiting enabled 2) IP blocked 3) Investigating data exposure. 
       Incident ID: SEC-2025-047"
```
**Expected**: CREATE new topic  
**What it tests**:
- Security indicators preserved (IP, endpoint, attack type)
- Actions taken captured
- Incident ID tracked
- Urgency set to critical
- Tags include security-related terms

---

### TC016: Customer Escalation
**Scenario**: Enterprise customer threatening churn  
**Message**:
```
Channel: #customer-escalations
User: Support
Thread: 800
Text: "🔥 ESCALATION: Enterprise customer AcmeCorp (account ID: 12847, MRR: $25k/mo) 
       threatening to churn. Issue: Data export feature broken for 3 weeks, no response 
       from eng team. CEO demanding call with our CEO. Need immediate fix + exec response."
```
**Expected**: CREATE new topic  
**What it tests**:
- Customer details preserved (account ID, MRR)
- Issue duration tracked (3 weeks)
- Escalation level detected
- Multiple action items (fix + exec call)
- Financial impact considered (MRR: $25k/mo)

---

### TC017: Vendor Invoice Issue
**Scenario**: Invoice discrepancy investigation  
**Message**:
```
Channel: #finance
User: Finance
Thread: 900
Text: "@accounting Invoice from AWS for June 2025 is $47,892.33 (expected: $35k). 
       Investigating: EC2 instances running in eu-west-2 (unauthorized region). 
       Need eng team to audit and terminate unused resources by July 10, 2025 to avoid overage."
```
**Expected**: CREATE new topic  
**What it tests**:
- Financial amounts preserved (actual vs expected)
- Root cause investigation captured
- Technical details (AWS, EC2, region)
- Action item with deadline
- Cross-team coordination (finance → engineering)

---

### TC018: Compliance Update
**Scenario**: New regulatory requirement  
**Message**:
```
Channel: #legal-compliance
User: Legal
Thread: 1000
Text: "📋 COMPLIANCE: New GDPR requirement effective August 1, 2025 - must implement 
       'Right to Data Portability' API. Requirements: JSON/CSV export, all user data, 
       within 30 days of request. Penalty: €20M or 4% revenue. Eng estimate: 6 weeks."
```
**Expected**: CREATE new topic  
**What it tests**:
- Regulatory deadline captured
- Requirements detailed
- Penalty information preserved
- Engineering estimate included
- High urgency due to financial risk

---

## DISCARD Scenarios (TC019-TC023)

### TC019: Casual Question - No Context
**Scenario**: Vague question referencing existing topic  
**Message**:
```
Channel: #general
User: Alex
Thread: 5000
Text: "Hey, I heard about the EcoBloom campaign. Can someone share the latest updates?"
```
**Expected**: DISCARDED  
**What it tests**:
- Questions WITHOUT context are not topic-worthy
- No actionable information
- No deadlines or commitments
- Should not create duplicate/noise topics
- LLM correctly identifies casual nature

---

### TC020: Simple Acknowledgment
**Scenario**: "Thanks" message  
**Message**:
```
Channel: #campaign-briefs
User: Sam
Thread: 5001
Text: "Thanks Devon! Got it 👍"
```
**Expected**: DISCARDED  
**What it tests**:
- Acknowledgments are not topic-worthy
- No business value
- Very short messages filtered
- Emoji-only content discarded

---

### TC021: Small Talk
**Scenario**: Greeting message  
**Message**:
```
Channel: #general
User: Maria
Thread: 5002
Text: "Good morning everyone! Hope you all have a great Friday! 🎉"
```
**Expected**: DISCARDED  
**What it tests**:
- Social messages filtered out
- No project/work content
- Maintains signal-to-noise ratio

---

### TC022: Vague Reference
**Scenario**: Question without specifics  
**Message**:
```
Channel: #engineering
User: Kevin
Thread: 5003
Text: "What's the status on that bug?"
```
**Expected**: DISCARDED  
**What it tests**:
- Vague references ("that bug") are not topic-worthy
- No context about which bug
- Cannot generate meaningful metadata
- Forces users to provide context

---

### TC023: Emoji Only
**Scenario**: Reaction with only emojis  
**Message**:
```
Channel: #product-ideas
User: Priya
Thread: 5004
Text: "👍👍👍"
```
**Expected**: DISCARDED  
**What it tests**:
- Content-less messages filtered
- No text to analyze
- Pure reactions don't create topics

---

## EDGE CASES (TC024-TC025)

### TC024: Borderline Topic-Worthiness
**Scenario**: Question with SOME context but incomplete  
**Message**:
```
Channel: #engineering
User: Alex
Thread: 6000
Text: "@backend-team Has anyone looked at the database slow query warnings from yesterday? 
       I'm seeing some in the logs but not sure if it's urgent."
```
**Expected**: CREATE or DISCARDED (depends on LLM evaluation)  
**What it tests**:
- Borderline cases tested
- Has some context (database, slow queries, logs)
- But lacks urgency/action
- Tests LLM judgment threshold
- System robustness for unclear cases

---

### TC025: High Semantic, Different Structure
**Scenario**: Similar keywords but completely different context  
**Message**:
```
Channel: #general (different from #campaign-briefs)
User: NewUser
Thread: 7000 (different from 1)
Text: "@team Starting work on EcoBloom summer campaign designs. Need access to brand 
       guidelines and asset library. When is the kickoff meeting?"
```
**Expected**: CREATE new topic  
**What it tests**:
- High semantic similarity ("EcoBloom", "summer", "campaign", "designs")
- But different channel (3x weight) and thread (3x weight)
- Structural metadata prevents false match
- Tests weighting effectiveness
- Verifies channel/thread > semantic content

---

## Test Execution

### Running the Tests

```bash
cd /home/ubuntu/vertex-set/deemerge/phase_evaluation_engine
python3 phases/step6_comprehensive_test_suite.py
```

### Expected Output

```
🚀 STEP 6 COMPREHENSIVE TEST SUITE
================================================================================
📊 Running 25 test cases:
  - 8 UPDATE scenarios
  - 10 CREATE scenarios
  - 5 DISCARD scenarios
  - 2 EDGE CASES
================================================================================

[Individual test results...]

================================================================================
📊 COMPREHENSIVE TEST SUMMARY
================================================================================
Total Tests: 25
  ✅ Passed: 23
  ❌ Failed: 2
  📈 Pass Rate: 92.0%

By Category:
  UPDATE:  8/8 passed
  CREATE:  9/10 passed
  DISCARD: 5/5 passed
  EDGE:    1/2 passed

⏱️  Total Duration: 125.3s
📁 Report saved to: output/step6_comprehensive_tests/comprehensive_test_report.json
```

---

## Success Criteria

### Minimum Passing Rates
- **UPDATE scenarios**: 100% (8/8) - Critical for existing topic accuracy
- **CREATE scenarios**: ≥90% (9/10) - Allow 1 borderline case variation
- **DISCARD scenarios**: 100% (5/5) - Critical for noise reduction
- **EDGE CASES**: ≥50% (1/2) - Acceptable variation for borderline cases

### Overall Target: ≥90% pass rate (23/25 tests)

---

## Validation Points

### For UPDATE Tests:
1. ✅ Similarity score ≥ 0.7 (thread/channel match)
2. ✅ Correct topic_id returned
3. ✅ Existing action items preserved
4. ✅ New action items added (not replaced)
5. ✅ Status changes detected correctly
6. ✅ Deadline/metadata updates applied

### For CREATE Tests:
1. ✅ Similarity score < 0.7 (structural mismatch)
2. ✅ New topic_id generated
3. ✅ All metadata fields populated
4. ✅ Action items extracted with owner/due date
5. ✅ Urgency correctly assessed
6. ✅ Tags/keywords captured

### For DISCARD Tests:
1. ✅ Message evaluated for topic-worthiness
2. ✅ Discarded with clear reason
3. ✅ No topic created
4. ✅ Reason logged for audit

---

## Troubleshooting

### If UPDATE tests fail:
- Check embedding weighting (channel 3x, thread 3x, from/to 2x)
- Verify thread_ts matching logic
- Inspect action item update prompt

### If CREATE tests fail:
- Check similarity threshold (0.7)
- Verify structural metadata differences
- Ensure channel/thread mismatches lower similarity

### If DISCARD tests fail:
- Review topic-worthiness LLM prompt
- Check for false negatives (discarding valid messages)
- Adjust evaluation criteria if needed

---

## Client Reporting

The comprehensive test report (`comprehensive_test_report.json`) includes:
- Individual test results with pass/fail status
- Similarity scores for each message
- Actual vs expected actions
- Execution time per test
- Detailed error messages for failures
- Category-wise breakdown
- Overall pass rate percentage

This report can be shared directly with the client to demonstrate system robustness.

---

## Next Steps

1. **Run the test suite** and review results
2. **Adjust threshold** if needed (currently 0.7)
3. **Refine LLM prompts** based on failures
4. **Add more edge cases** as discovered in production
5. **Validate with real Slack data** (100+ messages)

---

## Questions?

Contact the development team if you need:
- Additional test cases
- Custom validation criteria
- Performance optimization
- Real-world data validation


