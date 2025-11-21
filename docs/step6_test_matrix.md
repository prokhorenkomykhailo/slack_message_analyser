# Step 6 Test Matrix - Quick Reference

## Test Case Summary Table

| ID | Category | Test Name | Channel | Thread | Expected | Key Test Point |
|----|----------|-----------|---------|--------|----------|----------------|
| **UPDATE SCENARIOS** |
| TC001 | UPDATE | Status Update - Bug Fix | #campaign-briefs | 1 | UPDATE | Task completion detection |
| TC002 | UPDATE | Deadline Extension | #campaign-briefs | 1 | UPDATE | Deadline change detection |
| TC003 | UPDATE | New Action Item | #campaign-briefs | 1 | UPDATE | Add item, keep existing |
| TC004 | UPDATE | Task Completion | #campaign-briefs | 1 | UPDATE | Status change to 'completed' |
| TC005 | UPDATE | Multiple Participants | #campaign-briefs | 1 | UPDATE | Multiple @mentions handling |
| TC006 | UPDATE | Technical Question | #campaign-briefs | 1 | UPDATE | Question WITH context = worthy |
| TC007 | UPDATE | Resource Request | #campaign-briefs | 1 | UPDATE | Budget approval with amount |
| TC008 | UPDATE | Blocker Notification | #campaign-briefs | 1 | UPDATE | Urgency upgrade logic |
| **CREATE SCENARIOS** |
| TC009 | CREATE | API Performance Issue | #engineering | 100 | CREATE | Different channel prevention |
| TC010 | CREATE | TechNova Campaign | #campaign-briefs | 200 | CREATE | Different thread, same channel |
| TC011 | CREATE | Bug Report | #bug-reports | 300 | CREATE | Bug metadata extraction |
| TC012 | CREATE | Feature Request | #product-ideas | 400 | CREATE | Requirements + ROI capture |
| TC013 | CREATE | Deployment | #deployments | 500 | CREATE | Version + changes tracking |
| TC014 | CREATE | Hiring | #hr-announcements | 600 | CREATE | Job details preservation |
| TC015 | CREATE | Security Incident | #security-alerts | 700 | CREATE | Incident ID + actions |
| TC016 | CREATE | Customer Escalation | #customer-escalations | 800 | CREATE | MRR + impact tracking |
| TC017 | CREATE | Invoice Issue | #finance | 900 | CREATE | Amount discrepancy |
| TC018 | CREATE | Compliance | #legal-compliance | 1000 | CREATE | Regulation + penalty |
| **DISCARD SCENARIOS** |
| TC019 | DISCARD | Casual Question | #general | 5000 | DISCARD | No context = not worthy |
| TC020 | DISCARD | Acknowledgment | #campaign-briefs | 5001 | DISCARD | "Thanks" filtered out |
| TC021 | DISCARD | Small Talk | #general | 5002 | DISCARD | Greetings filtered |
| TC022 | DISCARD | Vague Reference | #engineering | 5003 | DISCARD | "that bug" too vague |
| TC023 | DISCARD | Emoji Only | #product-ideas | 5004 | DISCARD | Content-less messages |
| **EDGE CASES** |
| TC024 | EDGE | Borderline Worthiness | #engineering | 6000 | CREATE/DISCARD | Partial context test |
| TC025 | EDGE | High Semantic, Diff Structure | #general | 7000 | CREATE | Structure > semantic |

---

## Test Coverage Matrix

### Structural Metadata Coverage

| Metadata Field | Tested In | What's Verified |
|----------------|-----------|-----------------|
| **Channel** (3x weight) | TC009, TC010, TC025 | Different channel → CREATE |
| **Thread** (3x weight) | TC010, TC025 | Different thread → CREATE |
| **From/To** (2x weight) | TC005, TC006, TC007 | Multiple participants handled |
| **Text** (1x weight) | All tests | Semantic content extracted |

### Action Item Logic Coverage

| Scenario | Tested In | Expected Behavior |
|----------|-----------|-------------------|
| Add new item | TC003 | Existing items preserved |
| Update status | TC001, TC004 | Specific item status changed |
| Multiple owners | TC005 | Separate tasks OR multi-owner task |
| Budget/amount | TC007, TC017 | Financial values preserved |
| Deadline | TC002 | Date updated correctly |

### Topic-Worthiness Coverage

| Message Type | Tested In | Expected Decision |
|--------------|-----------|-------------------|
| Actionable (tasks, deadlines) | TC001-TC018 | CREATE/UPDATE |
| Questions WITH context | TC006, TC024 | CREATE/UPDATE |
| Questions WITHOUT context | TC019, TC022 | DISCARD |
| Acknowledgments | TC020 | DISCARD |
| Small talk | TC021 | DISCARD |
| Reactions only | TC023 | DISCARD |

### Edge Case Coverage

| Edge Case | Tested In | System Behavior |
|-----------|-----------|-----------------|
| Same keywords, different structure | TC025 | Structural metadata prevents false match |
| Borderline context | TC024 | LLM judgment tested |
| Multiple @mentions | TC005 | All captured correctly |
| Urgency keywords | TC008 | Priority upgraded |
| Cross-team coordination | TC017 | Multiple teams tracked |

---

## Test Execution Checklist

### Pre-Test Setup
- [ ] Step 3 results available (`output/step3_client_analysis/google_gemini-2.0-flash.json`)
- [ ] Step 4 embeddings generated (Supabase or local)
- [ ] Google API key configured (`GOOGLE_API_KEY`)
- [ ] Test output directory exists (`output/step6_comprehensive_tests/`)

### Expected Results
- [ ] **UPDATE**: 8/8 passed (100%)
- [ ] **CREATE**: ≥9/10 passed (≥90%)
- [ ] **DISCARD**: 5/5 passed (100%)
- [ ] **EDGE**: ≥1/2 passed (≥50%)
- [ ] **Overall**: ≥23/25 passed (≥92%)

### Validation Points
- [ ] All UPDATE tests return correct `topic_id`
- [ ] All UPDATE tests have similarity ≥ 0.7
- [ ] All CREATE tests have similarity < 0.7
- [ ] All DISCARD tests have worthiness evaluation
- [ ] No false positives (CREATE when should UPDATE)
- [ ] No false negatives (DISCARD when should CREATE)

---

## Similarity Score Expectations

### UPDATE Scenarios (Same Thread)
| Test | Expected Similarity | Why |
|------|---------------------|-----|
| TC001-TC008 | **0.75 - 0.95** | Channel (3x) + Thread (3x) + Participants (2x) = High |

### CREATE Scenarios (Different Structure)
| Test | Expected Similarity | Why |
|------|---------------------|-----|
| TC009 | **0.00 - 0.30** | Different channel (3x) = Very Low |
| TC010 | **0.20 - 0.50** | Same channel, different thread (3x) = Low-Medium |
| TC011-TC018 | **0.00 - 0.40** | Different channel + thread = Low |
| TC025 | **0.40 - 0.60** | High semantic, different structure = Medium |

### DISCARD Scenarios
| Test | Expected Similarity | Action |
|------|---------------------|--------|
| TC019-TC023 | **Any** | Worthiness check fails → DISCARD |

---

## Debugging Guide

### If Similarity Too High (False UPDATE)
**Problem**: CREATE test incorrectly matches existing topic  
**Check**:
1. Embedding includes channel/thread with 3x repetition?
2. Different thread_ts values in message vs topic?
3. Step 4 embeddings regenerated after weighting changes?

**Example**:
```python
# Message embedding should look like:
"#general #general #general 7000 7000 7000 NewUser NewUser @team @team Starting work on EcoBloom..."
# NOT:
"#general 7000 NewUser @team Starting work on EcoBloom..."
```

### If Similarity Too Low (False CREATE)
**Problem**: UPDATE test incorrectly creates new topic  
**Check**:
1. Same thread_ts in message and topic?
2. Channel names match exactly (case-sensitive)?
3. Embedding loaded from Step 4 correctly?

**Debug**:
```python
print(f"Message thread: {message.get('thread_ts')}")
print(f"Topic thread: {topic.get('metadata', {}).get('thread_root')}")
print(f"Match: {message.get('thread_ts') == topic.get('metadata', {}).get('thread_root')}")
```

### If Worthiness Evaluation Fails
**Problem**: DISCARD test incorrectly creates topic  
**Check**:
1. LLM prompt clearly defines topic-worthy criteria
2. API key working (no fallback to conservative default)
3. Response parsing correctly (JSON extraction)

**Fallback Behavior**: If LLM fails, system assumes message IS worthy (conservative approach)

---

## Performance Expectations

### Execution Time
- **Single test**: 2-5 seconds (LLM calls)
- **Full suite (25 tests)**: 60-120 seconds
- **Bottleneck**: LLM API calls (2-3 per test)

### Optimization Opportunities
1. Batch LLM calls (parallel processing)
2. Cache embeddings (avoid regeneration)
3. Local LLM for faster evaluation (Ollama, Llama)

---

## Client Reporting Format

### Pass/Fail Summary
```
✅ UPDATE:  8/8  (100%) - Critical functionality working
✅ CREATE:  9/10 (90%)  - High accuracy on new topics
✅ DISCARD: 5/5  (100%) - Noise filtering working
⚠️  EDGE:    1/2  (50%)  - Borderline cases need review

🎯 OVERALL: 23/25 (92%) - PASS (Target: ≥90%)
```

### Failed Test Analysis
```
❌ TC025: High Semantic, Different Structure
   Expected: CREATE (similarity < 0.7)
   Got: UPDATE (similarity: 0.78)
   Root Cause: Keyword repetition ("EcoBloom") boosted similarity
   Fix: Increase structural weight or adjust threshold
```

---

## Next Actions After Testing

### If Pass Rate ≥ 90%
1. ✅ Mark Step 6 as production-ready
2. ✅ Run on real Slack data (100+ messages)
3. ✅ Monitor discard rate (<20% recommended)
4. ✅ Document any edge case exceptions

### If Pass Rate < 90%
1. ❌ Analyze failed tests category-wise
2. ❌ Adjust similarity threshold (try 0.65 or 0.75)
3. ❌ Refine embedding weighting ratios
4. ❌ Update LLM prompts for clarity
5. ❌ Re-run tests after fixes

---

## File Locations

- **Test Suite**: `phases/step6_comprehensive_test_suite.py`
- **Test Documentation**: `docs/step6_comprehensive_test_cases.md`
- **Test Matrix**: `docs/step6_test_matrix.md` (this file)
- **Output Report**: `output/step6_comprehensive_tests/comprehensive_test_report.json`
- **Step 6 Code**: `phases/step6_new_message_processing.py`

---

## Quick Start

```bash
# Run comprehensive tests
cd /home/ubuntu/vertex-set/deemerge/phase_evaluation_engine
python3 phases/step6_comprehensive_test_suite.py

# View results
cat output/step6_comprehensive_tests/comprehensive_test_report.json | jq '.aggregate_metrics'

# Check failed tests
cat output/step6_comprehensive_tests/comprehensive_test_report.json | jq '.test_results[] | select(.passed == false)'
```


