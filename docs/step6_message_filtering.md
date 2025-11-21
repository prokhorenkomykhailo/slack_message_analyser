# Step 6: Message Filtering - Topic Worthiness Evaluation

## Overview

We've added intelligent message filtering to Step 6 to prevent creating topics from meaningless or casual messages that don't contain substantial information.

---

## Problem Statement

**Client Feedback:**
> "We do not create a topic with meaningless messages if they don't match the definition of a topic. We have to consider discarding messages that don't contain meaningful data. 'I heard about the EcoBloom campaign' - that is not very useful to anyone, and we won't create a topic with just that message. It doesn't meet the definition of a topic."

---

## Solution: Topic-Worthiness Evaluation

### New Processing Flow

```
New Message
    ↓
Generate Embedding
    ↓
Find Similar Topics (cosine similarity ≥ 0.7)
    ↓
┌─────────────────────────────┐
│ Similar Topics Found?       │
├─────────────────────────────┤
│ YES → UPDATE existing topic │
│ NO  → Check topic-worthiness│
└─────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Is Message Topic-Worthy?            │
├─────────────────────────────────────┤
│ YES → CREATE new topic              │
│ NO  → DISCARD (don't create topic)  │
└─────────────────────────────────────┘
```

---

## Definition of a Topic-Worthy Message

### ✅ SHOULD CREATE TOPIC (Topic-Worthy)

A message is topic-worthy if it contains **at least one** of these:

1. **Actionable Information**
   - Tasks, deadlines, action items
   - Decisions, commitments
   - Requests for work, approvals, resources

2. **Important Announcements**
   - New projects or initiatives
   - Policy changes
   - Hiring announcements
   - Major updates or milestones

3. **Substantial Information Exchange**
   - Detailed explanations or reports
   - Data, analysis, findings
   - Technical documentation

4. **Requests with Context**
   - Budget approvals with amounts and deadlines
   - Resource requests with justification
   - Work requests with specifications

---

### ❌ SHOULD DISCARD (NOT Topic-Worthy)

Messages that should **NOT** create topics:

1. **Casual Questions Without Context**
   - "I heard about X, can someone share updates?"
   - "What's the status on Y?"
   - "Anyone know about Z?"

2. **Simple Acknowledgments**
   - "Thanks!"
   - "Got it"
   - "OK"
   - "👍"

3. **Small Talk / Chit-Chat**
   - "Good morning everyone"
   - "Have a great weekend!"
   - Casual conversations

4. **Vague References**
   - Messages that mention existing topics but add no new information
   - Questions without sufficient context to generate meaningful metadata

---

## Implementation Details

### New Function: `is_message_topic_worthy()`

**File**: `phases/step6_new_message_processing.py`

This function uses an LLM to evaluate if a message contains enough substance to create a topic.

**Returns**:
```json
{
  "is_worthy": true/false,
  "reason": "Brief explanation",
  "suggested_action": "create" or "discard"
}
```

**Fallback Strategy**: If evaluation fails (API error, timeout), the system assumes the message is worthy to avoid losing potentially important messages.

---

## Example: Message Filtering in Action

### Example 1: DISCARD (Not Topic-Worthy)

**Message**:
```json
{
  "channel": "#general",
  "user": "Alex",
  "text": "I heard about the EcoBloom campaign. Can someone share the latest updates?"
}
```

**Evaluation**:
- ❌ No actionable information (no tasks, deadlines, decisions)
- ❌ Not an announcement (just asking about existing topic)
- ❌ No substantial information (just a casual question)
- ❌ Lacks context (doesn't specify what updates needed, why, or for what purpose)

**Result**: **DISCARDED** (not topic-worthy)

**Reason**: "This is a casual question without context. It doesn't contain actionable information, announcements, or substantial content to create a meaningful topic."

---

### Example 2: CREATE (Topic-Worthy)

**Message**:
```json
{
  "channel": "#finance-updates",
  "user": "Priya",
  "text": "@finance-team The Q3 invoice for Horizon Robotics (PO-HR-7784) is still unpaid. Please confirm that the wire transfer is scheduled before the July 3, 2025 cutoff to avoid penalties."
}
```

**Evaluation**:
- ✅ Actionable information (confirm wire transfer)
- ✅ Has deadline (July 3, 2025)
- ✅ Important announcement (unpaid invoice, potential penalties)
- ✅ Sufficient context (invoice number, client, cutoff date)

**Result**: **CREATE NEW TOPIC**

**Topic Generated**:
```json
{
  "title": "Unpaid Q3 Invoice for Horizon Robotics (PO-HR-7784)",
  "summary": "Priya notified the finance team about the unpaid Q3 invoice...",
  "action_items": [
    {
      "task": "Confirm wire transfer schedule for Horizon Robotics invoice",
      "owner": "@finance-team",
      "due_date": "2025-07-03",
      "priority": "high",
      "status": "pending"
    }
  ],
  "deadline": "2025-07-03",
  "urgency": "high"
}
```

---

## Test Suite Updates

### New Test Case: TC011

**Test ID**: TC011  
**Test Name**: Discard Meaningless Message (Casual Question)  
**Expected Action**: DISCARDED

**Message**:
```json
{
  "id": 311,
  "channel": "#general",
  "user": "Alex",
  "text": "I heard about the EcoBloom campaign. Can someone share the latest updates?",
  "thread_ts": "500",
  "timestamp": "2025-07-05T10:00:00"
}
```

**Validation Criteria**:
- Should be discarded
- Reason should contain: "casual", "question", "without context", "not topic-worthy"

---

## Statistics Tracking

### Updated Metrics

The system now tracks:
- **Updated Topics**: Messages that matched existing topics (similarity ≥ 0.7)
- **Created Topics**: Messages that created new topics (similarity < 0.7 AND topic-worthy)
- **Discarded Messages**: Messages that were filtered out (similarity < 0.7 AND NOT topic-worthy)

**Example Output**:
```
✅ Updated 2 existing topics
✅ Created 5 new topics
🗑️  Discarded 3 messages (not topic-worthy)
```

---

## Benefits

1. **Reduced Noise**: No topics created from casual questions or chit-chat
2. **Better Quality**: Topics contain only meaningful, actionable information
3. **Clearer Insights**: Easier to identify important conversations
4. **Cost Savings**: Fewer topics = less storage and processing overhead
5. **User Experience**: Users see only relevant, substantial topics

---

## Configuration

### Threshold Settings

- **Similarity Threshold**: 0.7 (70% similarity required for UPDATE)
- **Topic-Worthiness**: Evaluated by LLM (Gemini 2.0 Flash)
- **Fallback**: Conservative approach - assume worthy if evaluation fails

### Adjustable Parameters

You can adjust these settings in `phases/step6_new_message_processing.py`:
- `threshold` parameter in `find_similar_topics()` (default: 0.7)
- LLM prompt in `is_message_topic_worthy()` (add/remove criteria)
- Fallback behavior (currently assumes worthy to be safe)

---

## Next Steps

1. **Review test results** for TC011 (DISCARD test case)
2. **Validate filtering criteria** against real Slack data
3. **Adjust LLM prompt** if needed to refine filtering logic
4. **Monitor discard rate** to ensure not too many messages are filtered out

---

## Questions?

If you need to adjust the filtering criteria or have concerns about specific message types, please let us know.

