# Client Questions - Status & Testing Date Explanation

## Question 1: What's the difference between "active" and "in_progress" status?

There are **two types of status** in the Excel files:

### A) Topic/Cluster Level Status (Overall Project Status)
This appears at the topic level and shows the overall state of the entire project or conversation thread:

- **`active`** = Project is currently running and requires immediate attention
  - Work is happening now
  - Deadlines are approaching
  - Team is actively engaged
  - Example: "EcoBloom Summer Campaign" with deadline July 28, 2025
  
- **`in_progress`** = Project has started and work is underway
  - Tasks are being completed
  - Progress is being made
  - May not require immediate action
  - Example: "FitFusion Rebranding" with some tasks completed, others pending
  
- **`completed`** = Project is finished
  - All tasks done
  - Delivered to client
  - Example: "Q3 Content Calendar" completed ahead of deadline

### B) Action Item Status (Individual Task Status)
This appears for each individual task/action item:

- **`pending`** = Task not started yet, waiting to begin
- **`in_progress`** = Task is currently being worked on
- **`completed`** = Task is finished

### Summary Comparison:

| Level | Status | Meaning |
|-------|--------|---------|
| **Topic** | active | Urgent, needs immediate attention right now |
| **Topic** | in_progress | Work happening, but less urgent |
| **Topic** | completed | All done |
| **Action Item** | pending | Not started |
| **Action Item** | in_progress | Being worked on |
| **Action Item** | completed | Finished |

### Real Example from Your Data:

**Topic: "EcoBloom Summer Campaign"**
- Topic Status: `active` (urgent, deadline approaching July 28, 2025)
- Action Item 1: `pending` (not started yet)
- Action Item 2: `in_progress` (Sam is working on designs)
- Action Item 3: `completed` (already finished)

---

## Question 2: What is the current date used for the testing?

### Test Execution Date:
The AI models were tested on: **October 19, 2025**
- Timestamp: `2025-10-19T23:51:04`

### Simulated Message Timeframe:
The synthetic Slack messages (test data) span from:
- **Start Date:** June 15, 2025
- **End Date:** September 15, 2025

### Context for Understanding the Data:
The messages and conversations simulate a marketing agency's work over a 3-month period (June-September 2025). The AI models analyzed these historical conversations as if they were real workplace messages.

**Important Notes:**
1. All dates in the action items (due dates, deadlines) are from the simulated timeframe (June-September 2025)
2. These dates represent when tasks were due in the simulated scenario
3. The AI models were evaluated on their ability to extract and organize this date information correctly

### Why This Matters:
When you see dates like "2025-07-28" in the Excel files, these are:
- Dates mentioned in the original messages
- Deadlines for projects in the simulation
- NOT the date the test was run (which was October 19, 2025)

---

## Summary for Quick Reference:

| Question | Answer |
|----------|--------|
| **Difference between active/in_progress?** | `active` = urgent, needs immediate attention; `in_progress` = work happening, less urgent |
| **Testing date?** | October 19, 2025 (when models were tested) |
| **Message dates?** | June 15 - September 15, 2025 (simulated timeframe) |
| **Due dates in files?** | From the simulated messages (June-Sept 2025) |

---

If you have any other questions about the data or need clarification, please let me know!

