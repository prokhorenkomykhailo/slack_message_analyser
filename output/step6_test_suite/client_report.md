# Step 6 Test Results - Client Report

**Test Date**: 2025-11-19 13:00:56

## Test Overview

- **Total Tests**: 10
- **UPDATE Tests**: 5
- **CREATE Tests**: 5

## Test Results

### TC001: Update EcoBloom Campaign Deadline

**Status**: ✅ PASS

#### New Message:
- **ID**: 301
- **Channel**: #campaign-briefs
- **User**: Devon
- **Timestamp**: 2025-06-25T10:00:00
- **Text**: @team The EcoBloom campaign deadline has been moved to August 5, 2025. Please update your timelines accordingly.

#### Test Results:
- **Expected Action**: UPDATE
- **Actual Action**: UPDATE
- **Similarity Score**: 0.8022
- **Duration**: 8.302s

#### Original Topic (Before Update):
- **Topic ID**: topic_001
- **Title**: EcoBloom Summer Campaign Planning
- **Summary**: The team is planning the EcoBloom summer campaign with a deadline of July 28, 2025. @Leah will provide content by June 30, 2025, @Sam designs by July 10, 2025, and @Jordan legal reviews by July 14, 2025.
- **Participants**: @Devon, @Sam, @Leah, @Jordan
- **Action Items**: 5 items
- **Deadline**: 2025-07-28
- **Urgency**: high

#### Updated Topic (After Update):
- **Title**: EcoBloom Summer Campaign Planning
- **Summary**: The team is planning the EcoBloom summer campaign with a deadline of August 5, 2025. @Leah will provide content by June 30, 2025, @Sam designs by July 10, 2025, and @Jordan legal reviews by July 14, 2025. The campaign deadline has been moved to August 5, 2025.
- **Participants**: @Devon, @Sam, @Leah, @Jordan
- **Action Items**: 9 items
- **Deadline**: 2025-08-05
- **Urgency**: high

---

### TC002: Update EcoBloom Content Status

**Status**: ✅ PASS

#### New Message:
- **ID**: 302
- **Channel**: #campaign-briefs
- **User**: Leah
- **Timestamp**: 2025-06-27T14:30:00
- **Text**: @sam @jordan I've completed the first draft of the EcoBloom content. Please review by end of week. The tone is more casual than the initial brief requested.

#### Test Results:
- **Expected Action**: UPDATE
- **Actual Action**: UPDATE
- **Similarity Score**: 0.7624
- **Duration**: 7.163s

#### Original Topic (Before Update):
- **Topic ID**: topic_001
- **Title**: EcoBloom Summer Campaign Planning
- **Summary**: The team is planning the EcoBloom summer campaign with a deadline of July 28, 2025. @Leah will provide content by June 30, 2025, @Sam designs by July 10, 2025, and @Jordan legal reviews by July 14, 2025.
- **Participants**: @Devon, @Sam, @Leah, @Jordan
- **Action Items**: 5 items
- **Deadline**: 2025-07-28
- **Urgency**: high

#### Updated Topic (After Update):
- **Title**: EcoBloom Summer Campaign Planning
- **Summary**: The team is planning the EcoBloom summer campaign with a deadline of July 28, 2025. @Leah will provide content by June 30, 2025, @Sam designs by July 10, 2025, and @Jordan legal reviews by July 14, 2025. @Leah has completed the first draft of the EcoBloom content and requests @sam and @jordan to review it by the end of the week. The tone is more casual than initially requested.
- **Participants**: @Devon, @Sam, @Leah, @Jordan
- **Action Items**: 7 items
- **Deadline**: 2025-07-28
- **Urgency**: high

---

### TC003: Fallback Rule: Same Thread + Participants, Different Topic

**Status**: ❌ FAIL

#### New Message:
- **ID**: 303
- **Channel**: #campaign-briefs
- **User**: Sam
- **Timestamp**: 2025-06-28T09:00:00
- **Text**: @devon @leah Quick question about the meeting time tomorrow. Can we move it to 3pm instead?

#### Test Results:
- **Expected Action**: UPDATE
- **Actual Action**: CREATE
- **Similarity Score**: 0.0000
- **Duration**: 3.235s

#### New Topic Created:
- **Topic ID**: topic_016
- **Title**: Meeting Time Change Request - Campaign Brief Meeting
- **Summary**: Sam is requesting to reschedule the campaign brief meeting to 3pm tomorrow (June 29th, 2025). Devon and Leah need to confirm if this new time works for them.
- **Participants**: @sam, @devon, @leah
- **Action Items**: 3 items

**Action Items List:**
1. Confirm availability for 3pm meeting time. (Owner: @devon, Due: 2025-06-28, Priority: high)
2. Confirm availability for 3pm meeting time. (Owner: @leah, Due: 2025-06-28, Priority: high)
3. Update meeting invite if new time is confirmed. (Owner: @sam, Due: 2025-06-28, Priority: medium)

- **Deadline**: 2025-06-28
- **Urgency**: high
- **Channel**: #campaign-briefs
- **Tags**: meeting, reschedule, campaign brief, time change

---

### TC004: Channel/Thread Weighting: Same Structure, Different Tags

**Status**: ❌ FAIL

#### New Message:
- **ID**: 304
- **Channel**: #campaign-briefs
- **User**: Jordan
- **Timestamp**: 2025-06-29T11:00:00
- **Text**: @devon @sam @leah The legal review is progressing well. I've identified a few minor compliance items that need addressing before final approval.

#### Test Results:
- **Expected Action**: UPDATE
- **Actual Action**: UPDATE
- **Similarity Score**: 0.7750
- **Duration**: 8.519s

#### Original Topic (Before Update):
- **Topic ID**: topic_011
- **Title**: EcoBloom Campaign Final Approval & Delivery
- **Summary**: EcoBloom campaign designs finalized and approved for client submission. @Sam prepared the final package, @Leah organized content, and @Jordan completed legal review. The campaign is set for delivery by July 25, 2025, ahead of the July 28 launch.
- **Participants**: @Devon, @Sam, @Leah, @Jordan
- **Action Items**: 5 items
- **Deadline**: 2025-07-28
- **Urgency**: high

#### Updated Topic (After Update):
- **Title**: EcoBloom Campaign Final Approval & Delivery
- **Summary**: EcoBloom campaign designs finalized and approved for client submission. @Sam prepared the final package, @Leah organized content, and @Jordan completed legal review. The campaign is set for delivery by July 25, 2025, ahead of the July 28 launch. @Jordan has identified minor compliance items that need addressing before final legal approval.
- **Participants**: @Devon, @Sam, @Leah, @Jordan
- **Action Items**: 8 items
- **Deadline**: 2025-07-25
- **Urgency**: high

---

### TC005: Create New Finance Invoice Topic

**Status**: ✅ PASS

#### New Message:
- **ID**: 305
- **Channel**: #finance-updates
- **User**: Priya
- **Timestamp**: 2025-06-26T09:45:00
- **Text**: @finance-team The Q3 invoice for Horizon Robotics (PO-HR-7784) is still unpaid. Please confirm that the wire transfer is scheduled before the July 3, 2025 cutoff to avoid penalties.

#### Test Results:
- **Expected Action**: CREATE
- **Actual Action**: CREATE
- **Similarity Score**: 0.0000
- **Duration**: 2.534s

#### New Topic Created:
- **Topic ID**: topic_016
- **Title**: Unpaid Q3 Invoice for Horizon Robotics (PO-HR-7784)
- **Summary**: Priya notified the finance team about the unpaid Q3 invoice for Horizon Robotics (PO-HR-7784). Confirmation is needed that the wire transfer is scheduled before the July 3, 2025 cutoff to avoid penalties.
- **Participants**: Priya, @finance-team
- **Action Items**: 1 items

**Action Items List:**
1. Confirm wire transfer schedule for Horizon Robotics invoice (PO-HR-7784) (Owner: @finance-team, Due: 2025-07-03, Priority: high)

- **Deadline**: 2025-07-03
- **Urgency**: high
- **Channel**: #finance-updates
- **Tags**: invoice, payment, Horizon Robotics, PO-HR-7784, wire transfer, Q3

---

### TC006: Create New HR Hiring Topic

**Status**: ✅ PASS

#### New Message:
- **ID**: 306
- **Channel**: #hr-announcements
- **User**: Maria
- **Timestamp**: 2025-06-30T10:20:00
- **Text**: @team We're opening a new Senior Product Designer position. The role will focus on our mobile app redesign project. Applications close July 15, 2025. Please share with your networks.

#### Test Results:
- **Expected Action**: CREATE
- **Actual Action**: CREATE
- **Similarity Score**: 0.0000
- **Duration**: 2.733s

#### New Topic Created:
- **Topic ID**: topic_016
- **Title**: New Senior Product Designer Position - Mobile App Redesign
- **Summary**: Maria announced a new Senior Product Designer position focused on the mobile app redesign project. Applications are due July 15, 2025. Team members are asked to share the job posting within their networks.
- **Participants**: Maria, @team
- **Action Items**: 1 items

**Action Items List:**
1. Share the Senior Product Designer job posting with personal and professional networks. (Owner: @team, Due: 2025-07-15, Priority: medium)

- **Deadline**: 2025-07-15
- **Urgency**: medium
- **Channel**: #hr-announcements
- **Tags**: hiring, product design, mobile app, redesign, job posting, senior position

---

### TC007: Same Semantic Content, Different Structure (Should Create)

**Status**: ✅ PASS

#### New Message:
- **ID**: 307
- **Channel**: #general
- **User**: Alex
- **Timestamp**: 2025-07-01T14:00:00
- **Text**: @team I heard about the EcoBloom campaign. Can someone share the latest updates? I'm working on a similar project and would love to learn from your approach.

#### Test Results:
- **Expected Action**: CREATE
- **Actual Action**: CREATE
- **Similarity Score**: 0.0000
- **Duration**: 1.873s

#### New Topic Created:
- **Topic ID**: topic_016
- **Title**: EcoBloom Campaign Update Request
- **Summary**: Alex is working on a similar project to the EcoBloom campaign and is requesting updates from the team to learn from their approach.
- **Participants**: @Alex, @team
- **Action Items**: 1 items

**Action Items List:**
1. Share the latest EcoBloom campaign updates with Alex. (Owner: @team, Due: 2025-07-03, Priority: medium)

- **Deadline**: 2025-07-03
- **Urgency**: medium
- **Channel**: #general
- **Tags**: EcoBloom, campaign, updates, project, collaboration, knowledge sharing

---

### TC008: Same Channel/Thread, Different Participants (Should Create)

**Status**: ❌ FAIL

#### New Message:
- **ID**: 308
- **Channel**: #campaign-briefs
- **User**: Priya
- **Timestamp**: 2025-07-02T10:00:00
- **Text**: @finance-team @accounting I need budget approval for the Q4 marketing campaign. The total cost is $50,000 and we need to submit by July 10, 2025.

#### Test Results:
- **Expected Action**: CREATE
- **Actual Action**: CREATE
- **Similarity Score**: 0.7249
- **Duration**: 4.533s

#### New Topic Created:
- **Topic ID**: topic_016
- **Title**: Q4 Marketing Campaign Budget Approval Request
- **Summary**: Priya is requesting budget approval from the finance and accounting teams for the Q4 marketing campaign. The total budget requested is $50,000 and the deadline for submission is July 10, 2025.
- **Participants**: Priya, @finance-team, @accounting
- **Action Items**: 3 items

**Action Items List:**
1. Approve or reject the Q4 marketing campaign budget request. (Owner: @finance-team, Due: 2025-07-10, Priority: high)
2. Approve or reject the Q4 marketing campaign budget request. (Owner: @accounting, Due: 2025-07-10, Priority: high)
3. Submit the Q4 marketing campaign budget request. (Owner: Priya, Due: 2025-07-10, Priority: high)

- **Deadline**: 2025-07-10
- **Urgency**: high
- **Channel**: #campaign-briefs
- **Tags**: budget, marketing, Q4, approval, finance, accounting

---

### TC009: High Semantic Similarity, Different Thread (Should Create)

**Status**: ❌ FAIL

#### New Message:
- **ID**: 309
- **Channel**: #campaign-briefs
- **User**: Devon
- **Timestamp**: 2025-07-03T09:00:00
- **Text**: @sam @leah @jordan We need to discuss the EcoBloom summer campaign timeline. The deadline is July 28, 2025 and we need to coordinate content, design, and legal review.

#### Test Results:
- **Expected Action**: CREATE
- **Actual Action**: UPDATE
- **Similarity Score**: 0.9139
- **Duration**: 6.702s

#### Original Topic (Before Update):
- **Topic ID**: topic_001
- **Title**: EcoBloom Summer Campaign Planning
- **Summary**: The team is planning the EcoBloom summer campaign with a deadline of July 28, 2025. @Leah will provide content by June 30, 2025, @Sam designs by July 10, 2025, and @Jordan legal reviews by July 14, 2025.
- **Participants**: @Devon, @Sam, @Leah, @Jordan
- **Action Items**: 5 items
- **Deadline**: 2025-07-28
- **Urgency**: high

#### Updated Topic (After Update):
- **Title**: EcoBloom Summer Campaign Planning
- **Summary**: The team is planning the EcoBloom summer campaign with a deadline of July 28, 2025. @Leah will provide content by June 30, 2025, @Sam designs by July 10, 2025, and @Jordan legal reviews by July 14, 2025. @Devon initiated a discussion about the EcoBloom summer campaign timeline, emphasizing the need for coordination between content, design, and legal review.
- **Participants**: @Devon, @Sam, @Leah, @Jordan
- **Action Items**: 6 items
- **Deadline**: 2025-07-28
- **Urgency**: high

---

### TC010: Multiple Mentions in Same Thread (Should Update)

**Status**: ✅ PASS

#### New Message:
- **ID**: 310
- **Channel**: #campaign-briefs
- **User**: Devon
- **Timestamp**: 2025-07-04T15:00:00
- **Text**: @sam @leah @jordan @team All team members please note: The EcoBloom campaign kickoff meeting is scheduled for tomorrow at 2pm. Please confirm attendance.

#### Test Results:
- **Expected Action**: UPDATE
- **Actual Action**: UPDATE
- **Similarity Score**: 0.7760
- **Duration**: 6.610s

#### Original Topic (Before Update):
- **Topic ID**: topic_001
- **Title**: EcoBloom Summer Campaign Planning
- **Summary**: The team is planning the EcoBloom summer campaign with a deadline of July 28, 2025. @Leah will provide content by June 30, 2025, @Sam designs by July 10, 2025, and @Jordan legal reviews by July 14, 2025.
- **Participants**: @Devon, @Sam, @Leah, @Jordan
- **Action Items**: 5 items
- **Deadline**: 2025-07-28
- **Urgency**: high

#### Updated Topic (After Update):
- **Title**: EcoBloom Summer Campaign Planning
- **Summary**: The team is planning the EcoBloom summer campaign with a deadline of July 28, 2025. @Leah will provide content by June 30, 2025, @Sam designs by July 10, 2025, and @Jordan legal reviews by July 14, 2025. The EcoBloom campaign kickoff meeting is scheduled for July 5, 2025 at 2pm.
- **Participants**: @Devon, @Sam, @Leah, @Jordan
- **Action Items**: 9 items
- **Deadline**: 2025-07-28
- **Urgency**: high

---

## Summary Statistics

- **Total Tests**: 10
- **Passed**: 6
- **Failed**: 4
- **Pass Rate**: 60.0%
