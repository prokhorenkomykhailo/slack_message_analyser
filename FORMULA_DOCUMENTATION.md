# 📊 Formula Documentation - Complete Client Verification Guide

## 🎯 Overview
This document contains ALL formulas used in the Phase 3 clustering analysis. Every calculation is transparent and verifiable.

---

## 📈 **Core Metrics Formulas**

### **1. Message Count Deviation Percentage**
```
MESSAGE_COUNT_DEVIATION_PERCENT = ((LLM_MESSAGE_COUNT - BENCHMARK_MESSAGE_COUNT) / BENCHMARK_MESSAGE_COUNT) × 100

Example:
- Benchmark: 44 messages
- LLM: 44 messages  
- Deviation: ((44 - 44) / 44) × 100 = 0.0%
```

### **2. Coverage Percentage**
```
COVERAGE_PERCENTAGE = (MATCHED_MESSAGES / BENCHMARK_MESSAGE_COUNT) × 100

Example:
- Benchmark: 44 messages
- Matched: 44 messages
- Coverage: (44 / 44) × 100 = 100.0%
```

### **3. Precision Percentage**
```
PRECISION_PERCENT = (MATCHED_MESSAGES / LLM_MESSAGE_COUNT) × 100

Example:
- LLM: 60 messages
- Matched: 47 messages
- Precision: (47 / 60) × 100 = 78.33%
```

### **4. Recall Percentage**
```
RECALL_PERCENT = (MATCHED_MESSAGES / BENCHMARK_MESSAGE_COUNT) × 100

Example:
- Benchmark: 44 messages
- Matched: 44 messages
- Recall: (44 / 44) × 100 = 100.0%
```

---

## 🏆 **Scoring Formulas**

### **5. Final Score (Simple Version)**
```
FINAL_SCORE = (Deviation_Score × 0.4) + (Coverage_Score × 0.3) + (Precision_Score × 0.3)

Where:
- Deviation_Score = max(0, 100 - |MESSAGE_COUNT_DEVIATION_PERCENT|)
- Coverage_Score = COVERAGE_PERCENTAGE
- Precision_Score = PRECISION_PERCENT

Example:
- Deviation: 0% → Deviation_Score = 100
- Coverage: 100% → Coverage_Score = 100  
- Precision: 78.33% → Precision_Score = 78.33
- Final Score = (100 × 0.4) + (100 × 0.3) + (78.33 × 0.3) = 93.5
```

### **6. Improved Model Score (Advanced Version)**
```
IMPROVED_MODEL_SCORE = (Cluster_Count_Score × 0.25) + (Coverage_Score × 0.25) + (Precision_Score × 0.25) + (Deviation_Score × 0.25)

Where:
- Cluster_Count_Score = min(expected_clusters, total_clusters) / max(expected_clusters, total_clusters) × 100
- Coverage_Score = (found_benchmark_clusters ∩ expected_clusters) / expected_clusters × 100
- Precision_Score = average(PRECISION_PERCENT across all clusters)
- Deviation_Score = max(0, 100 - average(|MESSAGE_COUNT_DEVIATION_PERCENT|))

Example:
- Expected clusters: 6, Generated: 7 → Cluster_Count_Score = 6/7 × 100 = 85.7
- Found 6/6 expected → Coverage_Score = 100
- Average precision: 91.1 → Precision_Score = 91.1
- Average deviation: 8.9% → Deviation_Score = 100 - 8.9 = 91.1
- Improved Score = (85.7 × 0.25) + (100 × 0.25) + (91.1 × 0.25) + (91.1 × 0.25) = 91.1
```

---

## 💰 **Cost Calculation Formulas**

### **7. Token Cost**
```
TOKEN_COST = (INPUT_TOKENS / 1000) × COST_PER_INPUT_TOKEN + (OUTPUT_TOKENS / 1000) × COST_PER_OUTPUT_TOKEN

Example:
- Input tokens: 25,474
- Output tokens: 2,080
- Input cost: $0.000075 per 1K tokens
- Output cost: $0.0003 per 1K tokens
- Cost = (25,474/1000 × 0.000075) + (2,080/1000 × 0.0003) = $2.53455
```

---

## 🔍 **Cluster Matching Formulas**

### **8. Message Overlap Calculation**
```
overlap = benchmark_message_ids ∩ llm_message_ids
overlap_percentage = (len(overlap) / len(benchmark_message_ids)) × 100

Example:
- Benchmark messages: {1,2,3,4,5}
- LLM messages: {1,2,3,6,7}
- Overlap: {1,2,3}
- Overlap percentage: (3/5) × 100 = 60%
```

### **9. Missing Messages**
```
missing_messages = benchmark_message_ids - llm_message_ids

Example:
- Benchmark: {1,2,3,4,5}
- LLM: {1,2,3,6,7}
- Missing: {4,5}
```

### **10. Extra Messages**
```
extra_messages = llm_message_ids - benchmark_message_ids

Example:
- Benchmark: {1,2,3,4,5}
- LLM: {1,2,3,6,7}
- Extra: {6,7}
```

---

## 📊 **Validation Examples**

### **Example 1: Perfect Match**
```
Benchmark: EcoBloom Campaign (44 messages)
LLM: EcoBloom Summer Campaign: Planning & Revisions (44 messages)
Matched: 44 messages

Calculations:
- Deviation: ((44-44)/44) × 100 = 0.0%
- Coverage: (44/44) × 100 = 100.0%
- Precision: (44/44) × 100 = 100.0%
- Final Score: (100×0.4) + (100×0.3) + (100×0.3) = 100.0
```

### **Example 2: Partial Match**
```
Benchmark: FitFusion Rebranding (47 messages)
LLM: FitFusion Rebranding Project: Design, Messaging & Legal (60 messages)
Matched: 47 messages

Calculations:
- Deviation: ((60-47)/47) × 100 = 27.66%
- Coverage: (47/47) × 100 = 100.0%
- Precision: (47/60) × 100 = 78.33%
- Final Score: (72.34×0.4) + (100×0.3) + (78.33×0.3) = 82.0
```

---

## 🛠️ **How to Verify**

### **Step 1: Check Individual Calculations**
1. Open `llm_analysis_with_improved_scores.csv`
2. Pick any row
3. Use the formulas above to recalculate each metric
4. Compare with the values in the CSV

### **Step 2: Verify Score Calculations**
1. Take the component metrics (deviation, coverage, precision)
2. Apply the scoring formulas
3. Check if the final scores match

### **Step 3: Cross-Reference with Source Data**
1. Check `phases/phase3_clusters.json` for benchmark data
2. Check `output/phase3_topic_clustering/` for model results
3. Verify message ID matching manually

---

## 📁 **Source Files for Verification**

- **Benchmark Data**: `phases/phase3_clusters.json`
- **Model Results**: `output/phase3_topic_clustering/*.json`
- **Analysis Scripts**: 
  - `improved_model_scoring.py` (lines 34-104)
  - `add_final_score.py` (lines 22-47)
  - `enhanced_clustering_analysis.py` (lines 55-89)
- **Final Results**: `llm_analysis_with_improved_scores.csv`

---

## ✅ **Client Verification Checklist**

- [ ] All formulas are documented above
- [ ] Example calculations provided for each metric
- [ ] Source code files are accessible
- [ ] Raw data files are available
- [ ] Step-by-step verification process provided
- [ ] No hidden calculations or black boxes

**Every single number in the analysis can be traced back to these formulas and verified independently.**

