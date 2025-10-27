# 🎯 **How to Get Model Score: 91.1**

## 📊 **The IMPROVED_MODEL_SCORE Formula**

The Model Score of **91.1** is calculated using a **weighted formula** that combines four key performance metrics:

```
IMPROVED_MODEL_SCORE = (Cluster_Count_Score × 0.25) + 
                       (Coverage_Score × 0.25) + 
                       (Precision_Score × 0.25) + 
                       (Deviation_Score × 0.25)
```

---

## 🔍 **Step-by-Step Calculation for Model Score 91.1**

### **Example: gemini-1.5-flash Model**

Let's trace through the exact calculation using real data from your CSV:

#### **1. Cluster Count Score (25% weight)**
```
Cluster_Count_Score = min(expected_clusters, total_clusters) ÷ max(expected_clusters, total_clusters) × 100

Expected clusters: 6 (eco_bloom_campaign, fitfusion_rebrand, technova_launch, greenscape_report, q3_content_calendar, urbanedge_strategy)
Total clusters generated: 7

Cluster_Count_Score = min(6, 7) ÷ max(6, 7) × 100
Cluster_Count_Score = 6 ÷ 7 × 100
Cluster_Count_Score = 0.857 × 100
Cluster_Count_Score = 85.7
```

#### **2. Coverage Score (25% weight)**
```
Coverage_Score = (found_benchmark_clusters ∩ expected_cluster_ids) ÷ len(expected_cluster_ids) × 100

Found benchmark clusters: {eco_bloom_campaign, fitfusion_rebrand, technova_launch, greenscape_report, q3_content_calendar, urbanedge_strategy}
Expected cluster IDs: {eco_bloom_campaign, fitfusion_rebrand, technova_launch, greenscape_report, q3_content_calendar, urbanedge_strategy}

Coverage_Score = 6 ÷ 6 × 100
Coverage_Score = 1.0 × 100
Coverage_Score = 100.0
```

#### **3. Precision Score (25% weight)**
```
Precision_Score = Average of all PRECISION_PERCENT values across clusters

From CSV data:
- Row 2: 100.0%
- Row 3: 78.33%
- Row 4: 100.0%
- Row 5: 100.0%
- Row 6: 100.0%
- Row 7: 100.0%
- Row 8: 100.0%

Precision_Score = (100.0 + 78.33 + 100.0 + 100.0 + 100.0 + 100.0 + 100.0) ÷ 7
Precision_Score = 678.33 ÷ 7
Precision_Score = 96.9
```

#### **4. Deviation Score (25% weight)**
```
Deviation_Score = max(0, 100 - |average_deviation|)

Average deviation = Average of absolute MESSAGE_COUNT_DEVIATION_PERCENT values

From CSV data:
- Row 2: 0.0%
- Row 3: 27.66%
- Row 4: 0.0%
- Row 5: -43.94%
- Row 6: 0.0%
- Row 7: 0.0%
- Row 8: -56.06%

Average deviation = (0.0 + 27.66 + 0.0 + 43.94 + 0.0 + 0.0 + 56.06) ÷ 7
Average deviation = 127.66 ÷ 7
Average deviation = 18.24

Deviation_Score = max(0, 100 - 18.24)
Deviation_Score = max(0, 81.76)
Deviation_Score = 81.76
```

#### **5. Final IMPROVED_MODEL_SCORE Calculation**
```
IMPROVED_MODEL_SCORE = (85.7 × 0.25) + (100.0 × 0.25) + (96.9 × 0.25) + (81.76 × 0.25)
IMPROVED_MODEL_SCORE = 21.43 + 25.0 + 24.23 + 20.44
IMPROVED_MODEL_SCORE = 91.1
```

---

## 📋 **Real Data Verification**

### **From Your CSV (gemini-1.5-flash rows):**

| Row | Cluster | Precision | Recall | Coverage | Deviation | Matched | Missing | Extra |
|-----|---------|-----------|--------|----------|-----------|---------|---------|-------|
| 2 | EcoBloom Campaign | 100.0% | 100.0% | 100.0% | 0.0% | 44 | 0 | 0 |
| 3 | FitFusion Rebranding | 78.33% | 100.0% | 100.0% | 27.66% | 47 | 0 | 13 |
| 4 | TechNova Product Launch | 100.0% | 100.0% | 100.0% | 0.0% | 68 | 0 | 0 |
| 5 | GreenScape Report | 100.0% | 56.06% | 56.06% | -43.94% | 37 | 29 | 0 |
| 6 | UrbanEdge Strategy | 100.0% | 100.0% | 100.0% | 0.0% | 37 | 0 | 0 |
| 7 | Q3 Content Calendar | 100.0% | 100.0% | 100.0% | 0.0% | 25 | 0 | 0 |
| 8 | GreenScape Extension | 100.0% | 43.94% | 43.94% | -56.06% | 29 | 37 | 0 |

---

## 🎯 **Why Model Score is 91.1 (Not 100)**

The model doesn't achieve a perfect score because:

1. **Cluster Count Penalty**: Generated 7 clusters instead of expected 6 (-14.3 points)
2. **Precision Impact**: One cluster (FitFusion) had 78.33% precision instead of 100% (-3.1 points)
3. **Deviation Penalty**: Some clusters had significant message count deviations (-18.24 points)

**Perfect Score Breakdown:**
- Cluster Count: 100.0 (perfect cluster count)
- Coverage: 100.0 (found all expected clusters)
- Precision: 100.0 (all clusters at 100% precision)
- Deviation: 100.0 (no message count deviations)

**Actual Score Breakdown:**
- Cluster Count: 85.7 (7 clusters vs 6 expected)
- Coverage: 100.0 (found all 6 expected clusters)
- Precision: 96.9 (average across all clusters)
- Deviation: 81.76 (some clusters had message count deviations)

---

## ✅ **Formula Verification**

You can verify this calculation in Excel using these formulas:

```
A1: Cluster Count Score = MIN(6,7)/MAX(6,7)*100 = 85.7
A2: Coverage Score = 6/6*100 = 100.0
A3: Precision Score = AVERAGE(100,78.33,100,100,100,100,100) = 96.9
A4: Deviation Score = MAX(0,100-18.24) = 81.76
A5: Final Score = A1*0.25+A2*0.25+A3*0.25+A4*0.25 = 91.1
```

**The Model Score of 91.1 is the weighted average of these four performance metrics, giving equal importance to cluster count accuracy, coverage completeness, precision quality, and message count consistency.**
