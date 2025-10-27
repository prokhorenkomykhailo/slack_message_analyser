# 🚀 Quick Formula Reference Card

## **For Client Verification - All Formulas at a Glance**

---

## 📊 **Core Metrics (Row-Level)**

| Metric | Formula | Example |
|--------|---------|---------|
| **Deviation %** | `((LLM_COUNT - BENCHMARK_COUNT) / BENCHMARK_COUNT) × 100` | `((60-47)/47) × 100 = 27.66%` |
| **Coverage %** | `(MATCHED_MESSAGES / BENCHMARK_COUNT) × 100` | `(47/47) × 100 = 100%` |
| **Precision %** | `(MATCHED_MESSAGES / LLM_COUNT) × 100` | `(47/60) × 100 = 78.33%` |
| **Recall %** | `(MATCHED_MESSAGES / BENCHMARK_COUNT) × 100` | `(47/47) × 100 = 100%` |

---

## 🏆 **Scoring Formulas**

### **Final Score (Simple)**
```
FINAL_SCORE = (Deviation_Score × 0.4) + (Coverage × 0.3) + (Precision × 0.3)

Where: Deviation_Score = max(0, 100 - |Deviation%|)
```

### **Improved Model Score (Advanced)**
```
IMPROVED_SCORE = (Cluster_Count × 0.25) + (Coverage × 0.25) + (Precision × 0.25) + (Deviation × 0.25)

Where:
- Cluster_Count = min(expected, actual) / max(expected, actual) × 100
- Coverage = found_clusters / expected_clusters × 100  
- Precision = average(precision across all clusters)
- Deviation = max(0, 100 - average(|deviation%|))
```

---

## 💰 **Cost Formula**
```
COST = (INPUT_TOKENS/1000 × INPUT_RATE) + (OUTPUT_TOKENS/1000 × OUTPUT_RATE)
```

---

## 🔍 **How to Verify**

1. **Pick any row** from `llm_analysis_with_improved_scores.csv`
2. **Use formulas above** to recalculate each metric
3. **Compare results** with CSV values
4. **Run verification script**: `python verify_calculations.py`

---

## 📁 **Source Files**
- **Results**: `llm_analysis_with_improved_scores.csv`
- **Benchmark**: `phases/phase3_clusters.json`
- **Scripts**: `improved_model_scoring.py`, `add_final_score.py`
- **Full Docs**: `FORMULA_DOCUMENTATION.md`

**✅ Every calculation is transparent and verifiable!**

