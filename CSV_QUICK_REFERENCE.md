# 📊 CSV Quick Reference - `llm_analysis_with_improved_scores.csv`

## 🎯 **What This File Contains**

**Complete analysis of all AI models on Phase 3 topic clustering**  
**Every row = One model's performance on one cluster**  
**Every column = A specific metric or piece of data**

---

## 📋 **Column Quick Reference**

### **🔍 Model Info**
- `MODEL` → Which AI model (e.g., "gemini-1.5-flash")
- `SUCCESS` → Did it work? (True/False)
- `CLUSTER_ID` → Which cluster (1, 2, 3, etc.)

### **📊 Results**
- `CLUSTER_TITLE` → What the model called this cluster
- `CLUSTER_MESSAGES` → How many messages in this cluster
- `MESSAGE_IDS` → Specific message IDs (comma-separated)

### **⚡ Performance**
- `DURATION_SECONDS` → How long it took
- `TOKEN_COST` → Cost in dollars
- `INPUT_TOKENS` / `OUTPUT_TOKENS` → Token usage

### **🎯 Benchmark Comparison**
- `BENCHMARK_TITLE` → What this cluster should be called
- `BENCHMARK_MESSAGE_COUNT` → How many messages should be here
- `LLM_MESSAGE_COUNT` → How many messages the model put here

### **✅ Matching Analysis**
- `MATCHED_MESSAGES` → Messages correctly placed
- `MISSING_MESSAGES` → Messages that should be here but aren't
- `EXTRA_MESSAGES` → Messages that shouldn't be here but are

### **📈 Quality Metrics**
- `MESSAGE_COUNT_DEVIATION_PERCENT` → How far off the count is (0% = perfect)
- `COVERAGE_PERCENTAGE` → What % of expected messages were found (100% = perfect)
- `PRECISION_PERCENT` → What % of model's messages were correct (100% = perfect)
- `RECALL_PERCENT` → What % of expected messages were found (100% = perfect)

### **🏆 Final Score**
- `IMPROVED_MODEL_SCORE` → Overall performance (0-100, higher = better)

---

## 🔍 **How to Read a Row**

**Example:**
```
MODEL: gemini-1.5-flash
CLUSTER_TITLE: "EcoBloom Summer Campaign: Planning & Revisions"
BENCHMARK_TITLE: "EcoBloom Campaign"
BENCHMARK_MESSAGE_COUNT: 44
LLM_MESSAGE_COUNT: 44
MATCHED_MESSAGES: 44
MESSAGE_COUNT_DEVIATION_PERCENT: 0.0%
COVERAGE_PERCENTAGE: 100.0%
PRECISION_PERCENT: 100.0%
IMPROVED_MODEL_SCORE: 91.1
```

**Translation:**
- Model correctly identified 44 messages for EcoBloom Campaign
- Perfect match (0% deviation, 100% coverage, 100% precision)
- Overall model score: 91.1 (excellent)

---

## 📊 **How to Compare Models**

1. **Sort by `IMPROVED_MODEL_SCORE`** (highest first)
2. **Look at average scores** across all clusters for each model
3. **Check `TOKEN_COST`** for cost efficiency
4. **Consider `DURATION_SECONDS`** for speed

---

## ✅ **How to Verify**

1. **Pick any row**
2. **Use raw data** (BENCHMARK_MESSAGE_COUNT, LLM_MESSAGE_COUNT, MATCHED_MESSAGES)
3. **Apply formulas** from `FORMULA_DOCUMENTATION.md`
4. **Compare with calculated columns**
5. **Run verification**: `python verify_calculations.py`

---

## 🎯 **Quick Formulas**

- **Deviation**: `((LLM_COUNT - BENCHMARK_COUNT) / BENCHMARK_COUNT) × 100`
- **Coverage**: `(MATCHED_MESSAGES / BENCHMARK_COUNT) × 100`
- **Precision**: `(MATCHED_MESSAGES / LLM_COUNT) × 100`

**Every calculation is transparent and verifiable!**
