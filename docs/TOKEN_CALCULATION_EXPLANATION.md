# Token Calculation Explanation for Client

## Overview

This document explains exactly how we calculate and track token usage for all AI models in our evaluation system.

---

## 🎯 Key Principle: We Don't Calculate Tokens - The AI APIs Do

**Important**: We **DO NOT manually calculate tokens**. Instead, we:

1. ✅ **Use the official token counts returned by each AI provider's API**
2. ✅ **Extract these values directly from the API response**
3. ✅ **Store them in our output files for analysis**

This ensures 100% accuracy because each provider (OpenAI, Google, Anthropic, etc.) uses their own official tokenization method.

---

## 📊 How Token Tracking Works

### Step 1: API Call
When we send a prompt to an AI model, we make an API call that includes:
- **Input**: The prompt (instructions + data from `Synthetic_Slack_Messages.csv`)
- **Output**: The model's response (clustered topics in JSON format)

### Step 2: API Response
The AI provider's API returns a response that includes:
```json
{
  "response": "... the actual text response ...",
  "usage": {
    "prompt_tokens": 25234,      // Input tokens (what we sent)
    "completion_tokens": 1456,   // Output tokens (what model generated)
    "total_tokens": 26690        // Total = prompt + completion
  },
  "cost": 0.00123                // Calculated cost
}
```

### Step 3: We Extract and Store
We extract these **official** token counts from the API response and store them in our output files:
- `output/phase3_topic_clustering/google_gemini-2.0-flash-001.json` (Step 1)
- `output/phase4_balanced_refinement/google_gemini-2.0-flash-001_balanced.json` (Step 2)

---

## 🔍 Token Breakdown by Phase

### **Step 1 (Phase 3): Initial Topic Clustering**

**Input Data**: `data/Synthetic_Slack_Messages.csv`
- 300 Slack messages
- Each message contains: channel, user, timestamp, text, thread_id

**Prompt Structure**:
```
[Instructions: ~800 tokens]
- Task description
- Clustering criteria
- Output format requirements

[Data: ~24,000 tokens]
- All 300 formatted messages
- Format: "ID: X | Channel: Y | User: Z | Thread: W | Text: ..."

Total Input: ~25,000 tokens
```

**Output**: Model's clustering response (~1,500 tokens)

**Example for Gemini 2.0 Flash**:
- `prompt_tokens`: 25,234 (from API)
- `completion_tokens`: 1,456 (from API)
- `total_tokens`: 26,690 (from API)

---

### **Step 2 (Phase 4): Balanced Refinement**

**Input Data**: Results from Step 1 + refinement instructions
- Previous clusters from Step 1
- Instructions to merge/split clusters
- Benchmark comparison data

**Prompt Structure**:
```
[Instructions: ~1,200 tokens]
- Refinement task description
- Merge/split criteria
- Quality improvement guidelines

[Data: ~3,000 tokens]
- Step 1 clusters
- Message references
- Benchmark information

Total Input: ~4,200 tokens
```

**Output**: Refined clustering response (~200 tokens)

**Example for Gemini 2.0 Flash**:
- `prompt_tokens`: 4,245 (from API)
- `completion_tokens`: 183 (from API)
- `total_tokens`: 4,428 (from API)

---

## 💰 Cost Calculation

Once we have the **official token counts from the API**, we calculate costs using the provider's pricing:

### Formula:
```
INPUT_COST = prompt_tokens × COST_PER_INPUT_TOKEN
OUTPUT_COST = completion_tokens × COST_PER_OUTPUT_TOKEN
TOTAL_COST = INPUT_COST + OUTPUT_COST
```

### Example: Gemini 2.0 Flash (Combined Step 1 + Step 2)

**Pricing** (from Google's official pricing):
- Input: $0.0001 per 1,000 tokens
- Output: $0.0004 per 1,000 tokens

**Step 1 Tokens**:
- Input: 25,234 tokens
- Output: 1,456 tokens

**Step 2 Tokens**:
- Input: 4,245 tokens
- Output: 183 tokens

**Combined Calculation**:
```
Total Input Tokens:  25,234 + 4,245 = 29,479 tokens
Total Output Tokens: 1,456 + 183 = 1,639 tokens
Total Tokens:        29,479 + 1,639 = 31,118 tokens

Input Cost:  29,479 × $0.0001 = $0.0029479
Output Cost: 1,639 × $0.0004 = $0.0006556
Total Cost:  $0.0036035
```

---

## 🔬 Verification Process

To verify our token tracking is accurate, we created `calculate_step1_input_tokens.py`:

### What This Script Does:

1. **Reconstructs the exact prompt** we sent to the API
   - Loads the same CSV file (`Synthetic_Slack_Messages.csv`)
   - Uses the same prompt template
   - Formats messages identically

2. **Estimates tokens** using common methods:
   - Simple: 4 characters per token
   - Accurate: Weighted calculation (words + characters)

3. **Compares with actual API values**:
   - Reads the official `prompt_tokens` from the API response
   - Shows the difference between estimate and actual
   - Confirms our stored values match the API

### Purpose:
This script is for **verification only** - it helps us understand and validate that the token counts we store from the API are reasonable and accurate.

---

## 📁 Where Token Data is Stored

### Step 1 Output Files:
```
output/phase3_topic_clustering/
├── google_gemini-2.0-flash-001.json
├── openai_gpt-4o.json
├── anthropic_claude-3-5-sonnet-20241022.json
└── ... (all other models)
```

Each file contains:
```json
{
  "usage": {
    "prompt_tokens": 25234,
    "completion_tokens": 1456,
    "total_tokens": 26690
  },
  "cost": 0.00123
}
```

### Step 2 Output Files:
```
output/phase4_balanced_refinement/
├── google_gemini-2.0-flash-001_balanced.json
├── openai_gpt-4o_balanced.json
├── anthropic_claude-3-5-sonnet-20241022_balanced.json
└── ... (all other models)
```

Same structure with Step 2 token counts.

---

## 🎓 Why Different Models Have Different Token Counts

Even when given the **exact same input**, different models report different token counts because:

1. **Different Tokenization Methods**:
   - OpenAI uses tiktoken (BPE-based)
   - Google Gemini uses SentencePiece
   - Anthropic Claude uses custom tokenizer
   - Each breaks text into tokens differently

2. **Example**: The word "tokenization"
   - OpenAI GPT-4: `["token", "ization"]` = 2 tokens
   - Google Gemini: `["token", "iz", "ation"]` = 3 tokens
   - Anthropic Claude: `["tokenization"]` = 1 token

3. **This is normal and expected** - each provider's official count is correct for their system.

---

## ✅ Summary for Client

**Q: How do you calculate tokens?**

**A**: We don't calculate them manually. We use the **official token counts returned by each AI provider's API**. This ensures:

- ✅ **100% Accuracy**: Provider's own tokenization method
- ✅ **Consistency**: Same method used for billing
- ✅ **Transparency**: All values stored in output JSON files
- ✅ **Verifiability**: Can be cross-checked with provider's billing

**Q: Why do different models have different token counts?**

**A**: Each AI provider uses their own tokenization method. This is normal and expected. We report each provider's official count.

**Q: How do you verify the token counts are correct?**

**A**: 
1. We store the exact API response with token counts
2. We created verification scripts to reconstruct prompts and estimate tokens
3. We compare estimates with actual API values to ensure reasonableness
4. All token data can be cross-referenced with provider billing

---

## 📞 Technical Contact

For questions about token calculation methodology:
- Review the source code: `phases/phase3_topic_clustering.py`
- Review verification script: `calculate_step1_input_tokens.py`
- Check output files: `output/phase3_topic_clustering/*.json`
- Review analysis files: `output/step2_client_analysis/*.xlsx`

---

**Last Updated**: October 9, 2025

