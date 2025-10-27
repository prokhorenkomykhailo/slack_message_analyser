# Phase 4: Embedding-Based Similarity Analysis

## 🧠 How Similarity is Calculated

### 1. **Embedding Generation**
```python
# For each cluster, we:
1. Get all message texts from the cluster
2. Generate embeddings for each message using sentence-transformers
3. Average all message embeddings to get a cluster embedding
4. This gives us a 384-dimensional vector representing the cluster's semantic content
```

### 2. **Multiple Similarity Metrics**

#### **A. Cosine Similarity (Primary)**
```python
cosine_similarity = cosine_similarity([embedding1], [embedding2])[0][0]
# Range: 0.0 to 1.0
# Higher = more similar
# Measures the angle between embedding vectors
```

#### **B. Jaccard Similarity (Fallback)**
```python
# Word overlap similarity
words1 = set(text1.split())
words2 = set(text2.split())
jaccard = len(words1 ∩ words2) / len(words1 ∪ words2)
# Range: 0.0 to 1.0
# Measures word overlap between clusters
```

#### **C. Euclidean Distance**
```python
euclidean_distance = np.linalg.norm(embedding1 - embedding2)
# Range: 0.0 to ∞
# Lower = more similar
# Measures straight-line distance between embeddings
```

#### **D. Combined Score**
```python
combined_score = cosine_similarity * 0.7 + jaccard_similarity * 0.3
# Weighted combination of semantic and lexical similarity
```

### 3. **Merge Decision Logic**

```python
# Clusters are merged if:
1. combined_score > 0.3  # 30% similarity threshold
2. They are among the top 2 most similar pairs
3. Neither cluster has already been merged
```

### 4. **Split Decision Logic**

```python
# Clusters are split if:
1. Size > 20 messages OR
2. Coherence score < 0.3
```

### 5. **Coherence Analysis**

```python
# For each cluster, we calculate:
1. Pairwise similarity between all messages
2. Average pairwise similarity
3. Variance in similarities
4. Coherence score = avg_similarity * (1 - variance)
```

## 📊 Example Calculation

### **Cluster A**: "EcoBloom Campaign Planning"
- Messages: ["Let's plan the summer campaign", "We need budget for EcoBloom", "Campaign timeline discussion"]
- Embedding: [0.1, 0.3, 0.2, ..., 0.4] (384 dimensions)

### **Cluster B**: "EcoBloom Budget Planning"  
- Messages: ["Budget allocation for EcoBloom", "Summer campaign costs", "Financial planning"]
- Embedding: [0.2, 0.4, 0.1, ..., 0.3] (384 dimensions)

### **Similarity Calculation**:
```python
cosine_similarity = 0.85  # High semantic similarity
jaccard_similarity = 0.60  # Good word overlap
combined_score = 0.85 * 0.7 + 0.60 * 0.3 = 0.775

# Since 0.775 > 0.3 threshold → MERGE!
```

## 🎯 Why This Approach is Better

### **1. Semantic Understanding**
- **Old**: Only word overlap ("campaign" = "campaign")
- **New**: Semantic similarity ("planning" ≈ "budget" ≈ "financial")

### **2. Context Awareness**
- **Old**: "EcoBloom" and "FitFusion" = 0% similarity
- **New**: Both are "marketing campaigns" = high similarity

### **3. Robust to Variations**
- **Old**: "campaign" ≠ "Campaign" ≠ "CAMPAIGN"
- **New**: All variations have similar embeddings

### **4. Multi-Metric Validation**
- **Cosine**: Semantic similarity
- **Jaccard**: Lexical similarity  
- **Combined**: Best of both worlds

## 🔧 Technical Details

### **Embedding Model**: `all-MiniLM-L6-v2`
- **Size**: 384 dimensions
- **Training**: 1.1B sentence pairs
- **Performance**: Fast and accurate
- **Use Case**: Perfect for semantic similarity

### **Thresholds**:
- **Merge threshold**: 0.3 (30% similarity)
- **Split threshold**: 20 messages OR coherence < 0.3
- **Top pairs**: 2 most similar clusters

### **Fallback Strategy**:
- If embedding model fails → word-based similarity
- If API calls fail → simulation with real metrics
- Always preserves all original messages

## 📈 Output Example

```json
{
  "merge_operations": [
    {
      "operation": "merge",
      "clusters": ["cluster_002", "cluster_003"],
      "reason": "High embedding similarity (Cosine: 0.847, Jaccard: 0.623, Combined: 0.765) between FitFusion Rebranding and TechNova Product Launch - should be combined",
      "similarity_score": 0.765,
      "cosine_similarity": 0.847,
      "jaccard_similarity": 0.623,
      "euclidean_distance": 0.234
    }
  ],
  "split_operations": [
    {
      "operation": "split",
      "cluster": "cluster_001",
      "reason": "Large cluster (44 messages) with low coherence (0.234) - should be split by content",
      "cluster_size": 44,
      "coherence_score": 0.234
    }
  ]
}
```

## 🚀 Usage

```bash
# Install dependencies
pip install -r requirements_embeddings.txt

# Run embedding-based analysis
python phase4_merge_split_ai_embeddings.py
```

This approach provides **much more accurate** similarity analysis by understanding the **semantic meaning** of messages, not just word overlap! 🎯
