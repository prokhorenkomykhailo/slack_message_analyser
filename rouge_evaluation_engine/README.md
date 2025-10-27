# ROUGE-based Clustering Evaluation Engine

## 🎯 Overview

This engine provides **ROUGE-based evaluation** for topic clustering results, addressing the client's feedback that traditional F1 scores aren't sufficient for unlimited topics and summaries.

## 🔍 Why ROUGE Instead of Simple F1?

**Traditional F1 Score Issues:**
- Only measures message-level overlap
- Doesn't penalize cluster count mismatches (6 vs 7 vs 15 clusters)
- Ignores text quality and semantic similarity
- Can give high scores even with poor clustering structure

**ROUGE Advantages:**
- **Text Quality**: Measures how well cluster titles/descriptions match
- **Semantic Similarity**: Considers actual text content, not just message IDs
- **Unlimited Topics**: Works regardless of how many clusters are created
- **Summary Quality**: Evaluates the descriptive quality of each cluster

## 📊 Evaluation Metrics

### 1. **ROUGE Text Similarity**
- **ROUGE-1**: Unigram overlap between cluster descriptions
- **ROUGE-2**: Bigram overlap between cluster descriptions  
- **ROUGE-L**: Longest common subsequence (captures structure)

### 2. **Clustering Structure Metrics**
- **Adjusted Rand Index (ARI)**: Pairwise clustering similarity
- **V-Measure**: Harmonic mean of homogeneity and completeness
- **Homogeneity**: How well each cluster contains only one class
- **Completeness**: How well all members of a class are assigned to the same cluster

### 3. **Message-Level Metrics**
- **Precision**: How many predicted messages are correctly grouped
- **Recall**: How many reference messages are found
- **F1**: Harmonic mean of precision and recall

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd phase_evaluation_engine/rouge_evaluation_engine
pip install -r requirements.txt
```

### 2. Test with Single Model
```bash
python test_rouge_evaluation.py
```

### 3. Run Full Evaluation
```bash
python run_rouge_evaluation.py
```

## 📁 File Structure

```
rouge_evaluation_engine/
├── requirements.txt              # Python dependencies
├── rouge_clustering_evaluator.py # Main evaluation engine
├── run_rouge_evaluation.py      # Run evaluation on all models
├── test_rouge_evaluation.py     # Test with single model
└── README.md                    # This file
```

## 🔧 How It Works

### 1. **Text Extraction**
- Extracts meaningful text from each cluster (title, summary, description)
- Combines multiple text fields for comprehensive comparison

### 2. **ROUGE Calculation**
- Compares each predicted cluster with all reference clusters
- Finds best text similarity match using ROUGE-L F1
- Calculates aggregate ROUGE scores across all clusters

### 3. **Clustering Analysis**
- Maps predicted clusters to reference clusters
- Calculates structural similarity metrics (ARI, V-Measure)
- Analyzes message-level overlap and precision/recall

### 4. **Comprehensive Reporting**
- Individual model results (JSON + CSV)
- Cross-model comparison
- Metric correlations and rankings

## 📊 Example Output

```
📊 EVALUATION SUMMARY FOR google_gemini-1.5-flash
============================================================
🏗️  CLUSTERING STRUCTURE:
   Reference clusters: 6
   Predicted clusters: 7
   Adjusted Rand Index: 0.8234
   V-Measure: 0.7845
   Homogeneity: 0.8123
   Completeness: 0.7589

📨 MESSAGE-LEVEL METRICS:
   Overall Precision: 0.9532
   Overall Recall: 0.9233
   Overall F1: 0.9381
   Message Overlap: 265/278

🔍 ROUGE TEXT SIMILARITY:
   ROUGE-1 F1: 0.7845
   ROUGE-2 F1: 0.7234
   ROUGE-L F1: 0.8123
```

## 🎯 Key Benefits

1. **Better Evaluation**: ROUGE captures text quality, not just message overlap
2. **Structure-Aware**: ARI and V-Measure penalize cluster count mismatches
3. **Unlimited Topics**: Works regardless of how many clusters are created
4. **Comprehensive**: Combines text similarity, clustering structure, and message accuracy
5. **Client-Ready**: Addresses the specific concerns about unlimited topics and summaries

## 🔍 Understanding the Results

### **High ROUGE-L F1 + Low ARI**
- Good text descriptions but poor clustering structure
- Model creates good summaries but wrong number of clusters

### **High ARI + Low ROUGE-L F1**
- Good clustering structure but poor text descriptions
- Model groups messages correctly but writes poor titles/summaries

### **High Both**
- Excellent overall performance
- Model both clusters correctly AND writes good descriptions

### **Low Both**
- Poor performance across all dimensions
- Model needs improvement in both clustering and text generation

## 📈 Best Practices

1. **Primary Metric**: Use ROUGE-L F1 as the main performance indicator
2. **Structure Check**: Monitor ARI and V-Measure for clustering quality
3. **Message Accuracy**: Verify message-level precision/recall
4. **Cluster Count**: Compare predicted vs reference cluster counts
5. **Text Quality**: Ensure cluster titles/descriptions are meaningful

## 🚨 Troubleshooting

### **Import Errors**
```bash
pip install -r requirements.txt
```

### **File Not Found**
- Ensure you're running from the `rouge_evaluation_engine` directory
- Check that `../phases/phase3_clusters.json` exists
- Verify model files are in `../output/phase3_topic_clustering/`

### **ROUGE Calculation Errors**
- Check that cluster titles/descriptions contain meaningful text
- Ensure text fields are not empty or null

## 🔮 Future Enhancements

1. **BERT-based Similarity**: Add BERT embeddings for semantic similarity
2. **Custom Metrics**: Implement domain-specific evaluation criteria
3. **Visualization**: Add charts and graphs for better result interpretation
4. **Batch Processing**: Support for evaluating multiple model families
5. **Export Formats**: Additional output formats (Excel, LaTeX, etc.)

---

**This engine provides the comprehensive, ROUGE-based evaluation that addresses your client's concerns about unlimited topics and summaries!**
