# Phase 3: Topic Clustering Evaluation

## Overview

Phase 3 evaluates LLM models on their ability to perform topic clustering on synthetic Slack messages. The task involves analyzing 300 messages from a CSV file and grouping them into coherent topic clusters based on thread relationships, participants, semantic similarity, and temporal proximity.

## Dataset

The evaluation uses `data/Synthetic_Slack_Messages.csv` which contains:
- **300 messages** from synthetic Slack conversations
- **Columns**: channel, user_name, user_id, timestamp, text, thread_id
- **Topics**: Multiple project discussions (EcoBloom, FitFusion, TechNova, GreenScape, UrbanEdge)
- **Threads**: Messages are organized into 15 different conversation threads

## Features

### 🚀 **Token Management**
- Automatic token estimation and management
- Handles large message datasets efficiently
- Respects model context limits
- Conservative token allocation (80% for input, 20% for output)

### 📊 **Comprehensive Metrics**
- **Coverage**: Percentage of messages successfully clustered
- **Thread Coherence**: How well thread relationships are preserved
- **Cluster Statistics**: Size distribution, participant analysis
- **Cost Analysis**: Token usage and API costs per model

### 🎯 **Smart Clustering**
- Groups messages by thread relationships
- Considers semantic similarity and temporal proximity
- Identifies project-specific conversations
- Handles cross-channel topic relationships

## How to Run

### Prerequisites

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys** (at least one required):
   ```bash
   export OPENAI_API_KEY="your_openai_key"
   export GOOGLE_API_KEY="your_google_key"
   export GROQ_API_KEY="your_groq_key"
   export ANTHROPIC_API_KEY="your_anthropic_key"
   ```

3. **Ensure CSV file exists**:
   ```bash
   ls data/Synthetic_Slack_Messages.csv
   ```

### Running the Evaluation

#### Option 1: Simple Runner Script
```bash
cd phase_evaluation_engine
python run_phase3.py
```

#### Option 2: Direct Module Execution
```bash
cd phase_evaluation_engine
python phases/phase3_topic_clustering.py
```

#### Option 3: Run All Phases
```bash
cd phase_evaluation_engine
python run_all_phases.py
```

## Output

### Individual Results
Each model evaluation is saved as:
```
output/phase3_topic_clustering/{provider}_{model}.json
```

### Comprehensive Results
All results are compiled in:
```
output/phase3_topic_clustering/comprehensive_results.json
```

### Sample Output Structure
```json
{
  "provider": "openai",
  "model": "gpt-4",
  "phase": "phase3_topic_clustering",
  "timestamp": "2024-01-15T10:30:00",
  "success": true,
  "duration": 45.23,
  "usage": {
    "prompt_tokens": 15000,
    "completion_tokens": 2500,
    "total_tokens": 17500
  },
  "cost": {
    "input_cost": 0.45,
    "output_cost": 0.15,
    "total_cost": 0.60
  },
  "clusters": [
    {
      "cluster_id": "cluster_001",
      "message_ids": [1, 2, 3, 4, 5],
      "draft_title": "EcoBloom Summer Campaign Planning",
      "participants": ["Devon", "Sam", "Leah", "Jordan"],
      "channel": "#campaign-briefs",
      "thread_id": "thread_001"
    }
  ],
  "metrics": {
    "num_clusters": 12,
    "total_messages_clustered": 298,
    "avg_cluster_size": 24.83,
    "coverage": 0.993,
    "thread_coherence": 0.867
  }
}
```

## Evaluation Metrics

### Primary Metrics
- **Coverage**: Percentage of messages successfully assigned to clusters
- **Thread Coherence**: How well the model preserves thread relationships
- **Cluster Quality**: Distribution of cluster sizes and participant overlap

### Performance Metrics
- **Speed**: Response time in seconds
- **Cost**: Total API cost in USD
- **Token Efficiency**: Tokens used per message processed

### Quality Metrics
- **Cluster Titles**: Relevance and descriptiveness of cluster names
- **Participant Analysis**: Accuracy of user grouping
- **Channel Context**: Proper channel assignment

## Model Support

The evaluation supports models from:
- **OpenAI**: GPT-4, GPT-3.5-turbo, GPT-4o
- **Google**: Gemini 1.5/2.0/2.5 models
- **Anthropic**: Claude 3 models
- **Groq**: Llama 3, Mixtral models
- **x.ai**: Grok models

## Troubleshooting

### Common Issues

1. **CSV file not found**:
   ```bash
   # Ensure file exists
   ls -la data/Synthetic_Slack_Messages.csv
   ```

2. **API key errors**:
   ```bash
   # Check environment variables
   echo $OPENAI_API_KEY
   echo $GOOGLE_API_KEY
   ```

3. **Token limit exceeded**:
   - The script automatically handles token limits
   - Uses conservative estimates to prevent errors
   - May use subset of messages for very large datasets

4. **JSON parsing errors**:
   - Models sometimes return malformed JSON
   - Script includes robust error handling
   - Failed responses are logged for analysis

### Performance Tips

1. **For faster evaluation**: Use models with higher context limits
2. **For cost efficiency**: Use models with lower token costs
3. **For accuracy**: Use models with better reasoning capabilities

## Expected Results

### Typical Performance
- **Coverage**: 90-99% of messages should be clustered
- **Thread Coherence**: 80-95% of threads should remain intact
- **Cluster Count**: 10-15 clusters for 300 messages
- **Response Time**: 30-120 seconds depending on model

### Best Performing Models
- **High Coverage**: GPT-4, Claude-3-Opus, Gemini-2.5-Pro
- **Cost Efficient**: Gemini-1.5-Flash, Groq models
- **Fast**: Groq models, Claude-3-Haiku

## Next Steps

After running Phase 3:
1. Review individual model results in the output directory
2. Compare performance across different providers
3. Analyze cluster quality and coherence
4. Use results to select optimal models for your use case
5. Proceed to Phase 4 for additional evaluations

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the comprehensive results for error patterns
3. Examine individual model outputs for specific issues
4. Ensure all dependencies are properly installed
