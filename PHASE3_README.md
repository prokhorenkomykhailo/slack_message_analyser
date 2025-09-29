# Phase 3: Topic Clustering Evaluation


## Dataset

The evaluation uses `data/Synthetic_Slack_Messages.csv` which contains:
- **300 messages** from synthetic Slack conversations
- **Columns**: channel, user_name, user_id, timestamp, text, thread_id
- **Topics**: Multiple project discussions
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
