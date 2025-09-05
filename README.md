# Phase Evaluation Engine

Comprehensive evaluation engine for testing all AI models across all phases of topic detection and analysis.

## 🎯 Overview

This engine evaluates **37+ models** across **6 providers** (OpenAI, Google, Groq, Anthropic, x.ai, Local) on **6 different phases** of topic processing:

- **Phase 3**: Topic Clustering
- **Phase 4**: Merge/Split Operations  
- **Phase 5**: Metadata Generation
- **Phase 6**: Embedding Generation
- **Phase 7**: User Filtering
- **Phase 8**: New Message Processing

## 🏗️ Architecture

```
phase_evaluation_engine/
├── config/
│   └── model_config.py          # Model configurations and API keys
├── utils/
│   └── model_clients.py         # API client implementations
├── phases/
│   ├── phase3_topic_clustering.py
│   ├── phase4_merge_split.py
│   ├── phase5_metadata_generation.py
│   ├── phase6_embedding.py
│   ├── phase7_user_filtering.py
│   └── phase8_new_message_processing.py
├── output/                      # Evaluation results
├── run_all_phases.py           # Main runner script
└── requirements.txt
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set API Keys
```bash
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"
export GROQ_API_KEY="your-groq-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export XAI_API_KEY="your-xai-key"
```

### 3. Run All Phases
```bash
python run_all_phases.py
```

### 4. Run Individual Phases
```bash
# Topic clustering
python phases/phase3_topic_clustering.py

# Merge/split operations
python phases/phase4_merge_split.py

# Metadata generation
python phases/phase5_metadata_generation.py

# Embedding generation
python phases/phase6_embedding.py

# User filtering
python phases/phase7_user_filtering.py

# New message processing
python phases/phase8_new_message_processing.py
```

## 📊 Supported Models

### OpenAI (3 models)
- gpt-4
- gpt-3.5-turbo
- gpt-4o

### Google (20 models)
- gemini-1.5-flash
- gemini-1.5-pro
- gemini-2.0-flash
- gemini-2.5-flash
- gemini-2.5-pro
- gemma-3-1b-it
- gemma-3-4b-it
- gemma-3-12b-it
- gemma-3-27b-it
- And 11 more...

### Groq (3 models)
- llama3-8b-8192
- llama3-70b-8192
- mixtral-8x7b-32768

### Anthropic (3 models)
- claude-3-opus-20240229
- claude-3-sonnet-20240229
- claude-3-haiku-20240307

### x.ai (3 models)
- grok-2-1212
- grok-2-vision-1212
- grok-3

## 📈 Evaluation Metrics

Each phase evaluates models on:

- **Success Rate**: Percentage of successful API calls
- **Performance**: Response time and token usage
- **Cost Analysis**: API costs per request
- **Quality Metrics**: Phase-specific quality measures
- **Reliability**: Error handling and retry logic

## 📁 Output Structure

```
output/
├── phase3_topic_clustering/
│   ├── openai_gpt-4.json
│   ├── google_gemini-1.5-flash.json
│   └── comprehensive_results.json
├── phase4_merge_split/
│   └── ...
├── phase5_metadata_generation/
│   └── ...
├── phase6_embedding/
│   └── ...
├── phase7_user_filtering/
│   └── ...
├── phase8_new_message_processing/
│   └── ...
└── comprehensive_evaluation_report.json
```

## 🔧 Configuration

### Model Limits
Configure token limits and context lengths in `config/model_config.py`:

```python
MODEL_LIMITS = {
    "gpt-4": {"max_tokens": 8192, "context_length": 8192},
    "gemini-1.5-flash": {"max_tokens": 8192, "context_length": 1000000},
    # ... more models
}
```

### Cost Tracking
Track API costs with per-model pricing:

```python
COST_PER_1K_TOKENS = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    # ... more models
}
```

## 📊 Results Analysis

### Best Performing Models
The engine automatically identifies:
- **Highest Success Rate**: Most reliable models
- **Fastest Response**: Quickest models
- **Most Cost Efficient**: Best value for money
- **Best Quality**: Highest quality outputs per phase

### Comprehensive Reports
Each run generates:
- Individual model results per phase
- Cross-phase performance analysis
- Cost-benefit analysis
- Quality comparison metrics

## 🛠️ Customization

### Add New Models
1. Add model configuration to `config/model_config.py`
2. Implement client in `utils/model_clients.py`
3. Update API key handling

### Add New Phases
1. Create new phase script in `phases/`
2. Implement evaluator class
3. Add to phase list in `run_all_phases.py`

### Custom Prompts
Modify prompts in each phase script to test different approaches.

## 🐛 Troubleshooting

### API Key Issues
- Check environment variables are set correctly
- Verify API keys have sufficient credits
- Ensure API keys have access to required models

### Timeout Issues
- Increase timeout values in model clients
- Check network connectivity
- Consider using faster models for testing

### Memory Issues
- Process fewer models at once
- Reduce batch sizes
- Use models with lower token limits

## 📝 Notes

- **Prerequisites**: Requires `message_dataset.json` and `benchmark_topics.json` from Phase 1 and 2
- **Cost Warning**: Running all models can be expensive. Start with a subset for testing
- **Time**: Full evaluation can take 1-2 hours depending on model availability
- **Storage**: Results can be several GB for comprehensive evaluation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.
