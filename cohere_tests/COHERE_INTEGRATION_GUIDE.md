# Cohere Command R+ Integration Guide

The Cohere Command R+ models have been integrated into your existing Phase 3 evaluation system. Here's how to use them:

## 🚀 Quick Start

### Option 1: Run Everything Together (Recommended)
```bash
cd phase_evaluation_engine
python run_phase3_with_cohere.py
```

This will:
- Check and install Cohere dependencies if needed
- Run your existing Phase 3 evaluation
- Add Cohere Command R+ models to the evaluation
- Merge all results together

### Option 2: Run Cohere Models Separately
```bash
cd phase_evaluation_engine

# Install dependencies first
python install_cohere_dependencies.py

# Test setup
python test_cohere_setup.py

# Run Cohere evaluation
python evaluate_cohere_models.py
```

### Option 3: Run All Models Together
```bash
cd phase_evaluation_engine
python evaluate_all_models_with_cohere.py
```

## 📁 Integrated Files

The following files have been moved to `phase_evaluation_engine/`:

- `cohere_clustering.py` - Cohere Command R+ clustering implementation
- `evaluate_cohere_models.py` - Cohere model evaluation script
- `evaluate_all_models_with_cohere.py` - Complete evaluation with all models
- `run_phase3_with_cohere.py` - Enhanced Phase 3 runner with Cohere
- `test_cohere_setup.py` - Setup verification script
- `install_cohere_dependencies.py` - Dependency installer
- `requirements_cohere.txt` - Cohere-specific requirements
- `README_Cohere_Setup.md` - Detailed setup guide

## 🔧 Cohere Models Evaluated

1. **cohere_command-r-plus** - Full precision (104B parameters)
2. **cohere_command-r-plus-8bit** - 8-bit quantized
3. **cohere_command-r-plus-4bit** - 4-bit quantized

## 📊 Integration with Existing System

The Cohere models are fully integrated with your existing evaluation:

- **Same output format** as other models
- **Merged with comprehensive results** in `comprehensive_results.json`
- **Included in CSV analysis** files
- **Compatible with existing analysis scripts**

## 🎯 Expected Results

Cohere Command R+ should perform excellently due to:
- 104B parameters with advanced reasoning
- Specialized RAG and tool use training
- High benchmark scores (74.6 average on Open LLM leaderboard)
- 128K context window for comprehensive analysis

## 🔍 Results Location

After running the evaluation, results will be in:
- `output/phase3_topic_clustering/comprehensive_results.json` - All models
- `output/phase3_topic_clustering/cohere_*.json` - Individual Cohere results
- Updated CSV files with Cohere performance included

## 🐛 Troubleshooting

### Common Issues

1. **Dependencies not installed**
   ```bash
   python install_cohere_dependencies.py
   ```

2. **Hugging Face token missing**
   ```bash
   export HUGGINGFACE_TOKEN=your_token_here
   ```

3. **CUDA out of memory**
   - The system will automatically try quantized versions
   - Ensure you have at least 8GB GPU VRAM

4. **Import errors**
   ```bash
   python test_cohere_setup.py
   ```

## 📈 Performance Comparison

After evaluation, you can compare Cohere against:
- Google Gemini models (Flash, Pro variants)
- OpenAI models (GPT-4o, GPT-3.5-turbo)
- xAI Grok models (Grok-2, Grok-3)
- Google Gemma models (various sizes)

The results will be included in your existing analysis and comparison files.
