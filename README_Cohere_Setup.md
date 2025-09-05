# Cohere Command R+ Integration Guide

This guide will help you set up and test the Cohere Command R+ model for Phase 3 clustering evaluation.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install all required packages
python install_cohere_dependencies.py

# Or install manually:
pip install -r requirements_cohere.txt
pip install 'git+https://github.com/huggingface/transformers.git'
pip install bitsandbytes accelerate
```

### 2. Set Up Hugging Face Token
```bash
# Get your token from: https://huggingface.co/settings/tokens
export HUGGINGFACE_TOKEN=your_token_here

# Or add to .env file:
echo "HUGGINGFACE_TOKEN=your_token_here" >> .env
```

### 3. Test Setup
```bash
# Run the test script to verify everything is working
python test_cohere_setup.py
```

### 4. Run Evaluation
```bash
# Run the Cohere Command R+ evaluation
python phase3_evaluation_with_cohere.py
```

## 📁 Files Created

- `clustering_engine/cohere_clustering.py` - Cohere Command R+ clustering implementation
- `phase3_evaluation_with_cohere.py` - Main evaluation script
- `requirements_cohere.txt` - Required Python packages
- `install_cohere_dependencies.py` - Automated dependency installer
- `test_cohere_setup.py` - Setup verification script
- `README_Cohere_Setup.md` - This guide

## 🔧 Model Variants

The evaluation will test three variants of Cohere Command R+:

1. **Full Precision** (`cohere_command-r-plus`)
   - 104B parameters, full precision
   - Best quality, requires most GPU memory

2. **8-bit Quantized** (`cohere_command-r-plus-8bit`)
   - 104B parameters, 8-bit quantization
   - Good quality, reduced memory usage

3. **4-bit Quantized** (`cohere_command-r-plus-4bit`)
   - 104B parameters, 4-bit quantization
   - Acceptable quality, minimal memory usage

## 💾 System Requirements

### Minimum Requirements
- **GPU**: 8GB VRAM (for 4-bit quantized)
- **RAM**: 16GB system RAM
- **Storage**: 50GB free space
- **Python**: 3.8+

### Recommended Requirements
- **GPU**: 24GB+ VRAM (for full precision)
- **RAM**: 32GB+ system RAM
- **Storage**: 100GB+ free space
- **Python**: 3.9+

## 🐛 Troubleshooting

### Common Issues

1. **"Model not found" error**
   - Ensure you have a valid Hugging Face token
   - Check internet connectivity
   - Verify model access permissions

2. **CUDA out of memory**
   - Try the 4-bit quantized version
   - Close other GPU applications
   - Reduce batch size

3. **Import errors**
   - Run `python install_cohere_dependencies.py`
   - Update pip: `pip install --upgrade pip`
   - Check Python version compatibility

4. **Slow performance**
   - Ensure CUDA is available and working
   - Use quantized versions for faster inference
   - Check GPU utilization

### Getting Help

If you encounter issues:

1. Run the test script: `python test_cohere_setup.py`
2. Check the error messages carefully
3. Verify your system meets the requirements
4. Try the quantized versions if you have memory issues

## 📊 Expected Results

Cohere Command R+ should perform excellently on clustering tasks due to:

- **High benchmark scores** (74.6 average on Open LLM leaderboard)
- **Advanced reasoning capabilities** (70.7 on GSM8K)
- **Strong language understanding** (75.7 on MMLU)
- **Specialized training** for RAG and tool use

The model should create high-quality topic clusters with good semantic coherence and participant grouping.

## 🔄 Integration with Existing System

The Cohere evaluation integrates seamlessly with your existing Phase 3 evaluation:

- Results are saved in the same format as other models
- Automatically merged with existing comprehensive results
- Compatible with your existing analysis scripts
- Generates the same CSV outputs for comparison

## 📈 Performance Comparison

After running the evaluation, you can compare Cohere Command R+ against your existing models:

- **Google Gemini models** (Flash, Pro variants)
- **OpenAI models** (GPT-4o, GPT-3.5-turbo)
- **xAI Grok models** (Grok-2, Grok-3)
- **Google Gemma models** (various sizes)

The results will be included in your comprehensive CSV analysis files.
