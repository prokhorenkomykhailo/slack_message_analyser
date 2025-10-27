# Phase Evaluation Engine - Project Structure

## 📁 Organized Directory Structure

```
phase_evaluation_engine/
├── 📁 analysis/                    # Analysis scripts and results
│   ├── CSV analysis files
│   ├── Excel analysis files  
│   ├── JSON data files
│   └── Analysis Python scripts
├── 📁 client_deliverables/         # Client-facing documents and results
│   ├── CLIENT_*.md files
│   └── proper_clustering_results/
├── 📁 cohere_tests/               # Cohere AI model testing
│   ├── cohere_*.py files
│   ├── setup_cohere_*.py
│   └── COHERE_INTEGRATION_GUIDE.md
├── 📁 config/                     # Configuration files
│   └── model_config.py
├── 📁 data/                       # Dataset files
│   └── benchmark_topics_corrected_fixed.json
├── 📁 docs/                       # Documentation
│   ├── *.md files
│   ├── *.html files
│   └── Guide documents
├── 📁 evaluations/                # Evaluation engines
│   ├── evaluate_*.py files
│   ├── enhanced_rouge_evaluator.py
│   ├── rouge_evaluation_engine/
│   └── enhanced_rouge_results/
├── 📁 output/                     # Generated output files
│   └── Phase results and logs
├── 📁 phase3_tests/               # Phase 3 (Topic Clustering) tests
│   ├── phase3_*.py files
│   └── run_phase*.py files
├── 📁 phase4_tests/               # Phase 4 (Merge/Split) tests
│   ├── phase4_*.py files
│   └── step2_merge_split_updated.py
├── 📁 phases/                     # Core phase implementations
│   ├── phase3_topic_clustering.py
│   ├── phase4_merge_split.py
│   ├── phase5_metadata_generation.py
│   ├── phase6_embedding.py
│   ├── phase7_user_filtering.py
│   └── phase8_new_message_processing.py
├── 📁 scripts/                    # Utility scripts
│   ├── run_cohere_evaluation.py
│   └── cleanup_duplicate_models.py
├── 📁 tests/                      # Test files
│   ├── test_*.py files
│   ├── check_*.py files
│   ├── debug_*.py files
│   ├── simple_*.py files
│   └── proper_clustering_evaluator.py
├── 📁 utilities/                  # Utility tools
│   ├── convert_*.py files
│   ├── setup_*.py files
│   ├── combine_step1_step2_tokens.py
│   ├── gpu_memory_analysis.py
│   └── requirements_embeddings.txt
├── 📁 utils/                      # Utility modules
│   └── model_clients.py
├── 📁 venv/                       # Virtual environment
├── 📄 requirements.txt            # Main dependencies
├── 📄 run_all_phases.py          # Main runner script
├── 📄 setup.py                   # Setup script
├── 📄 PROJECT_STRUCTURE.md       # This file
└── 📄 README.md                  # Main project README
```

## 🎯 File Categories

### Analysis Files (`analysis/`)
- CSV/Excel analysis results
- JSON data files
- Analysis and calculation scripts
- Score calculation and verification scripts

### Client Deliverables (`client_deliverables/`)
- Client-facing documentation
- Final analysis results
- Delivery summaries

### Cohere Tests (`cohere_tests/`)
- Cohere AI model integration
- Cohere-specific setup and testing
- Cohere integration guides

### Documentation (`docs/`)
- All markdown documentation
- HTML reports
- Phase guides and explanations
- Formula documentation

### Evaluations (`evaluations/`)
- Evaluation engines
- ROUGE evaluation tools
- Model evaluation scripts

### Phase-Specific Tests
- **Phase 3 Tests**: Topic clustering implementations
- **Phase 4 Tests**: Merge/split operations

### Core Phases (`phases/`)
- Main phase implementations
- Core evaluation logic

### Scripts (`scripts/`)
- Utility scripts
- Cleanup scripts

### Tests (`tests/`)
- General test files
- Debug scripts
- Simple test implementations

### Utilities (`utilities/`)
- Conversion tools
- Setup scripts
- Memory analysis tools

## 🚀 Usage

### Main Scripts (Root Level)
- `run_all_phases.py` - Run all phase evaluations
- `setup.py` - Project setup
- `requirements.txt` - Dependencies

### Phase-Specific Testing
- Phase 3: `phase3_tests/`
- Phase 4: `phase4_tests/`

### Analysis and Results
- Analysis: `analysis/`
- Client deliverables: `client_deliverables/`
- Output: `output/`

## 📝 Notes

- Duplicate analysis files have been removed
- Only the most recent/final versions are kept
- All scripts maintain their functionality after reorganization
- Import paths may need updating in some scripts
