#!/usr/bin/env python3
"""
Rename Phase 4 Analysis Files to Match Step 1 Format
Renames files to follow the pattern: {model}_updated_analysis_FIXED.xlsx
"""

import os
import shutil
from typing import Dict

def rename_analysis_files():
    """Rename analysis files to match Step 1 format"""
    
    source_dir = "output/individual_model_analysis"
    target_dir = "output/phase4_individual_analysis"
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Mapping of model names to Step 1 format
    model_mappings = {
        'google_gemini_1.5_flash_phase4_analysis.xlsx': 'gemini_1.5_flash_updated_analysis_FIXED.xlsx',
        'google_gemini_1.5_flash_002_phase4_analysis.xlsx': 'gemini_1.5_flash_002_updated_analysis_FIXED.xlsx',
        'google_gemini_1.5_flash_8b_phase4_analysis.xlsx': 'gemini_1.5_flash_8b_updated_analysis_FIXED.xlsx',
        'google_gemini_1.5_flash_latest_phase4_analysis.xlsx': 'gemini_1.5_flash_latest_updated_analysis_FIXED.xlsx',
        'google_gemini_1.5_pro_phase4_analysis.xlsx': 'gemini_1.5_pro_updated_analysis_FIXED.xlsx',
        'google_gemini_1.5_pro_002_phase4_analysis.xlsx': 'gemini_1.5_pro_002_updated_analysis_FIXED.xlsx',
        'google_gemini_1.5_pro_latest_phase4_analysis.xlsx': 'gemini_1.5_pro_latest_updated_analysis_FIXED.xlsx',
        'google_gemini_2.0_flash_phase4_analysis.xlsx': 'gemini_2.0_flash_updated_analysis_FIXED.xlsx',
        'google_gemini_2.0_flash_001_phase4_analysis.xlsx': 'gemini_2.0_flash_001_updated_analysis_FIXED.xlsx',
        'google_gemini_2.0_flash_lite_phase4_analysis.xlsx': 'gemini_2.0_flash_lite_updated_analysis_FIXED.xlsx',
        'google_gemini_2.0_flash_lite_001_phase4_analysis.xlsx': 'gemini_2.0_flash_lite_001_updated_analysis_FIXED.xlsx',
        'google_gemini_2.5_flash_phase4_analysis.xlsx': 'gemini_2.5_flash_updated_analysis_FIXED.xlsx',
        'google_gemini_2.5_flash_lite_phase4_analysis.xlsx': 'gemini_2.5_flash_lite_updated_analysis_FIXED.xlsx',
        'google_gemini_2.5_pro_phase4_analysis.xlsx': 'gemini_2.5_pro_updated_analysis_FIXED.xlsx',
        'google_gemma_3_12b_it_phase4_analysis.xlsx': 'gemma_3_12b_it_updated_analysis_FIXED.xlsx',
        'google_gemma_3_1b_it_phase4_analysis.xlsx': 'gemma_3_1b_it_updated_analysis_FIXED.xlsx',
        'google_gemma_3_27b_it_phase4_analysis.xlsx': 'gemma_3_27b_it_updated_analysis_FIXED.xlsx',
        'google_gemma_3_4b_it_phase4_analysis.xlsx': 'gemma_3_4b_it_updated_analysis_FIXED.xlsx',
        'google_gemma_3n_e2b_it_phase4_analysis.xlsx': 'gemma_3n_e2b_it_updated_analysis_FIXED.xlsx',
        'google_gemma_3n_e4b_it_phase4_analysis.xlsx': 'gemma_3n_e4b_it_updated_analysis_FIXED.xlsx',
        'openai_gpt_4o_phase4_analysis.xlsx': 'gpt_4o_updated_analysis_FIXED.xlsx',
        'openai_gpt_5_phase4_analysis.xlsx': 'gpt_5_updated_analysis_FIXED.xlsx',
        'xai_grok_2_1212_phase4_analysis.xlsx': 'grok_2_1212_updated_analysis_FIXED.xlsx',
        'xai_grok_2_vision_1212_phase4_analysis.xlsx': 'grok_2_vision_1212_updated_analysis_FIXED.xlsx',
        'xai_grok_3_phase4_analysis.xlsx': 'grok_3_updated_analysis_FIXED.xlsx'
    }
    
    print("🔄 Renaming Phase 4 analysis files to match Step 1 format...")
    print("=" * 60)
    
    copied_files = 0
    
    for source_name, target_name in model_mappings.items():
        source_path = os.path.join(source_dir, source_name)
        target_path = os.path.join(target_dir, target_name)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            print(f"✅ {source_name} → {target_name}")
            copied_files += 1
        else:
            print(f"⚠️ Source file not found: {source_name}")
    
    # Copy summary file
    summary_source = os.path.join(source_dir, "all_models_summary.xlsx")
    summary_target = os.path.join(target_dir, "phase4_all_models_summary.xlsx")
    
    if os.path.exists(summary_source):
        shutil.copy2(summary_source, summary_target)
        print(f"✅ all_models_summary.xlsx → phase4_all_models_summary.xlsx")
        copied_files += 1
    
    print(f"\n✅ Renaming complete!")
    print(f"📁 Copied {copied_files} files to: {target_dir}")
    print(f"📋 Files now follow Step 1 naming convention: {{model}}_updated_analysis_FIXED.xlsx")

if __name__ == "__main__":
    rename_analysis_files()
