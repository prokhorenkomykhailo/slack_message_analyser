#!/usr/bin/env python3
"""
Rename Step 2 Analysis Files to Match Step 1 Format
Renames files to follow the pattern: {model}_updated_analysis_FIXED.xlsx
"""

import os
import shutil
from typing import Dict

def rename_step2_analysis_files():
    """Rename Step 2 analysis files to match Step 1 format"""
    
    source_dir = "output/step2_individual_analysis"
    target_dir = "output/step2_client_analysis"
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Mapping of model names to Step 1 format
    model_mappings = {
        'google_gemini_1.5_flash_step2_analysis.xlsx': 'gemini_1.5_flash_step2_updated_analysis_FIXED.xlsx',
        'google_gemini_1.5_flash_002_step2_analysis.xlsx': 'gemini_1.5_flash_002_step2_updated_analysis_FIXED.xlsx',
        'google_gemini_1.5_flash_8b_step2_analysis.xlsx': 'gemini_1.5_flash_8b_step2_updated_analysis_FIXED.xlsx',
        'google_gemini_1.5_flash_latest_step2_analysis.xlsx': 'gemini_1.5_flash_latest_step2_updated_analysis_FIXED.xlsx',
        'google_gemini_1.5_pro_step2_analysis.xlsx': 'gemini_1.5_pro_step2_updated_analysis_FIXED.xlsx',
        'google_gemini_1.5_pro_002_step2_analysis.xlsx': 'gemini_1.5_pro_002_step2_updated_analysis_FIXED.xlsx',
        'google_gemini_1.5_pro_latest_step2_analysis.xlsx': 'gemini_1.5_pro_latest_step2_updated_analysis_FIXED.xlsx',
        'google_gemini_2.0_flash_step2_analysis.xlsx': 'gemini_2.0_flash_step2_updated_analysis_FIXED.xlsx',
        'google_gemini_2.0_flash_001_step2_analysis.xlsx': 'gemini_2.0_flash_001_step2_updated_analysis_FIXED.xlsx',
        'google_gemini_2.0_flash_lite_step2_analysis.xlsx': 'gemini_2.0_flash_lite_step2_updated_analysis_FIXED.xlsx',
        'google_gemini_2.0_flash_lite_001_step2_analysis.xlsx': 'gemini_2.0_flash_lite_001_step2_updated_analysis_FIXED.xlsx',
        'google_gemini_2.5_flash_step2_analysis.xlsx': 'gemini_2.5_flash_step2_updated_analysis_FIXED.xlsx',
        'google_gemini_2.5_flash_lite_step2_analysis.xlsx': 'gemini_2.5_flash_lite_step2_updated_analysis_FIXED.xlsx',
        'google_gemini_2.5_pro_step2_analysis.xlsx': 'gemini_2.5_pro_step2_updated_analysis_FIXED.xlsx',
        'google_gemma_3_12b_it_step2_analysis.xlsx': 'gemma_3_12b_it_step2_updated_analysis_FIXED.xlsx',
        'google_gemma_3_1b_it_step2_analysis.xlsx': 'gemma_3_1b_it_step2_updated_analysis_FIXED.xlsx',
        'google_gemma_3_27b_it_step2_analysis.xlsx': 'gemma_3_27b_it_step2_updated_analysis_FIXED.xlsx',
        'google_gemma_3_4b_it_step2_analysis.xlsx': 'gemma_3_4b_it_step2_updated_analysis_FIXED.xlsx',
        'google_gemma_3n_e2b_it_step2_analysis.xlsx': 'gemma_3n_e2b_it_step2_updated_analysis_FIXED.xlsx',
        'google_gemma_3n_e4b_it_step2_analysis.xlsx': 'gemma_3n_e4b_it_step2_updated_analysis_FIXED.xlsx',
        'openai_gpt_4o_step2_analysis.xlsx': 'gpt_4o_step2_updated_analysis_FIXED.xlsx',
        'openai_gpt_5_step2_analysis.xlsx': 'gpt_5_step2_updated_analysis_FIXED.xlsx',
        'xai_grok_2_1212_step2_analysis.xlsx': 'grok_2_1212_step2_updated_analysis_FIXED.xlsx',
        'xai_grok_2_vision_1212_step2_analysis.xlsx': 'grok_2_vision_1212_step2_updated_analysis_FIXED.xlsx',
        'xai_grok_3_step2_analysis.xlsx': 'grok_3_step2_updated_analysis_FIXED.xlsx'
    }
    
    print("🔄 Renaming Step 2 analysis files to match Step 1 format...")
    print("=" * 70)
    
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
    summary_source = os.path.join(source_dir, "step2_all_models_summary.xlsx")
    summary_target = os.path.join(target_dir, "step2_all_models_summary_FIXED.xlsx")
    
    if os.path.exists(summary_source):
        shutil.copy2(summary_source, summary_target)
        print(f"✅ step2_all_models_summary.xlsx → step2_all_models_summary_FIXED.xlsx")
        copied_files += 1
    
    print(f"\n✅ Renaming complete!")
    print(f"📁 Copied {copied_files} files to: {target_dir}")
    print(f"📋 Files now follow Step 1 naming convention with Step 2 suffix")
    print(f"   Format: {{model}}_step2_updated_analysis_FIXED.xlsx")

if __name__ == "__main__":
    rename_step2_analysis_files()
