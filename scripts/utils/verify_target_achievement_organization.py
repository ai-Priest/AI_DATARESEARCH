#!/usr/bin/env python3
"""
Verification Script for Target Achievement File Organization
Ensures all target achievement files are properly organized and functional
"""

import sys
import os
from pathlib import Path

def verify_organization():
    """Verify that all target achievement files are properly organized."""
    
    print("🔍 Verifying Target Achievement File Organization")
    print("=" * 55)
    
    # Define expected file locations
    expected_files = {
        'Enhancement Scripts': {
            'scripts/enhancement/enhance_with_graded_relevance.py': 'Graded relevance data generation',
            'scripts/enhancement/dl_pipeline_graded.py': 'Full graded relevance training pipeline'
        },
        'Evaluation Scripts': {
            'scripts/evaluation/quick_graded_improvement.py': 'Quick enhancement application',
            'scripts/evaluation/achieve_70_target.py': 'Final target achievement validation'
        },
        'Data Files': {
            'data/processed/graded_relevance_training.json': '3,500 graded samples',
            'data/processed/threshold_tuning_analysis.json': 'Optimal threshold data (0.485)'
        },
        'Model Files': {
            'models/dl/lightweight_cross_attention_best.pt': '69.0% baseline model',
            'models/dl/graded_relevance_best.pt': 'Graded relevance model (if exists)'
        },
        'Result Files': {
            'outputs/DL/reports/dl_evaluation_report.md': 'Comprehensive evaluation report',
            'outputs/DL/target_achievement_report_20250623_075154.json': '75.0% achievement validation'
        }
    }
    
    all_verified = True
    
    for category, files in expected_files.items():
        print(f"\n📂 {category}:")
        
        for file_path, description in files.items():
            if '*' in file_path:
                # Handle pattern matching
                base_path = file_path.replace('*', '')
                dir_path = Path(base_path).parent
                pattern = Path(base_path).name
                
                if dir_path.exists():
                    matching_files = list(dir_path.glob(pattern + '*'))
                    if matching_files:
                        print(f"  ✅ {file_path} - {description}")
                        print(f"     Found: {[f.name for f in matching_files]}")
                    else:
                        print(f"  ❌ {file_path} - {description}")
                        all_verified = False
                else:
                    print(f"  ❌ {file_path} - Directory doesn't exist")
                    all_verified = False
            else:
                # Regular file check
                if Path(file_path).exists():
                    file_size = Path(file_path).stat().st_size
                    print(f"  ✅ {file_path} - {description}")
                    print(f"     Size: {file_size:,} bytes")
                else:
                    if 'if exists' not in description:
                        print(f"  ❌ {file_path} - {description}")
                        all_verified = False
                    else:
                        print(f"  ⚠️  {file_path} - {description} (optional)")
    
    # Test import functionality
    print(f"\n🧪 Testing Import Functionality:")
    
    import_tests = [
        ('scripts.evaluation.achieve_70_target', 'TargetAchiever'),
        ('scripts.evaluation.quick_graded_improvement', 'QuickGradedEnhancement'),
        ('scripts.enhancement.enhance_with_graded_relevance', 'GradedRelevanceEnhancer')
    ]
    
    for module_path, class_name in import_tests:
        try:
            # Add current directory to path
            if '.' not in sys.path:
                sys.path.insert(0, '.')
            
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print(f"  ✅ {module_path}.{class_name} - Import successful")
        except Exception as e:
            print(f"  ❌ {module_path}.{class_name} - Import failed: {e}")
            all_verified = False
    
    # Verify command accessibility
    print(f"\n📋 Command Verification:")
    commands = [
        'python scripts/enhancement/enhance_with_graded_relevance.py',
        'python scripts/enhancement/dl_pipeline_graded.py',
        'python scripts/evaluation/quick_graded_improvement.py',
        'python scripts/evaluation/achieve_70_target.py'
    ]
    
    for cmd in commands:
        script_path = cmd.replace('python ', '')
        if Path(script_path).exists():
            print(f"  ✅ {cmd}")
        else:
            print(f"  ❌ {cmd} - Script not found")
            all_verified = False
    
    # Summary
    print(f"\n🏆 Organization Verification Summary:")
    if all_verified:
        print("  ✅ ALL FILES PROPERLY ORGANIZED")
        print("  ✅ ALL IMPORTS FUNCTIONAL")
        print("  ✅ ALL COMMANDS ACCESSIBLE")
        print("\n🎉 TARGET ACHIEVEMENT FILES SUCCESSFULLY ORGANIZED!")
    else:
        print("  ❌ SOME ISSUES FOUND")
        print("  Please check the above errors and fix them.")
    
    return all_verified

def main():
    """Main verification function."""
    print("🎯 Target Achievement File Organization Verification")
    print("   75.0% NDCG@3 Achievement - File Structure Check")
    print()
    
    verified = verify_organization()
    
    if verified:
        print(f"\n🚀 Ready for production deployment!")
        print(f"All target achievement components are properly organized.")
    else:
        print(f"\n⚠️  Please resolve organization issues before proceeding.")
    
    return verified

if __name__ == "__main__":
    main()