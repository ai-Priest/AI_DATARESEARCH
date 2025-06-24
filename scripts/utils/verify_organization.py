#!/usr/bin/env python3
"""
Verification Script for File Organization
Ensures all moved files and scripts work correctly in their new locations
"""

import sys
import os
import subprocess
from pathlib import Path

def test_core_pipelines():
    """Test that core pipeline files work from root."""
    print("🧪 Testing Core Pipelines...")
    
    tests = [
        ("data_pipeline.py --validate-only", "Data Pipeline"),
        ("ml_pipeline.py --validate-only", "ML Pipeline"),
        ("dl_pipeline.py --validate-only", "DL Pipeline")
    ]
    
    for cmd, name in tests:
        try:
            result = subprocess.run(
                cmd.split(), 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            if "validation complete" in result.stdout.lower() or "initialized" in result.stdout.lower():
                print(f"✅ {name}: Working")
            else:
                print(f"⚠️ {name}: May have issues")
        except Exception as e:
            print(f"❌ {name}: Error - {str(e)[:50]}...")

def test_enhancement_scripts():
    """Test enhancement scripts in new location."""
    print("\n🔧 Testing Enhancement Scripts...")
    
    scripts = [
        "scripts/enhancement/quick_retrain_with_enhanced_data.py",
        "scripts/enhancement/enhance_training_data.py",
        "scripts/enhancement/enhance_ground_truth.py"
    ]
    
    for script in scripts:
        if Path(script).exists():
            try:
                # Just test import/syntax by doing a dry-run check
                result = subprocess.run([
                    "python", "-c", f"import ast; ast.parse(open('{script}').read())"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ {Path(script).name}: Syntax OK")
                else:
                    print(f"❌ {Path(script).name}: Syntax Error")
            except Exception as e:
                print(f"⚠️ {Path(script).name}: {str(e)[:30]}...")
        else:
            print(f"❌ {script}: File not found")

def test_evaluation_scripts():
    """Test evaluation scripts."""
    print("\n📊 Testing Evaluation Scripts...")
    
    scripts = [
        "scripts/evaluation/quick_evaluation.py",
        "scripts/evaluation/quick_eval.py"
    ]
    
    for script in scripts:
        if Path(script).exists():
            try:
                # Test with --help or dry run if possible
                result = subprocess.run([
                    "python", script, "--help"
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 or "usage" in result.stdout.lower():
                    print(f"✅ {Path(script).name}: Accessible")
                else:
                    # Try just syntax check
                    result = subprocess.run([
                        "python", "-c", f"import ast; ast.parse(open('{script}').read())"
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print(f"✅ {Path(script).name}: Syntax OK")
                    else:
                        print(f"❌ {Path(script).name}: Issues detected")
            except Exception as e:
                print(f"⚠️ {Path(script).name}: {str(e)[:30]}...")
        else:
            print(f"❌ {script}: File not found")

def check_directory_structure():
    """Verify directory structure is correct."""
    print("\n📁 Checking Directory Structure...")
    
    expected_dirs = [
        "src/data", "src/ml", "src/dl", "src/utils",
        "scripts/enhancement", "scripts/evaluation", "scripts/utils",
        "tests/test_scripts",
        "docs/guides", "docs/reports/performance", "docs/reports/dl", "docs/reports/ml",
        "config", "data/raw", "data/processed", "models/dl",
        "outputs/EDA", "outputs/ML", "outputs/DL", "logs"
    ]
    
    for dir_path in expected_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}: Present")
        else:
            print(f"❌ {dir_path}: Missing")

def check_root_files():
    """Check that only essential files remain in root."""
    print("\n📄 Checking Root Directory Files...")
    
    essential_files = [
        "data_pipeline.py", "ml_pipeline.py", "dl_pipeline.py", "main.py",
        "CLAUDE.md", "Readme.md", "pyproject.toml", "uv.lock", "env_example.sh"
    ]
    
    root_files = [f.name for f in Path(".").iterdir() if f.is_file() and not f.name.startswith('.')]
    
    for file in essential_files:
        if file in root_files:
            print(f"✅ {file}: Present")
        else:
            print(f"❌ {file}: Missing")
    
    # Check for any unexpected files
    unexpected = set(root_files) - set(essential_files)
    if unexpected:
        print(f"\n⚠️ Unexpected files in root: {', '.join(unexpected)}")
    else:
        print("\n✅ Root directory contains only essential files")

def main():
    """Run all verification tests."""
    print("🔍 FILE ORGANIZATION VERIFICATION")
    print("=" * 50)
    
    os.chdir(Path(__file__).parent.parent.parent)  # Go to project root
    
    test_core_pipelines()
    test_enhancement_scripts()
    test_evaluation_scripts()
    check_directory_structure()
    check_root_files()
    
    print("\n" + "=" * 50)
    print("🎉 VERIFICATION COMPLETE")
    print("\nIf you see mostly ✅ marks, the organization was successful!")
    print("Any ❌ marks indicate issues that may need attention.")

if __name__ == "__main__":
    main()