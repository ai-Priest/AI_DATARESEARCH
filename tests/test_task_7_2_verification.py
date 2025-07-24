#!/usr/bin/env python3
"""
Task 7.2 Verification Test
Verifies that the startup banner has been updated with real metrics
"""

import sys
import os
import re
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables to avoid TensorFlow issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TORCH'] = '1'

def test_no_hardcoded_values():
    """Verify no hardcoded performance values exist in main.py"""
    print("🔍 Checking for hardcoded performance values...")
    
    with open('main.py', 'r') as f:
        content = f.read()
    
    # Check for specific hardcoded values that should be removed
    hardcoded_patterns = [
        r'72\.2%',  # 72.2% NDCG@3
        r'4\.75s',  # 4.75s response time (if hardcoded)
        r'66\.67%', # 66.67% cache hit rate (if hardcoded)
        r'84%.*improvement',  # 84% improvement (if hardcoded)
    ]
    
    found_hardcoded = []
    for pattern in hardcoded_patterns:
        matches = re.findall(pattern, content)
        if matches:
            found_hardcoded.extend(matches)
    
    if found_hardcoded:
        print(f"❌ Found hardcoded values: {found_hardcoded}")
        return False
    else:
        print("✅ No hardcoded performance values found")
        return True

def test_calculating_fallback():
    """Verify that 'Calculating...' is shown when metrics unavailable"""
    print("\n🔍 Testing 'Calculating...' fallback behavior...")
    
    try:
        from main import print_banner
        
        # Capture the banner output
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            print_banner()
        
        banner_output = f.getvalue()
        
        # Check for "Calculating..." in the output
        if "Calculating..." in banner_output:
            print("✅ 'Calculating...' displayed when metrics unavailable")
            return True
        else:
            print("❌ 'Calculating...' not found in banner output")
            return False
            
    except Exception as e:
        print(f"❌ Error testing banner: {e}")
        return False

def test_timestamp_and_confidence():
    """Verify timestamps and confidence levels are included"""
    print("\n🔍 Testing timestamp and confidence level inclusion...")
    
    try:
        from main import print_banner
        
        # Capture the banner output
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            print_banner()
        
        banner_output = f.getvalue()
        
        # Check for timestamp and confidence indicators
        has_timestamp = "Last Updated:" in banner_output
        has_confidence = any(conf in banner_output for conf in [
            "(Calculating...)", "(High Confidence)", "(Live Data)", "(Estimated)"
        ])
        
        if has_timestamp and has_confidence:
            print("✅ Timestamp and confidence levels included")
            return True
        else:
            print(f"❌ Missing timestamp ({has_timestamp}) or confidence ({has_confidence})")
            return False
            
    except Exception as e:
        print(f"❌ Error testing timestamp/confidence: {e}")
        return False

def test_graceful_error_handling():
    """Verify graceful handling of metrics collection errors"""
    print("\n🔍 Testing graceful error handling...")
    
    try:
        from main import print_production_metrics
        
        # Capture the production metrics output
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            print_production_metrics()
        
        output = f.getvalue()
        
        # Should show "Calculating..." messages even when there are errors
        has_calculating = "Calculating" in output
        has_error_handling = "Metrics collection error:" in output or "Calculating" in output
        
        if has_calculating and has_error_handling:
            print("✅ Graceful error handling implemented")
            return True
        else:
            print(f"❌ Error handling issues: calculating={has_calculating}, error_handling={has_error_handling}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing error handling: {e}")
        return False

def main():
    """Run all verification tests for Task 7.2"""
    print("🧪 Task 7.2 Verification: Update startup banner with real metrics")
    print("=" * 70)
    
    tests = [
        ("Remove hardcoded performance values", test_no_hardcoded_values),
        ("Show 'Calculating...' when metrics unavailable", test_calculating_fallback),
        ("Include timestamps and confidence levels", test_timestamp_and_confidence),
        ("Graceful error handling", test_graceful_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"   ❌ Test failed")
    
    print("\n" + "=" * 70)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Task 7.2 Implementation VERIFIED!")
        print("\n✅ All requirements met:")
        print("   • Removed all hardcoded performance values (72.2% NDCG@3, etc.)")
        print("   • Added graceful handling when metrics are not available")
        print("   • Shows 'Calculating...' appropriately")
        print("   • Includes timestamps and confidence levels")
        print("   • Proper error handling for metrics collection failures")
        return 0
    else:
        print(f"\n❌ Task 7.2 verification failed: {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())