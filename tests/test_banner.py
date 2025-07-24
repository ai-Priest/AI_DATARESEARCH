#!/usr/bin/env python3
"""Test script to verify banner displays correctly with real metrics"""

import sys
import os
sys.path.append('.')

# Set environment variables before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TORCH'] = '1'

from main import print_banner

if __name__ == "__main__":
    print("Testing banner display...")
    print_banner()
    print("Banner test completed.")