#!/usr/bin/env python3
"""
Test minimal performance metrics collector
"""

import sys
sys.path.insert(0, 'src')

# Test minimal class definition
class TestPerformanceMetricsCollector:
    def __init__(self):
        self.test = True
    
    def get_test(self):
        return "test successful"

# Test import
if __name__ == "__main__":
    collector = TestPerformanceMetricsCollector()
    print(f"Test result: {collector.get_test()}")