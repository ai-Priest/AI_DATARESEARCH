#!/usr/bin/env python3
"""
Debug normalization for specific case
"""
import sys
import os
import re
sys.path.append('src')

def debug_normalization():
    query = 'Get me some machine learning datasets about education'
    normalized = query.lower().strip()
    
    print(f"Original: '{query}'")
    print(f"Initial: '{normalized}'")
    
    # Remove common conversational patterns in order
    conversational_patterns = [
        # Complex patterns first
        r'^(can you|could you|please)\s+(find me|get me|show me)\s+',
        r'^(i need|i want|i\'m looking for|looking for)\s+',
        r'^(find me|get me|show me|search for)\s+',
        r'^(can you|could you|please)\s+',
        r'^(find|search)\s+',
        # Remove remaining filler words
        r'\b(some|any)\s+',
        r'\s+(about)\s+',  # Keep space around "about"
        r'\s+(related to|regarding|concerning)\s+',
        # Remove trailing words
        r'\s+(please|thanks|thank you)$',
        r'\s+(data|dataset|datasets)$'
    ]
    
    for i, pattern in enumerate(conversational_patterns):
        before = normalized
        normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE).strip()
        if before != normalized:
            print(f"Step {i+1}: '{before}' -> '{normalized}' (pattern: {pattern})")
    
    # Clean up multiple spaces and extra whitespace
    before = normalized
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    if before != normalized:
        print(f"Cleanup: '{before}' -> '{normalized}'")
    
    print(f"Final: '{normalized}'")

if __name__ == "__main__":
    debug_normalization()