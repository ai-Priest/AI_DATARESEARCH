import os
import subprocess
import sys

# Load .env file
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                try:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
                except ValueError:
                    continue

# Verify key environment variables
print(f"CLAUDE_API_KEY set: {'Yes' if os.environ.get('CLAUDE_API_KEY') else 'No'}")

# Start backend
os.execv(sys.executable, [sys.executable, 'main.py', '--backend'])
