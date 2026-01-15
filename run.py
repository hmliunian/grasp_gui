#!/usr/bin/env python3
"""
Simple startup script for SAM3 GUI
"""
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import and run the app
from sam_gui import app

if __name__ == '__main__':
    print("Starting SAM3 GUI...")
    print("Open your browser to: http://localhost:50052")
    app.run(host='0.0.0.0', port=50052, debug=True)
