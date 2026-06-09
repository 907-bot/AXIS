#!/bin/bash
# AXIS Startup Script

# Create necessary directories
mkdir -p logs data models .cache

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run server
python backend/main.py