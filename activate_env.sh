#!/bin/bash

# Activate conda environment
conda activate ecs-sehi

# Set environment variables
export PYTHONPATH=src
export STREAMLIT_THEME="dark"

# Print status
echo "ECS SEHI Analysis environment activated!"
echo "Python path: $(which python)"
echo "Streamlit version: $(streamlit --version)"