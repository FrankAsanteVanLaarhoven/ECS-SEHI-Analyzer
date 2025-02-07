#!/bin/bash

# Enable Metal performance shaders for PyTorch
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MPS_DEVICE_MEMORY_LIMIT=0.8  # Use 80% of available VRAM

# Verify conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Deactivate any active conda environment first
conda deactivate 2>/dev/null

# Remove existing environment if it exists
conda env remove -n sehi_env -y

# Create fresh environment
echo "Creating new conda environment 'sehi_env'..."
conda create -n sehi_env python=3.10 -y

# Ensure proper activation of conda environment
eval "$(conda shell.bash hook)"
conda activate sehi_env

# Install PyTorch and dependencies
echo "Installing PyTorch and other dependencies..."
conda install -y -c pytorch -c conda-forge \
    pytorch=2.3.0 \
    torchvision=0.18.0 \
    torchaudio=2.3.0

# Install other dependencies
echo "Installing additional dependencies..."
conda install -y -c conda-forge \
    numpy=1.24.3 \
    pandas=2.0.3 \
    open3d=0.17.0 \
    plotly=5.18.0 \
    scipy=1.11.1 \
    scikit-learn=1.3.0

# Install Streamlit and other pip packages
echo "Installing additional requirements..."
pip install streamlit==1.28.0 pydantic==2.5.3 cryptography==41.0.5

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e .

# Verify installations
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    echo "Error: Failed to import PyTorch"
    exit 1
}

python -c "import streamlit; print(f'Streamlit version: {streamlit.__version__}')" || {
    echo "Error: Failed to import Streamlit"
    exit 1
}

# Run the application with proper module structure
echo "Starting Streamlit application..."
streamlit run ecs_sehi_analyzer.py 