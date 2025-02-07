#!/bin/bash

# Ensure we're in the correct conda environment
if [[ "${CONDA_DEFAULT_ENV}" != "sehi_env" ]]; then
    echo "Activating sehi_env..."
    eval "$(conda shell.bash hook)"
    conda activate sehi_env
fi

# Verify environment and dependencies
if ! conda list | grep -q "pytorch"; then
    echo "Error: Not in the correct environment or missing dependencies."
    echo "Please run: conda activate sehi_env"
    exit 1
fi

# Save current working environment
conda env export > environment_working.yml

# Create a backup of the environment with version info
{
    echo "# Working configuration as of $(date)"
    echo "# Environment: ${CONDA_DEFAULT_ENV}"
    
    # Check PyTorch
    if python -c "import torch; print(f'# PyTorch: {torch.__version__}')" 2>/dev/null; then
        python -c "import torch; print(f'# PyTorch: {torch.__version__}')"
    else
        echo "# PyTorch: Not installed"
    fi
    
    # Check Pydantic
    if python -c "import pydantic; print(f'# Pydantic: {pydantic.__version__}')" 2>/dev/null; then
        python -c "import pydantic; print(f'# Pydantic: {pydantic.__version__}')"
    else
        echo "# Pydantic: Not installed"
    fi
    
    # Check Streamlit
    if python -c "import streamlit; print(f'# Streamlit: {streamlit.__version__}')" 2>/dev/null; then
        python -c "import streamlit; print(f'# Streamlit: {streamlit.__version__}')"
    else
        echo "# Streamlit: Not installed"
    fi
    
    # Export full environment
    conda env export
} > environment_backup.yml

# Print confirmation
echo "Environment configurations saved to:"
echo "- environment_working.yml (full export)"
echo "- environment_backup.yml (annotated backup)"

# Print current status
echo -e "\nCurrent environment status:"
echo "Active environment: ${CONDA_DEFAULT_ENV}"
conda list | grep -E "pytorch|pydantic|streamlit" 