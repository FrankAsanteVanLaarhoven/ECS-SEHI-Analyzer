from functools import wraps
from typing import Callable, TypeVar, Dict, Any
import numpy as np
import streamlit as st

F = TypeVar('F', bound=Callable)

class ValidationError(Exception):
    """Custom exception for validation failures"""

def validate_input(*constraints: Callable):
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for constraint in constraints:
                if not constraint(*args, **kwargs):
                    raise ValidationError(
                        f"Constraint {constraint.__name__} failed"
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator

def non_negative(value: float) -> bool:
    return value >= 0

def valid_resolution(res: int) -> bool:
    return 128 <= res <= 4096 and (res & (res - 1)) == 0  # Power of 2

def validate_session_state():
    """Initialize and validate Streamlit session state variables"""
    
    # Default configuration values
    defaults: Dict[str, Any] = {
        'data_source': "Sample Data",
        'resolution': 512,
        'noise_reduction': 0.5,
        'analysis_method': "Basic",
        'visualization_type': "2D Map",
        'uploaded_file': None,
        'current_tab': "Chemical Analysis",
        'confidence_level': 0.95,
        'show_advanced_options': False,
        'color_scheme': 'viridis',
        'auto_update': True
    }
    
    # Initialize session state variables
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def validate_data_input(data: Any) -> bool:
    """
    Validate input data format and dimensions
    
    Args:
        data: Input data to validate
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    try:
        # Check if data is numpy array
        if not isinstance(data, np.ndarray):
            st.error("Input must be a numpy array")
            return False
            
        # Check dimensions
        if len(data.shape) != 2:
            st.error("Input must be a 2D array")
            return False
            
        # Check data type
        if not np.issubdtype(data.dtype, np.number):
            st.error("Input must contain numerical values")
            return False
            
        # Check for NaN or Inf values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            st.error("Input contains NaN or Inf values")
            return False
            
        # Check minimum size
        if data.shape[0] < 128 or data.shape[1] < 128:
            st.error("Input dimensions must be at least 128x128")
            return False
            
        return True
        
    except Exception as e:
        st.error(f"Validation error: {str(e)}")
        return False

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate analysis configuration parameters
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    # Required configuration parameters
    required_params = {
        'resolution': (int, lambda x: 128 <= x <= 4096),
        'noise_reduction': (float, lambda x: 0.0 <= x <= 1.0),
        'analysis_method': (str, lambda x: x in ["Basic", "Advanced", "Expert"]),
        'visualization_type': (str, lambda x: x in ["2D Map", "3D Surface", "Contour Plot"])
    }
    
    try:
        # Check all required parameters
        for param, (param_type, validator) in required_params.items():
            if param not in config:
                st.error(f"Missing required parameter: {param}")
                return False
                
            value = config[param]
            
            # Check type
            if not isinstance(value, param_type):
                st.error(f"Invalid type for {param}: expected {param_type.__name__}")
                return False
                
            # Check value range/validity
            if not validator(value):
                st.error(f"Invalid value for {param}")
                return False
                
        return True
        
    except Exception as e:
        st.error(f"Configuration validation error: {str(e)}")
        return False

def validate_analysis_results(results: Dict[str, Any]) -> bool:
    """
    Validate analysis results format and content
    
    Args:
        results: Analysis results dictionary to validate
        
    Returns:
        bool: True if results are valid, False otherwise
    """
    required_fields = {
        'timestamp': lambda x: isinstance(x, (str, np.datetime64)),
        'config': lambda x: isinstance(x, dict),
        'results': lambda x: isinstance(x, dict)
    }
    
    try:
        # Check all required fields
        for field, validator in required_fields.items():
            if field not in results:
                st.error(f"Missing required field in results: {field}")
                return False
                
            if not validator(results[field]):
                st.error(f"Invalid type for results field: {field}")
                return False
                
        # Validate specific result metrics based on analysis method
        if 'results' in results:
            metrics = results['results']
            
            # Basic metrics should always be present
            basic_metrics = ['mean', 'std', 'min', 'max']
            for metric in basic_metrics:
                if metric not in metrics:
                    st.error(f"Missing basic metric in results: {metric}")
                    return False
                    
        return True
        
    except Exception as e:
        st.error(f"Results validation error: {str(e)}")
        return False
