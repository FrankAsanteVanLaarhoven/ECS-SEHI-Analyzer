import numpy as np
from typing import Optional, Tuple
import streamlit as st

class ChemicalMapper:
    """Handle chemical mapping analysis."""
    
    def __init__(self):
        pass
        
    def process_chemical_map(self, data: np.ndarray, wavelength_idx: Optional[int] = None) -> np.ndarray:
        """Process spectral data into chemical map."""
        try:
            # Validate input data
            if not isinstance(data, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(data)}")
                
            # Handle different data dimensions
            if len(data.shape) == 2:
                # Already a 2D chemical map
                return data
            elif len(data.shape) == 3:
                # Spectral data cube - extract specific wavelength
                if wavelength_idx is None:
                    wavelength_idx = data.shape[2] // 2  # Default to middle wavelength
                return data[:, :, wavelength_idx]
            else:
                raise ValueError(f"Unexpected data dimensions: {data.shape}")
                
        except Exception as e:
            raise ValueError(f"Failed to process chemical map: {str(e)}")
            
    def calculate_intensity_map(self, data: np.ndarray, 
                              wavelength_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Calculate intensity map for a wavelength range."""
        try:
            # Validate input data
            if not isinstance(data, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(data)}")
                
            # Handle different data dimensions
            if len(data.shape) == 2:
                return data  # Already 2D intensity map
            elif len(data.shape) == 3:
                if wavelength_range is None:
                    # Use full range
                    return np.mean(data, axis=2)
                else:
                    start_idx, end_idx = wavelength_range
                    return np.mean(data[:, :, start_idx:end_idx], axis=2)
            else:
                raise ValueError(f"Unexpected data dimensions: {data.shape}")
                
        except Exception as e:
            raise ValueError(f"Failed to calculate intensity map: {str(e)}") 