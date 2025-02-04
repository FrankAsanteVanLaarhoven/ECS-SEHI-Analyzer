import os
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

# Try importing optional dependencies
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    st.warning("h5py not installed. HDF5 file support will be limited.")

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    st.warning("open3d not installed. Point cloud support will be limited.")

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    st.warning("scikit-learn not installed. Some preprocessing features will be limited.")

__all__ = ['SEHIPreprocessor']  # Explicitly declare what should be exported

class SEHIPreprocessor:
    """Handles SEHI data preprocessing with graceful fallbacks."""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.txt']  # Basic formats always supported
        
        # Add format support based on available libraries
        if HAS_H5PY:
            self.supported_formats.extend(['.h5', '.hdf5'])
        if HAS_OPEN3D:
            self.supported_formats.extend(['.ply', '.pcd'])
        
        # Initialize scalers if sklearn is available
        self.scaler = StandardScaler() if HAS_SKLEARN else None

    def _setup_logger(self):
        """Setup logging with proper configuration."""
        try:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            return logger
        except Exception as e:
            st.error(f"Failed to setup logger: {e}")
            raise

    def preprocess_point_cloud(self, data):
        """Preprocess point cloud data."""
        try:
            if data is None:
                return None
                
            normalized_data = self.scaler.fit_transform(data)
            return normalized_data
        except Exception as e:
            st.error(f"Error preprocessing point cloud: {e}")
            return None
        
    def reduce_noise(self, data, noise_level=0.83):
        """Apply noise reduction."""
        try:
            if data is None:
                return None
                
            denoised_data = data * (1 - noise_level * np.random.random(data.shape))
            return denoised_data
        except Exception as e:
            st.error(f"Error reducing noise: {e}")
            return None
        
    def extract_features(self, data, level="Expert"):
        """Extract features based on the specified level."""
        try:
            if data is None:
                return None
                
            if level == "Basic":
                features = {
                    'mean': np.mean(data, axis=0),
                    'std': np.std(data, axis=0),
                    'max': np.max(data, axis=0),
                    'min': np.min(data, axis=0)
                }
            else:  # Expert level
                features = {
                    'mean': np.mean(data, axis=0),
                    'std': np.std(data, axis=0),
                    'max': np.max(data, axis=0),
                    'min': np.min(data, axis=0),
                    'skew': np.mean((data - np.mean(data, axis=0))**3, axis=0),
                    'kurtosis': np.mean((data - np.mean(data, axis=0))**4, axis=0),
                    'percentiles': np.percentile(data, [25, 50, 75], axis=0)
                }
            return features
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            return None

    def load_data(self, file) -> Optional[Dict[str, Any]]:
        """Load data with graceful fallbacks for missing dependencies."""
        try:
            if file is None:
                return None

            file_path = Path(file.name) if hasattr(file, 'name') else Path(file)
            suffix = file_path.suffix.lower()

            if suffix not in self.supported_formats:
                st.error(f"Unsupported file format: {suffix}")
                return None

            # Basic file handling (always available)
            if suffix in ['.csv', '.txt']:
                return self._load_text_data(file)

            # HDF5 handling
            if suffix in ['.h5', '.hdf5']:
                if not HAS_H5PY:
                    st.error("HDF5 support requires h5py package")
                    return None
                return self._load_hdf5_data(file)

            # Point cloud handling
            if suffix in ['.ply', '.pcd']:
                if not HAS_OPEN3D:
                    st.error("Point cloud support requires open3d package")
                    return None
                return self._load_point_cloud(file)

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    def _load_text_data(self, file) -> Optional[Dict[str, Any]]:
        """Load basic text data formats."""
        try:
            # Use numpy for basic file loading
            data = np.loadtxt(file)
            return {
                'data': data,
                'type': 'numeric',
                'shape': data.shape
            }
        except Exception as e:
            st.error(f"Error loading text data: {str(e)}")
            return None

    def _load_hdf5_data(self, file) -> Optional[Dict[str, Any]]:
        """Load HDF5 data if h5py is available."""
        if not HAS_H5PY:
            return None
            
        try:
            with h5py.File(file, 'r') as f:
                return {
                    'data': np.array(f['data']),
                    'type': 'hdf5',
                    'metadata': dict(f.attrs)
                }
        except Exception as e:
            st.error(f"Error loading HDF5 data: {str(e)}")
            return None

    def _load_point_cloud(self, file) -> Optional[Dict[str, Any]]:
        """Load point cloud data if open3d is available."""
        if not HAS_OPEN3D:
            return None
            
        try:
            pcd = o3d.io.read_point_cloud(str(file))
            return {
                'data': np.asarray(pcd.points),
                'type': 'point_cloud',
                'raw_point_cloud': pcd
            }
        except Exception as e:
            st.error(f"Error loading point cloud: {str(e)}")
            return None

    def normalize_data(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Normalize data with fallback for missing sklearn."""
        if data is None:
            return None

        try:
            if HAS_SKLEARN and self.scaler is not None:
                return self.scaler.fit_transform(data)
            else:
                # Basic normalization fallback
                return (data - data.mean()) / data.std()
        except Exception as e:
            st.error(f"Error normalizing data: {str(e)}")
            return None