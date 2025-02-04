import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import open3d as o3d
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timedelta

class SEHISampleData:
    """Generate sample data for SEHI analysis dashboard."""
    
    def __init__(self):
        self.data_path = Path(__file__).parent.parent / "data"
        self.data_path.mkdir(exist_ok=True)
        self.rng = np.random.default_rng(42)  # For reproducibility
        
    def generate_wavelengths(self):
        """Generate sample wavelength data."""
        return np.linspace(400, 800, 100)
    
    def generate_chemical_map(self):
        """Generate sample chemical mapping data."""
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        return np.sin(np.sqrt(X**2 + Y**2))
    
    def generate_spectral_data(self):
        """Generate sample spectral data."""
        return np.random.normal(size=(100, 100, 100))

    def generate_intensities(self) -> np.ndarray:
        """Generate sample spectral intensities."""
        wavelengths = self.generate_wavelengths()
        # Create synthetic spectrum with peaks
        intensities = np.zeros_like(wavelengths)
        
        # Add some characteristic peaks
        peaks = [(1200, 100), (1700, 80), (900, 120), (2200, 90)]
        for pos, height in peaks:
            intensities += height * np.exp(-(wavelengths - pos)**2 / 1000)
            
        # Add noise
        intensities += self.rng.normal(0, 5, size=len(wavelengths))
        return intensities
        
    def generate_point_cloud(self) -> np.ndarray:
        """Generate sample point cloud data."""
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        X, Y = np.meshgrid(x, y)
        
        # Create surface with features
        Z = (np.sin(2*np.pi*X) * np.cos(2*np.pi*Y) + 
             self.rng.normal(0, 0.1, size=X.shape))
        return Z
        
    def generate_sehi_sample(self) -> np.ndarray:
        """Generate sample SEHI data."""
        size = (100, 100)
        base = np.zeros(size)
        
        # Add some features
        x = np.linspace(-5, 5, size[0])
        y = np.linspace(-5, 5, size[1])
        X, Y = np.meshgrid(x, y)
        base += np.exp(-(X**2 + Y**2)/10)
        
        # Add noise
        base += self.rng.normal(0, 0.1, size=size)
        return base
        
    def generate_ecs_data(self) -> pd.DataFrame:
        """Generate sample environmental control system data."""
        # Generate timestamps for the last 24 hours
        timestamps = [
            datetime.now() - timedelta(hours=x) 
            for x in range(24*60)
        ]  # Minute-by-minute data
        
        # Generate parameter values
        data = {
            'temperature': self._generate_parameter_series(
                mean=50, std=5, n=len(timestamps)
            ),
            'pressure': self._generate_parameter_series(
                mean=150, std=10, n=len(timestamps)
            ),
            'humidity': self._generate_parameter_series(
                mean=50, std=7, n=len(timestamps)
            )
        }
        
        return pd.DataFrame(data, index=timestamps)
        
    def generate_spatial_coordinates(self) -> np.ndarray:
        """Generate spatial coordinates for SGP analysis."""
        return np.linspace(0, 10, 100).reshape(-1, 1)
        
    def generate_predictions(self) -> np.ndarray:
        """Generate sample predictions for SGP analysis."""
        X = self.generate_spatial_coordinates()
        return np.sin(X.flatten()) + 0.1 * self.rng.normal(size=len(X))
        
    def generate_uncertainties(self) -> np.ndarray:
        """Generate sample uncertainty values for SGP analysis."""
        X = self.generate_spatial_coordinates()
        base_uncertainty = 0.1 + 0.1 * np.abs(np.sin(X.flatten()))
        return base_uncertainty + 0.05 * self.rng.normal(size=len(X))
        
    def _generate_parameter_series(self, 
                                 mean: float, 
                                 std: float, 
                                 n: int) -> np.ndarray:
        """Generate a smooth time series with noise."""
        # Generate smooth trend
        t = np.linspace(0, 4*np.pi, n)
        trend = mean + std * 0.5 * np.sin(t)
        
        # Add noise
        noise = self.rng.normal(0, std*0.1, size=n)
        return trend + noise
        
    def generate_point_cloud_sample(self, n_points=1000):
        """Generate sample 3D point cloud with realistic surface features."""
        # Create base surface with realistic topography
        x = np.random.uniform(-1, 1, n_points)
        y = np.random.uniform(-1, 1, n_points)
        z = 0.2 * np.sin(5*x) + 0.3 * np.cos(5*y)  # Surface features
        
        # Add some random surface defects
        defects = np.random.choice([0, 1], size=n_points, p=[0.9, 0.1])
        z += defects * np.random.normal(0, 0.1, n_points)
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        points = np.column_stack([x, y, z])
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Add colors based on height (blue-green-red colormap)
        colors = np.zeros((n_points, 3))
        normalized_z = (z - z.min()) / (z.max() - z.min())
        colors[:, 0] = normalized_z  # Red channel
        colors[:, 2] = 1 - normalized_z  # Blue channel
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save sample data
        o3d.io.write_point_cloud(str(self.data_path / "sample_surface.ply"), pcd)
        return pcd
        
    def _create_spectral_signature(self, peaks, width=2.0):
        """Create realistic spectral signature with given peaks."""
        x = np.linspace(0, 31, 32)
        signature = np.zeros_like(x)
        
        for peak in peaks:
            signature += np.exp(-(x - peak)**2 / width)
            
        return signature / signature.max() 