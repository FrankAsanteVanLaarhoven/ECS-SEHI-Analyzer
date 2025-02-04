"""Unified surface analysis module."""

import numpy as np
import plotly.graph_objects as go
from scipy import ndimage
from typing import Dict, Any

class SurfaceAnalyzer:
    """Handles 3D surface analysis and visualization."""
    
    def __init__(self):
        self.surface_data = None
        self.stats = {}
    
    def generate_surface(self, resolution: int, noise_reduction: float) -> np.ndarray:
        """Generate a sample surface with interesting features."""
        # Create coordinate grid
        x = np.linspace(-5, 5, resolution)
        y = np.linspace(-5, 5, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Generate surface with multiple features
        Z = (
            2 * np.sin(X) * np.cos(Y) +  # Base pattern
            0.5 * np.exp(-(X**2 + Y**2) / 4) +  # Central peak
            0.3 * np.random.normal(0, 1, (resolution, resolution))  # Random noise
        )
        
        # Apply noise reduction if specified
        if noise_reduction > 0:
            Z = ndimage.gaussian_filter(Z, sigma=noise_reduction)
        
        return Z
    
    def analyze_surface(self, resolution: int, noise_reduction: float, view_mode: str) -> Dict[str, Any]:
        """Analyze surface and return results."""
        # Generate surface
        Z = self.generate_surface(resolution, noise_reduction)
        
        # Calculate statistics
        stats = {
            'mean_height': float(np.mean(Z)),
            'rms_roughness': float(np.std(Z)),
            'peak_height': float(np.max(Z) - np.min(Z)),
            'surface_area': float(np.sum(np.sqrt(1 + np.gradient(Z)[0]**2 + np.gradient(Z)[1]**2)))
        }
        
        # Store results
        self.surface_data = Z
        self.stats = stats
        
        return {
            'surface_data': Z,
            'stats': stats
        }
    
    def create_surface_plot(self, surface_data: np.ndarray, view_mode: str) -> go.Figure:
        """Create an interactive 3D surface plot."""
        # Create the 3D surface plot
        fig = go.Figure(data=[go.Surface(
            z=surface_data,
            colorscale='viridis',
            colorbar=dict(
                title=dict(
                    text="Height (nm)",
                    side="right"
                ),
                thickness=20,
                len=0.75,
                x=0.95
            )
        )])
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"3D Surface Analysis - {view_mode}",
                x=0.5,
                y=0.95
            ),
            scene=dict(
                xaxis_title="X (μm)",
                yaxis_title="Y (μm)",
                zaxis_title="Z (nm)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                aspectratio=dict(x=1, y=1, z=0.7)
            ),
            width=800,
            height=600,
            margin=dict(l=0, r=0, t=30, b=0),
            template="plotly_dark"
        )
        
        return fig

    def analyze_2d(self):
        """2D surface analysis"""
        pass
        
    def analyze_3d(self):
        """3D surface analysis"""
        pass 