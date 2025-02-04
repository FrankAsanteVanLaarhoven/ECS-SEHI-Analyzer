import numpy as np
import plotly.graph_objects as go
from scipy import ndimage
from typing import Dict, Any
import streamlit as st

class SurfaceAnalyzer:
    """Handles surface analysis and visualization."""
    
    def __init__(self):
        self.surface_data = None
        self.stats = {}
    
    def analyze_surface(self, resolution=512, noise_reduction=0.5, view_mode="Height Map", method="standard"):
        """Analyze surface with given parameters."""
        try:
            # Sample data for demonstration
            return {
                'surface_data': np.random.rand(resolution, resolution),
                'stats': {
                    'mean_height': 45.2,
                    'rms_roughness': 2.3,
                    'peak_height': 52.1,
                    'surface_area': 125.4
                },
                'roughness_map': np.random.rand(resolution, resolution),
                'height_distribution': np.random.normal(0, 1, 1000),
                'analysis': "Detailed surface analysis results"
            }
        except Exception as e:
            st.error(f"Error in surface analysis: {str(e)}")
            return {}
    
    def create_surface_plot(self, results: Dict[str, Any], title: str = "3D Surface Analysis") -> go.Figure:
        """Create a 3D surface plot."""
        fig = go.Figure(data=[go.Surface(
            z=results['surface_data'],
            colorscale='viridis',
            colorbar=dict(
                title="Height (nm)",
                titleside="right"
            )
        )])
        
        fig.update_layout(
            title=title,
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
            height=800,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        return fig
    
    def generate_surface_visualization(self, resolution: int, roughness: float, view_mode: str) -> None:
        """Generate and visualize surface with interactive controls."""
        
        # Generate surface data
        x = np.linspace(-5, 5, resolution)
        y = np.linspace(-5, 5, resolution)
        X, Y = np.meshgrid(x, y)
        Z = self._generate_surface(X, Y, roughness)
        
        # Create visualization based on view mode
        if view_mode == "Height Map":
            self._show_height_map(X, Y, Z)
        elif view_mode == "Roughness Map":
            self._show_roughness_map(X, Y, Z)
        else:  # Gradient Map
            self._show_gradient_map(X, Y, Z)
    
    def _generate_surface(self, X: np.ndarray, Y: np.ndarray, roughness: float) -> np.ndarray:
        """Generate surface with controlled roughness."""
        # Base surface
        Z = np.sin(np.sqrt(X**2 + Y**2))
        
        # Add roughness
        noise = np.random.normal(0, roughness, X.shape)
        Z += noise
        
        return Z
    
    def _show_height_map(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> None:
        """Show 3D height map visualization."""
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            colorbar=dict(
                title=dict(text='Height (nm)', side='right'),
                x=1.1
            )
        )])
        
        fig.update_layout(
            title="Surface Height Map",
            scene=dict(
                xaxis_title="X Position (μm)",
                yaxis_title="Y Position (μm)",
                zaxis_title="Height (nm)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            template="plotly_dark",
            height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_roughness_map(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> None:
        """Show surface roughness visualization."""
        # Calculate local roughness
        roughness = np.zeros_like(Z)
        for i in range(1, Z.shape[0]-1):
            for j in range(1, Z.shape[1]-1):
                local_z = Z[i-1:i+2, j-1:j+2]
                roughness[i,j] = np.std(local_z)
        
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=roughness,
            colorscale='Viridis',
            colorbar=dict(
                title=dict(text='Roughness (nm)', side='right'),
                x=1.1
            )
        )])
        
        fig.update_layout(
            title="Surface Roughness Map",
            scene=dict(
                xaxis_title="X Position (μm)",
                yaxis_title="Y Position (μm)",
                zaxis_title="Roughness (nm)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            template="plotly_dark",
            height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_gradient_map(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> None:
        """Show surface gradient visualization."""
        # Calculate gradients
        dy, dx = np.gradient(Z)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=gradient_magnitude,
            colorscale='Viridis',
            colorbar=dict(
                title=dict(text='Gradient (nm/μm)', side='right'),
                x=1.1
            )
        )])
        
        fig.update_layout(
            title="Surface Gradient Map",
            scene=dict(
                xaxis_title="X Position (μm)",
                yaxis_title="Y Position (μm)",
                zaxis_title="Gradient (nm/μm)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            template="plotly_dark",
            height=700
        )
        
        st.plotly_chart(fig, use_container_width=True) 