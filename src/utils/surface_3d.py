import plotly.graph_objects as go
import numpy as np
import streamlit as st
import plotly.express as px
import open3d as o3d
from typing import Dict, List, Optional, Tuple

class Surface3DPlotter:
    """Handles 3D surface visualization and analysis."""
    
    def __init__(self):
        self.colorscales = {
            'height': 'viridis',
            'composition': 'plasma',
            'roughness': 'magma',
            'defects': 'inferno'
        }
        self.current_view = None
        self.surface_stats = {}

    def plot_surface(self,
                    x: np.ndarray,
                    y: np.ndarray,
                    z: np.ndarray,
                    colormap: Optional[np.ndarray] = None,
                    title: str = "Surface Analysis") -> None:
        """Create interactive 3D surface plot."""
        try:
            # Create the surface plot
            fig = go.Figure(data=[
                go.Surface(
                    x=x,
                    y=y,
                    z=z,
                    colorscale=self.colorscales['height'],
                    colorbar=dict(title='Height (nm)'),
                    surfacecolor=colormap if colormap is not None else z
                )
            ])
            
            # Update layout
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='X (µm)',
                    yaxis_title='Y (µm)',
                    zaxis_title='Z (nm)',
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                width=800,
                height=800
            )
            
            # Store current view
            self.current_view = {
                'x': x, 'y': y, 'z': z,
                'colormap': colormap
            }
            
            # Calculate surface statistics
            self._calculate_surface_stats()
            
            # Display plot
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Error creating 3D plot: {str(e)}")

    def plot_point_cloud(self,
                        points: np.ndarray,
                        colors: Optional[np.ndarray] = None) -> None:
        """Visualize point cloud data."""
        try:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Convert to plotly format
            points = np.asarray(pcd.points)
            
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=colors if colors is not None else points[:, 2],
                        colorscale=self.colorscales['height']
                    )
                )
            ])
            
            # Update layout
            fig.update_layout(
                title="Point Cloud Visualization",
                scene=dict(
                    xaxis_title='X (µm)',
                    yaxis_title='Y (µm)',
                    zaxis_title='Z (nm)'
                ),
                width=800,
                height=800
            )
            
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Error creating point cloud visualization: {str(e)}")

    def analyze_surface(self,
                       z_data: np.ndarray,
                       pixel_size: float = 1.0) -> Dict[str, float]:
        """Analyze surface properties."""
        try:
            analysis_results = {
                'roughness': self._calculate_roughness(z_data),
                'peak_height': np.max(z_data),
                'valley_depth': np.min(z_data),
                'mean_height': np.mean(z_data),
                'surface_area': self._calculate_surface_area(z_data, pixel_size)
            }
            
            return analysis_results
            
        except Exception as e:
            st.error(f"Error analyzing surface: {str(e)}")
            return {}

    def plot_height_distribution(self, z_data: np.ndarray) -> None:
        """Plot height distribution histogram."""
        try:
            fig = px.histogram(
                z_data.flatten(),
                title="Height Distribution",
                labels={'value': 'Height (nm)', 'count': 'Frequency'},
                nbins=50
            )
            
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Error creating height distribution plot: {str(e)}")

    def _calculate_surface_stats(self) -> None:
        """Calculate surface statistics from current view."""
        if self.current_view is None or 'z' not in self.current_view:
            return
            
        z_data = self.current_view['z']
        
        self.surface_stats = {
            'mean_height': np.mean(z_data),
            'std_height': np.std(z_data),
            'max_height': np.max(z_data),
            'min_height': np.min(z_data),
            'roughness': self._calculate_roughness(z_data)
        }

    def _calculate_roughness(self, z_data: np.ndarray) -> float:
        """Calculate surface roughness (Ra)."""
        try:
            # Remove mean plane
            z_mean = np.mean(z_data)
            z_centered = z_data - z_mean
            
            # Calculate roughness (Ra)
            roughness = np.mean(np.abs(z_centered))
            
            return roughness
            
        except Exception as e:
            st.error(f"Error calculating roughness: {str(e)}")
            return 0.0

    def _calculate_surface_area(self,
                              z_data: np.ndarray,
                              pixel_size: float) -> float:
        """Calculate true surface area using triangulation."""
        try:
            rows, cols = z_data.shape
            surface_area = 0.0
            
            for i in range(rows-1):
                for j in range(cols-1):
                    # Get four corners of each grid cell
                    z1 = z_data[i, j]
                    z2 = z_data[i+1, j]
                    z3 = z_data[i, j+1]
                    z4 = z_data[i+1, j+1]
                    
                    # Calculate areas of two triangles
                    triangle1_area = self._triangle_area(
                        pixel_size, pixel_size,
                        z2-z1, z3-z1
                    )
                    triangle2_area = self._triangle_area(
                        pixel_size, pixel_size,
                        z4-z2, z4-z3
                    )
                    
                    surface_area += triangle1_area + triangle2_area
                    
            return surface_area
            
        except Exception as e:
            st.error(f"Error calculating surface area: {str(e)}")
            return 0.0

    @staticmethod
    def _triangle_area(dx: float, dy: float, dz1: float, dz2: float) -> float:
        """Calculate area of a triangle in 3D space."""
        # Calculate cross product of two vectors
        v1 = np.array([dx, 0, dz1])
        v2 = np.array([0, dy, dz2])
        cross = np.cross(v1, v2)
        
        # Area is half the magnitude of cross product
        return np.linalg.norm(cross) / 2.0

    def display_surface_stats(self) -> None:
        """Display surface statistics."""
        if not self.surface_stats:
            st.warning("No surface statistics available")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Mean Height", f"{self.surface_stats['mean_height']:.2f} nm")
            st.metric("Max Height", f"{self.surface_stats['max_height']:.2f} nm")
            st.metric("Min Height", f"{self.surface_stats['min_height']:.2f} nm")
            
        with col2:
            st.metric("Roughness (Ra)", f"{self.surface_stats['roughness']:.2f} nm")
            st.metric("Height Std Dev", f"{self.surface_stats['std_height']:.2f} nm") 