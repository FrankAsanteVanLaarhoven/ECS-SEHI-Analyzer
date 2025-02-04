import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import open3d as o3d
from typing import Dict, Any, Optional

class DataVisualizer:
    """Enhanced visualization for multi-modal SEHI analysis."""
    
    def __init__(self):
        self.colorscales = {
            'chemical': 'Viridis',  # For chemical composition
            'height': 'Earth',      # For topography
            'defect': 'RdYlBu',     # For defect analysis
            'uncertainty': 'RdGy'    # For uncertainty visualization
        }
        
        self.default_layout = {
            'template': 'plotly_dark',
            'font': {'family': 'Arial', 'size': 14},
            'margin': dict(l=20, r=20, t=40, b=20),
            'height': 600
        }

        self.material_colors = {
            'catalyst': 'Viridis',
            'fuel_cell': 'Plasma',
            'battery': 'Magma',
            'defect': 'RdYlBu'
        }

    def render_particle_mapping(self, data: Dict[str, np.ndarray] = None) -> None:
        """Render particle distribution map with error handling."""
        try:
            if data is None:
                np.random.seed(42)
                n_particles = 1000
                x = np.random.normal(0, 2, n_particles)
                y = np.random.normal(0, 2, n_particles)
                composition = np.random.choice(['C', 'O', 'Fe'], n_particles)
                
                data = {
                    'X Position (nm)': x,
                    'Y Position (nm)': y,
                    'composition': composition
                }
            
            df = pd.DataFrame(data)
            
            fig = px.scatter(
                df,
                x='X Position (nm)',
                y='Y Position (nm)',
                color='composition',
                title='Particle Distribution Map',
                color_discrete_map={
                    'C': '#1f77b4',
                    'O': '#ff7f0e',
                    'Fe': '#2ca02c'
                }
            )
            
            fig.update_layout(
                **self.default_layout,
                xaxis=dict(
                    range=[-6, 6],
                    gridcolor='rgba(128,128,128,0.2)',
                    zerolinecolor='rgba(128,128,128,0.2)'
                ),
                yaxis=dict(
                    range=[-6, 6],
                    gridcolor='rgba(128,128,128,0.2)',
                    zerolinecolor='rgba(128,128,128,0.2)'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to render particle mapping: {str(e)}")

    def visualize_sehi_data(self, data: np.ndarray, wavelengths: np.ndarray):
        """Create interactive SEHI data visualization."""
        try:
            st.subheader("📊 SEHI Chemical Mapping")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Chemical Map", "Spectral Profile", "3D View"])
            
            with tab1:
                # Interactive channel selection
                channel = st.slider("Select wavelength channel", 
                                  0, len(wavelengths)-1, 
                                  value=len(wavelengths)//2)
                
                fig = px.imshow(
                    data[:, :, channel],
                    title=f"Chemical Distribution at {wavelengths[channel]:.2f} eV",
                    color_continuous_scale=self.colorscales['chemical'],
                    labels={'color': 'Intensity'}
                )
                fig.update_layout(**self.default_layout)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add explanation for kids
                st.info("🔍 What are we looking at?\n\n"
                       "This is like a special map that shows different materials "
                       "in different colors! Brighter colors usually mean more of "
                       "a certain material is present.")
                
            with tab2:
                # Interactive point selection
                col1, col2 = st.columns(2)
                with col1:
                    x = st.slider("X position", 0, data.shape[0]-1)
                with col2:
                    y = st.slider("Y position", 0, data.shape[1]-1)
                
                fig = px.line(
                    x=wavelengths,
                    y=data[x, y, :],
                    title=f"Spectral Profile at Position ({x}, {y})",
                    labels={'x': 'Energy (eV)', 'y': 'Intensity'}
                )
                fig.update_layout(**self.default_layout)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("📈 Understanding the Graph:\n\n"
                       "This line shows how much of different types of light "
                       "are absorbed at the point you selected. Different materials "
                       "create different patterns!")
                
            with tab3:
                # 3D surface plot
                fig = go.Figure(data=[go.Surface(
                    z=data[:, :, channel],
                    colorscale=self.colorscales['chemical']
                )])
                fig.update_layout(
                    **self.default_layout,
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Intensity'
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("🌋 3D View Explained:\n\n"
                       "Imagine this as a landscape where the heights show how much "
                       "of a material is present. Higher peaks mean more material!")
                
        except Exception as e:
            st.error(f"Visualization failed: {str(e)}")

    def visualize_point_cloud(self, point_cloud: o3d.geometry.PointCloud) -> None:
        """Create interactive 3D visualization of point cloud data."""
        try:
            # Convert Open3D point cloud to numpy arrays
            points = np.asarray(point_cloud.points)
            colors = np.asarray(point_cloud.colors)

            # Create 3D scatter plot
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=colors[:, 0] if colors.size > 0 else None,
                        colorscale=self.material_colors['catalyst'],
                        opacity=0.8
                    ),
                    hovertemplate=(
                        'X: %{x:.2f}<br>'
                        'Y: %{y:.2f}<br>'
                        'Z: %{z:.2f}<br>'
                        '<extra></extra>'
                    )
                )
            ])

            # Update layout
            fig.update_layout(
                **self.default_layout,
                scene=dict(
                    xaxis_title="X (μm)",
                    yaxis_title="Y (μm)",
                    zaxis_title="Height (nm)",
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                title="3D Surface Analysis"
            )

            # Add surface statistics
            stats = self._calculate_surface_stats(points)
            
            # Display the 3D visualization
            st.plotly_chart(fig, use_container_width=True)

            # Display surface statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Surface Roughness",
                    f"{stats['roughness']:.2f} nm",
                    delta="Optimal range: 10-50 nm"
                )
            with col2:
                st.metric(
                    "Peak Height",
                    f"{stats['peak_height']:.2f} nm",
                    delta="Target: 100-200 nm"
                )
            with col3:
                st.metric(
                    "Surface Area",
                    f"{stats['surface_area']:.2f} μm²",
                    delta=f"{stats['area_increase']:.1%} increase"
                )

            # Add material-specific insights
            st.info("🔍 Surface Analysis Insights:\n"
                   f"• Average roughness indicates {self._get_roughness_quality(stats['roughness'])}\n"
                   f"• Peak distribution suggests {self._get_peak_quality(stats['peak_height'])}\n"
                   f"• Surface area enhancement shows {self._get_area_quality(stats['area_increase'])}")

        except Exception as e:
            st.error(f"Failed to visualize point cloud: {str(e)}")

    def _calculate_surface_stats(self, points: np.ndarray) -> Dict[str, float]:
        """Calculate surface statistics from point cloud data."""
        try:
            # Calculate surface roughness (RMS height)
            roughness = np.std(points[:, 2])
            
            # Calculate peak height (max - min)
            peak_height = np.max(points[:, 2]) - np.min(points[:, 2])
            
            # Calculate approximate surface area
            xy_area = np.prod(np.max(points[:, :2], axis=0) - np.min(points[:, :2], axis=0))
            surface_area = xy_area * (1 + np.std(points[:, 2]) / np.mean(np.diff(points[:, :2])))
            area_increase = (surface_area / xy_area) - 1

            return {
                'roughness': roughness,
                'peak_height': peak_height,
                'surface_area': surface_area,
                'area_increase': area_increase
            }
        except Exception as e:
            st.error(f"Failed to calculate surface statistics: {str(e)}")
            return {
                'roughness': 0.0,
                'peak_height': 0.0,
                'surface_area': 0.0,
                'area_increase': 0.0
            }

    def _get_roughness_quality(self, roughness: float) -> str:
        """Evaluate surface roughness quality."""
        if 10 <= roughness <= 50:
            return "optimal catalyst distribution"
        elif roughness < 10:
            return "potentially insufficient surface area"
        else:
            return "possible agglomeration"

    def _get_peak_quality(self, peak_height: float) -> str:
        """Evaluate peak height quality."""
        if 100 <= peak_height <= 200:
            return "good material loading"
        elif peak_height < 100:
            return "low material loading"
        else:
            return "possible overloading"

    def _get_area_quality(self, area_increase: float) -> str:
        """Evaluate surface area increase quality."""
        if 0.5 <= area_increase <= 2.0:
            return "optimal surface enhancement"
        elif area_increase < 0.5:
            return "limited active area"
        else:
            return "excellent surface utilization"