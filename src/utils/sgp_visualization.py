import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Tuple, Optional
import streamlit as st
from scipy import ndimage, stats
import plotly.express as px
import pandas as pd

class SGPVisualizer:
    """Surface Growth Process (SGP) visualization and analysis tool."""
    
    def __init__(self):
        self.growth_parameters = {
            "Nucleation Rate": (0.001, 0.1),
            "Growth Rate": (0.1, 2.0),
            "Surface Diffusion": (0.01, 1.0),
            "Temperature": (300, 1000),  # Kelvin
            "Pressure": (1e-6, 1e-3)     # Torr
        }
        
        self.color_scales = {
            "Height": "Viridis",
            "Growth Rate": "Plasma",
            "Temperature": "Thermal",
            "Composition": "RdYlBu"
        }

    def show_sgp_analysis(self, surface_data: np.ndarray, time_series: Optional[np.ndarray] = None):
        """Display comprehensive SGP analysis tools."""
        st.subheader("Surface Growth Process Analysis")
        
        # Create analysis tabs
        tabs = st.tabs([
            "3D Growth Visualization",
            "Growth Kinetics",
            "Surface Evolution",
            "Process Parameters"
        ])
        
        with tabs[0]:
            self._show_3d_growth(surface_data)
            
        with tabs[1]:
            self._show_growth_kinetics(surface_data, time_series)
            
        with tabs[2]:
            self._show_surface_evolution(surface_data)
            
        with tabs[3]:
            self._show_process_parameters()

    def _show_3d_growth(self, surface_data: np.ndarray):
        """Show 3D visualization of growth process."""
        fig = go.Figure(data=[go.Surface(z=surface_data)])
        
        fig.update_layout(
            title="Surface Growth Visualization",
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position",
                zaxis_title="Height"
            ),
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _show_growth_kinetics(self, surface_data: np.ndarray, time_series: Optional[np.ndarray] = None):
        """Show growth kinetics analysis."""
        if time_series is not None:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=np.arange(len(time_series)),
                y=time_series,
                mode='lines+markers',
                name='Growth Rate'
            ))
            
            fig.update_layout(
                title="Growth Kinetics",
                xaxis_title="Time Step",
                yaxis_title="Growth Rate",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def _show_surface_evolution(self, surface_data: np.ndarray):
        """Show surface evolution analysis."""
        # Calculate surface statistics
        roughness = np.std(surface_data)
        mean_height = np.mean(surface_data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Surface Roughness", f"{roughness:.2f} nm")
        with col2:
            st.metric("Mean Height", f"{mean_height:.2f} nm")
        
        # Show height distribution
        fig = px.histogram(
            surface_data.flatten(),
            title="Height Distribution",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    def _show_process_parameters(self):
        """Show process parameters controls and visualization."""
        st.write("### Process Parameters")
        
        # Parameter controls
        cols = st.columns(3)
        params = {}
        
        for i, (param, (min_val, max_val)) in enumerate(self.growth_parameters.items()):
            with cols[i % 3]:
                params[param] = st.slider(
                    param,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val) / 2),
                    format="%.3f"
                )
        
        # Show parameter relationships
        if st.checkbox("Show Parameter Relationships"):
            fig = px.scatter_matrix(
                pd.DataFrame(params, index=[0]),
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

    def create_animation(self, time_series_data: List[np.ndarray], speed: float = 1.0):
        """Create animated visualization of growth process."""
        frames = []
        for i, surface in enumerate(time_series_data):
            frames.append(
                go.Frame(
                    data=[go.Surface(z=surface)],
                    name=f"frame{i}"
                )
            )
        
        fig = go.Figure(
            frames=frames,
            layout=go.Layout(
                scene=dict(
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    ),
                    aspectmode='data'
                ),
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 50/speed, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate'
                        }]
                    }]
                }],
                template="plotly_dark"
            )
        )
        
        return fig

    def create_cross_section_plot(self, surface_data: np.ndarray):
        """Create interactive cross-section visualization."""
        try:
            x_section = surface_data[surface_data.shape[0]//2, :]
            y_section = surface_data[:, surface_data.shape[1]//2]
            diagonal = np.diagonal(surface_data)
            
            fig = go.Figure()
            
            # Add traces for each section
            fig.add_trace(go.Scatter(
                y=x_section,
                name='X Cross-section',
                visible=True
            ))
            
            fig.add_trace(go.Scatter(
                y=y_section,
                name='Y Cross-section',
                visible=False
            ))
            
            fig.add_trace(go.Scatter(
                y=diagonal,
                name='Diagonal Cross-section',
                visible=False
            ))
            
            # Update layout
            fig.update_layout(
                title="Surface Cross-sections",
                xaxis_title="Position",
                yaxis_title="Height",
                template="plotly_dark",
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="right",
                        x=0.7,
                        y=1.2,
                        showactive=True,
                        buttons=[
                            dict(
                                label="X-Section",
                                method="update",
                                args=[{"visible": [True, False, False]}]
                            ),
                            dict(
                                label="Y-Section",
                                method="update",
                                args=[{"visible": [False, True, False]}]
                            ),
                            dict(
                                label="Diagonal",
                                method="update",
                                args=[{"visible": [False, False, True]}]
                            )
                        ]
                    )
                ]
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating cross-section plot: {str(e)}")
            return None

    def create_sgp_analysis(self, data):
        """Create comprehensive SGP analysis visualization."""
        try:
            surface = data['surface']
            
            # Calculate surface statistics
            roughness = np.std(surface)
            mean_height = np.mean(surface)
            peak_height = np.max(surface) - np.min(surface)
            
            # Create metrics display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RMS Roughness", f"{roughness:.2f} nm")
            with col2:
                st.metric("Mean Height", f"{mean_height:.2f} nm")
            with col3:
                st.metric("Peak Height", f"{peak_height:.2f} nm")
            
            return True
            
        except Exception as e:
            st.error(f"Error in SGP analysis: {str(e)}")
            return False 