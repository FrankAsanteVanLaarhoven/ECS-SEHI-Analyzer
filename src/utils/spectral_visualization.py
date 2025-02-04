import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, Tuple
import streamlit as st
from plotly.subplots import make_subplots

class SpectralVisualizer:
    """Visualize and analyze spectral data from SEHI, LiDAR, and ECS."""
    
    def __init__(self):
        self.spectral_ranges = {
            'sehi': {
                'range': (400, 2500),  # nm
                'key_bands': {
                    'visible': (400, 700),
                    'near_ir': (700, 1400),
                    'short_ir': (1400, 2500)
                },
                'markers': {
                    'carbon': [1200, 1700],  # Carbon absorption bands
                    'metal': [900, 1100],    # Metal oxide features
                    'polymer': [1600, 2200]   # Polymer features
                }
            },
            'lidar': {
                'range': (900, 1064),  # nm
                'resolution': 0.1,      # mm
                'key_features': {
                    'surface': 'Topography and roughness',
                    'depth': 'Layer thickness',
                    'intensity': 'Material properties'
                }
            },
            'ecs': {
                'parameters': {
                    'temperature': {'unit': '°C', 'range': (20, 80)},
                    'pressure': {'unit': 'kPa', 'range': (100, 200)},
                    'humidity': {'unit': '%', 'range': (30, 70)}
                }
            }
        }

    def _preprocess_3d_data(self, data: np.ndarray) -> Tuple[np.ndarray, str]:
        """Preprocess 3D data to 2D with user selection."""
        if len(data.shape) == 3:
            processing_method = st.radio(
                "Select Processing Method",
                ["Select Slice", "Average", "Maximum Intensity"],
                help="Choose how to process 3D data into 2D"
            )

            if processing_method == "Select Slice":
                slice_index = st.slider("Select Slice", 0, data.shape[0] - 1, 0)
                return data[slice_index, :, :], f"Slice {slice_index}"
            elif processing_method == "Average":
                return np.mean(data, axis=0), "Average Projection"
            else:  # Maximum Intensity
                return np.max(data, axis=0), "Maximum Intensity Projection"
        
        return data, "Original Data"

    def _validate_data(self, data: np.ndarray, expected_dims: Union[int, tuple], function_name: str) -> bool:
        """Validate input data dimensions and type."""
        try:
            if not isinstance(data, np.ndarray):
                st.error(f"{function_name}: Input must be a numpy array, got {type(data)}")
                return False
            
            if isinstance(expected_dims, int):
                expected_dims = (expected_dims,)
            
            if len(data.shape) not in expected_dims:
                st.error(f"{function_name}: Expected {expected_dims}D array, got {len(data.shape)}D array with shape {data.shape}")
                return False
                
            if np.isnan(data).any():
                st.warning(f"{function_name}: Data contains NaN values which will be replaced with 0")
                data = np.nan_to_num(data)
                
            return True
        except Exception as e:
            st.error(f"{function_name} validation error: {str(e)}")
            return False

    def visualize_sehi_spectrum(self, wavelengths: np.ndarray, intensities: np.ndarray) -> None:
        """Visualize SEHI spectral data with annotations."""
        try:
            fig = go.Figure()

            # Add main spectrum trace
            fig.add_trace(go.Scatter(
                x=wavelengths,
                y=intensities,
                mode='lines',
                name='SEHI Spectrum',
                line=dict(color='cyan', width=2)
            ))

            # Add key band regions
            for band_name, (start, end) in self.spectral_ranges['sehi']['key_bands'].items():
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor=f"rgba(100,100,100,0.1)",
                    layer="below",
                    line_width=0,
                    annotation_text=band_name.title(),
                    annotation_position="top left"
                )

            # Add marker annotations for key features
            for feature, [start, end] in self.spectral_ranges['sehi']['markers'].items():
                mask = (wavelengths >= start) & (wavelengths <= end)
                if mask.any():
                    peak_idx = np.argmax(intensities[mask])
                    peak_wavelength = wavelengths[mask][peak_idx]
                    peak_intensity = intensities[mask][peak_idx]
                    
                    fig.add_annotation(
                        x=peak_wavelength,
                        y=peak_intensity,
                        text=f"{feature.title()} Feature",
                        showarrow=True,
                        arrowhead=1
                    )

            fig.update_layout(
                title="SEHI Spectral Analysis",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Intensity (a.u.)",
                template="plotly_dark",
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add spectral analysis insights
            with st.expander("🔍 Spectral Analysis Insights"):
                st.markdown("""
                ### Key Features
                - **Visible Range (400-700nm)**: Surface color and visual properties
                - **Near-IR (700-1400nm)**: Molecular bonding and structure
                - **Short-IR (1400-2500nm)**: Chemical composition
                
                ### Analysis Applications
                1. Material identification
                2. Chemical mapping
                3. Degradation assessment
                """)
        except Exception as e:
            st.error(f"Failed to visualize SEHI spectrum: {str(e)}")

    def visualize_lidar_data(self, point_cloud: np.ndarray, intensities: np.ndarray) -> None:
        """Visualize LiDAR data with depth and intensity mapping."""
        try:
            # Create two subplots
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                subplot_titles=('Surface Topology', 'Intensity Map')
            )

            # Add surface plot
            fig.add_trace(
                go.Surface(z=point_cloud, colorscale='Viridis'),
                row=1, col=1
            )

            # Add intensity plot
            fig.add_trace(
                go.Surface(z=intensities, colorscale='Plasma'),
                row=1, col=2
            )

            fig.update_layout(
                title="LiDAR Analysis",
                height=700,
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add LiDAR analysis insights
            with st.expander("🔍 LiDAR Analysis Insights"):
                st.markdown("""
                ### Surface Analysis
                - **Resolution**: 0.1mm spatial resolution
                - **Depth Accuracy**: ±0.05mm
                - **Applications**: Surface roughness, defect detection
                
                ### Intensity Analysis
                - Material properties
                - Surface reflectivity
                - Layer boundaries
                """)
        except Exception as e:
            st.error(f"Failed to visualize LiDAR data: {str(e)}")

    def visualize_ecs_data(self, time_series_data: pd.DataFrame) -> None:
        """Visualize ECS parameters over time with environmental controls."""
        try:
            fig = go.Figure()

            # Add traces for each parameter
            for param, info in self.spectral_ranges['ecs']['parameters'].items():
                fig.add_trace(go.Scatter(
                    x=time_series_data.index,
                    y=time_series_data[param],
                    name=f"{param.title()} ({info['unit']})",
                    mode='lines+markers'
                ))

                # Add reference ranges
                fig.add_hline(
                    y=info['range'][0],
                    line_dash="dash",
                    annotation_text=f"Min {param}"
                )
                fig.add_hline(
                    y=info['range'][1],
                    line_dash="dash",
                    annotation_text=f"Max {param}"
                )

            fig.update_layout(
                title="Environmental Control System Analysis",
                xaxis_title="Time",
                yaxis_title="Parameter Value",
                template="plotly_dark",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add ECS analysis insights
            with st.expander("🔍 ECS Analysis Insights"):
                st.markdown("""
                ### Environmental Parameters
                - **Temperature**: Critical for reaction kinetics
                - **Pressure**: Affects material properties
                - **Humidity**: Influences surface chemistry
                
                ### Control Applications
                1. Process optimization
                2. Quality control
                3. Environmental stability
                """)
        except Exception as e:
            st.error(f"Failed to visualize ECS data: {str(e)}")

    def visualize_chemical_map(self, data: np.ndarray, wavelength: float) -> None:
        """Visualize chemical distribution map at specific wavelength."""
        try:
            st.write("Chemical map data shape:", data.shape)
            
            if len(data.shape) == 3:
                slice_index = st.slider("Select wavelength", 0, data.shape[2]-1, 0)
                data = data[:, :, slice_index]
            
            fig = go.Figure(data=[
                go.Heatmap(
                    z=data,
                    colorscale='Viridis',
                    colorbar=dict(title=f'Intensity at {wavelength:.0f}nm')
                )
            ])

            fig.update_layout(
                title=f"Chemical Distribution Map at {wavelength:.0f}nm",
                xaxis_title="X Position (μm)",
                yaxis_title="Y Position (μm)",
                template="plotly_dark",
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Chemical map visualization failed: {str(e)}")
            st.exception(e)

    def visualize_spectral_profile(self, wavelengths: np.ndarray, intensities: np.ndarray) -> None:
        """Visualize spectral profile with band annotations."""
        try:
            st.write("Wavelengths shape:", wavelengths.shape)
            st.write("Intensities shape:", intensities.shape)
            
            # Ensure 1D arrays
            wavelengths = np.squeeze(wavelengths)
            intensities = np.squeeze(intensities)

            fig = go.Figure(data=[
                go.Scatter(
                    x=wavelengths,
                    y=intensities,
                    mode='lines',
                    name='Spectrum'
                )
            ])

            fig.update_layout(
                title="Spectral Profile",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Intensity (a.u.)",
                template="plotly_dark",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Spectral profile visualization failed: {str(e)}")
            st.exception(e)

    def visualize_3d_surface(self, data: np.ndarray) -> None:
        """Visualize 3D surface plot of the sample."""
        try:
            # Print initial data shape for debugging
            st.write("Initial data shape:", data.shape)

            # Basic data validation
            if not isinstance(data, np.ndarray):
                st.error("Input must be a numpy array")
                return

            # Handle different data dimensions
            if len(data.shape) == 3:
                st.info(f"3D data detected with shape {data.shape}")
                slice_index = st.slider("Select slice", 0, data.shape[0]-1, 0)
                data = data[slice_index]  # Select 2D slice
                st.write("Selected 2D slice shape:", data.shape)
            elif len(data.shape) != 2:
                st.error(f"Unexpected data shape: {data.shape}")
                return

            # Create basic 2D surface plot
            fig = go.Figure(data=[
                go.Surface(
                    z=data,
                    colorscale='Viridis',
                    colorbar=dict(title='Intensity')
                )
            ])

            # Update layout
            fig.update_layout(
                title="3D Surface Analysis",
                scene=dict(
                    xaxis_title="X Position (μm)",
                    yaxis_title="Y Position (μm)",
                    zaxis_title="Intensity",
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2)
                    )
                ),
                template="plotly_dark",
                height=700
            )

            st.plotly_chart(fig, use_container_width=True)

            # Analysis metrics
            with st.expander("Analysis Metrics"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Intensity", f"{np.mean(data):.2f}")
                    st.metric("Std Deviation", f"{np.std(data):.2f}")
                with col2:
                    st.metric("Min Value", f"{np.min(data):.2f}")
                    st.metric("Max Value", f"{np.max(data):.2f}")

        except Exception as e:
            st.error(f"Visualization failed: {str(e)}")
            st.exception(e)

    def _get_quality_description(self, score: float) -> str:
        """Get descriptive text for quality score."""
        if score >= 0.9:
            return "Excellent quality"
        elif score >= 0.8:
            return "Good quality"
        elif score >= 0.6:
            return "Fair quality"
        else:
            return "Needs improvement"

    def visualize(self, wavelengths, spectral_data):
        """Visualize spectral data."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=wavelengths,
            y=spectral_data,
            mode='lines+markers',
            name='Spectral Response',
            line=dict(color='#1E88E5', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Spectral Profile",
            xaxis_title="Wavelength",
            yaxis_title="Intensity",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True) 