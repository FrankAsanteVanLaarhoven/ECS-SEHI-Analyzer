import plotly.graph_objects as go
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, List, Optional, Tuple

class ChemicalMapVisualizer:
    def __init__(self):
        pass

    def visualize(self, data: np.ndarray, wavelength: float) -> None:
        """Visualize chemical distribution map."""
        try:
            st.write("Chemical map data shape:", data.shape)
            
            # Handle 3D data (spectral cube)
            if len(data.shape) == 3:
                wavelength_idx = st.slider(
                    "Select wavelength channel", 
                    0, data.shape[2]-1, 0, 
                    key="wavelength_channel_slider",
                    help="Choose wavelength channel to display"
                )
                data = data[:, :, wavelength_idx]
            
            # Create base heatmap
            fig = go.Figure()
            
            # Main chemical distribution
            fig.add_trace(go.Heatmap(
                z=data,
                colorscale='Viridis',
                colorbar=dict(
                    title=f'Intensity at {wavelength:.0f}nm',  # Correct property
                    x=1.1
                )
            ))

            # Highlight dark regions (graphite material)
            dark_regions = data < np.percentile(data, 10)
            fig.add_trace(go.Heatmap(
                z=dark_regions,
                colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,255,0.3)']],
                showscale=False,
                name='Graphite Regions'
            ))

            # Highlight bright spots (potential defects)
            bright_spots = data > np.percentile(data, 90)
            fig.add_trace(go.Heatmap(
                z=bright_spots,
                colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(255,0,0,0.3)']],
                showscale=False,
                name='Potential Defects'
            ))

            # Update layout
            fig.update_layout(
                title=f"Chemical Distribution Map at {wavelength:.0f}nm",
                xaxis_title="X Position (μm)",
                yaxis_title="Y Position (μm)",
                template="plotly_dark",
                height=600,
                margin=dict(l=50, r=50, t=50, b=50),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(0,0,0,0.5)"
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add analysis insights
            with st.expander("Chemical Distribution Analysis", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Dark Regions",
                        f"{np.sum(dark_regions)/dark_regions.size*100:.1f}%",
                        help="Percentage of graphite material"
                    )
                with col2:
                    st.metric(
                        "Bright Spots",
                        f"{np.sum(bright_spots)/bright_spots.size*100:.1f}%",
                        help="Percentage of potential defects"
                    )
                with col3:
                    st.metric(
                        "Contrast Ratio",
                        f"{np.max(data)/np.min(data):.2f}",
                        help="Ratio between highest and lowest intensity"
                    )

        except Exception as e:
            st.error(f"Chemical map visualization failed: {str(e)}")
            st.exception(e)

class ChemicalMapper:
    """Maps and analyzes chemical compositions from spectral data."""
    
    def __init__(self):
        self.element_peaks = {
            'C': [284.6, 291.2],   # Carbon peaks
            'O': [532.0, 533.5],   # Oxygen peaks
            'N': [398.5, 400.1],   # Nitrogen peaks
            'Fe': [706.8, 719.9],  # Iron peaks
            'Pt': [71.2, 74.5]     # Platinum peaks
        }
        
        self.phase_signatures = {
            'Metallic': {'peak_ratio': 0.8, 'peak_width': 1.2},
            'Oxide': {'peak_ratio': 1.2, 'peak_width': 1.5},
            'Carbide': {'peak_ratio': 0.9, 'peak_width': 1.0}
        }

    def analyze_composition(self, 
                          spectral_data: np.ndarray,
                          energy_axis: np.ndarray) -> Dict[str, any]:
        """Analyze chemical composition from spectral data."""
        try:
            # Element identification
            elements = self._identify_elements(spectral_data, energy_axis)
            
            # Phase analysis
            phases = self._analyze_phases(spectral_data, energy_axis)
            
            # Composition mapping
            composition_map = self._generate_composition_map(
                spectral_data,
                elements,
                phases
            )
            
            return {
                'elements': elements,
                'phases': phases,
                'composition_map': composition_map,
                'statistics': self._calculate_statistics(composition_map)
            }
            
        except Exception as e:
            st.error(f"Composition analysis failed: {str(e)}")
            return None

    def _identify_elements(self,
                         spectral_data: np.ndarray,
                         energy_axis: np.ndarray) -> Dict[str, float]:
        """Identify elements from spectral peaks."""
        elements = {}
        
        for element, peaks in self.element_peaks.items():
            # Find peaks in the spectrum
            peak_intensities = []
            for peak in peaks:
                # Find nearest energy value
                idx = np.abs(energy_axis - peak).argmin()
                intensity = np.mean(spectral_data[:, idx-2:idx+3])
                peak_intensities.append(intensity)
            
            # Calculate element presence
            elements[element] = {
                'intensity': np.mean(peak_intensities),
                'peaks': peaks,
                'confidence': self._calculate_peak_confidence(peak_intensities)
            }
            
        return elements

    def _analyze_phases(self,
                       spectral_data: np.ndarray,
                       energy_axis: np.ndarray) -> Dict[str, float]:
        """Analyze chemical phases from spectral patterns."""
        phases = {}
        
        for phase_name, signature in self.phase_signatures.items():
            # Calculate phase indicators
            phase_score = self._calculate_phase_score(
                spectral_data,
                signature
            )
            
            phases[phase_name] = {
                'score': phase_score,
                'distribution': self._calculate_phase_distribution(
                    spectral_data,
                    signature
                )
            }
            
        return phases

    def visualize_composition(self,
                            composition_results: Dict[str, any]) -> None:
        """Create interactive visualization of composition analysis."""
        if not composition_results:
            st.warning("No composition data available")
            return
            
        # Create tabs for different visualizations
        tabs = st.tabs(["Elements", "Phases", "Distribution"])
        
        with tabs[0]:
            self._plot_elemental_composition(composition_results['elements'])
            
        with tabs[1]:
            self._plot_phase_distribution(composition_results['phases'])
            
        with tabs[2]:
            self._plot_spatial_distribution(
                composition_results['composition_map']
            )

    def _plot_elemental_composition(self, elements: Dict[str, Dict]) -> None:
        """Plot elemental composition analysis."""
        # Prepare data for plotting
        elements_df = pd.DataFrame([
            {
                'Element': element,
                'Intensity': data['intensity'],
                'Confidence': data['confidence']
            }
            for element, data in elements.items()
        ])
        
        # Create bar plot
        fig = px.bar(
            elements_df,
            x='Element',
            y='Intensity',
            color='Confidence',
            title="Elemental Composition"
        )
        
        st.plotly_chart(fig)

    def _plot_phase_distribution(self, phases: Dict[str, Dict]) -> None:
        """Plot phase distribution analysis."""
        # Create pie chart of phase distribution
        fig = px.pie(
            values=[phase['score'] for phase in phases.values()],
            names=list(phases.keys()),
            title="Phase Distribution"
        )
        
        st.plotly_chart(fig)

    def _plot_spatial_distribution(self, composition_map: np.ndarray) -> None:
        """Plot spatial distribution of compositions."""
        # Create heatmap
        fig = px.imshow(
            composition_map,
            title="Spatial Distribution of Composition"
        )
        
        st.plotly_chart(fig)

    def _calculate_statistics(self, composition_map: np.ndarray) -> Dict[str, float]:
        """Calculate composition statistics."""
        return {
            'mean_composition': np.mean(composition_map),
            'std_composition': np.std(composition_map),
            'uniformity_index': 1 - (np.std(composition_map) / np.mean(composition_map))
        }

    def _calculate_peak_confidence(self, peak_intensities: List[float]) -> float:
        """Calculate confidence score for peak identification."""
        # Simple confidence calculation based on peak intensity and consistency
        mean_intensity = np.mean(peak_intensities)
        std_intensity = np.std(peak_intensities)
        
        if mean_intensity == 0:
            return 0.0
            
        return 1.0 - (std_intensity / mean_intensity)

    def _calculate_phase_score(self,
                             spectral_data: np.ndarray,
                             signature: Dict) -> float:
        """Calculate phase presence score."""
        # Implement phase scoring logic
        return np.random.random()  # Placeholder

    def _calculate_phase_distribution(self,
                                   spectral_data: np.ndarray,
                                   signature: Dict) -> np.ndarray:
        """Calculate spatial distribution of phases."""
        # Implement phase distribution calculation
        return np.random.random(spectral_data.shape[:-1])  # Placeholder

    def _generate_composition_map(self,
                                spectral_data: np.ndarray,
                                elements: Dict[str, Dict],
                                phases: Dict[str, Dict]) -> np.ndarray:
        """Generate spatial map of composition."""
        # Implement composition mapping logic
        return np.random.random(spectral_data.shape[:-1])  # Placeholder 