from typing import Dict, List, Optional, Union, Any
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import scipy.stats
import time
import logging
import datetime

from src.utils.preprocessing import SEHIPreprocessor
from src.utils.visualization import DataVisualizer
from src.utils.analysis import SEHIAnalyzer
from src.utils.data_generator import SEHISampleData
from src.utils.sustainability_metrics import SustainabilityMetrics
from src.utils.defect_visualization import DefectVisualizer
from src.utils.spectral_visualization import SpectralVisualizer
from src.utils.sgp_visualization import SGPVisualizer
from src.utils.chemical_map import ChemicalMapVisualizer
from src.utils.spectral_profile import SpectralProfileVisualizer
from src.utils.surface_3d import Surface3DPlotter
from src.utils.multimodal_analyzer import MultiModalAnalyzer
from src.utils.model_manager import ModelManager
from src.utils.analysis_playground import AnalysisPlayground
from src.utils.surface_analysis import SurfaceAnalyzer
from src.utils.chemical_analysis import ChemicalAnalyzer
from src.utils.defect_analysis import DefectAnalyzer
from src.utils.surface.surface_analyzer import SurfaceAnalyzer

from .styles import inject_styles
from .pages.overview import render_overview
from .pages.data_analysis import render_data_analysis
from .pages.model_playground import render_model_playground
from .pages.advanced_analysis import render_advanced_analysis
from .pages.defect_detection import render_defect_detection
from .pages.chemical_analysis import render_chemical_analysis
from .pages.surface_analysis import render_surface_analysis
from .pages.analysis_results import render_analysis_results
from .pages.sustainability_metrics import render_sustainability_metrics
from .pages.multimodal_analysis import render_multimodal_analysis
from .pages.data_management import render_data_management
from .pages.sound_therapy import render_sound_therapy, render_floating_player
from .pages.settings import render_settings
from .pages.collaboration import render_collaboration_hub

class Analyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_analysis = None
        self.results_cache = {}
        
    def analyze_multimodal(self, data: Dict, resolution: int) -> Dict:
        """Perform multimodal analysis on the data."""
        # Generate sample multimodal analysis results
        height, width = resolution, resolution
        
        # Chemical distribution map (RGB)
        chemical_map = np.random.normal(0.5, 0.1, (height, width, 3))
        chemical_map = np.clip(chemical_map, 0, 1)
        
        # Surface topology (2D)
        surface_map = np.zeros((height, width))
        x = np.linspace(0, 10, width)
        y = np.linspace(0, 10, height)
        X, Y = np.meshgrid(x, y)
        surface_map = 2 * np.sin(X) * np.cos(Y) + np.random.normal(0, 0.2, (height, width))
        
        # Generate spectral data
        wavelengths = np.linspace(300, 800, 50)
        spectral_data = np.zeros((height, width, len(wavelengths)))
        
        for i in range(height):
            for j in range(width):
                spectrum = (
                    100 * np.exp(-(wavelengths - 400)**2 / 1000) +
                    80 * np.exp(-(wavelengths - 550)**2 / 1000) +
                    60 * np.exp(-(wavelengths - 700)**2 / 1000)
                )
                variation = 0.2 * np.sin(2 * np.pi * i / height) * np.sin(2 * np.pi * j / width)
                spectrum *= (1 + variation)
                noise = np.random.normal(0, 2, len(wavelengths))
                spectral_data[i,j] = spectrum + noise
        
        return {
            'Chemical Distribution': chemical_map,
            'Surface Topology': surface_map,
            'Spectral Features': spectral_data,
            'Statistics': {
                'chemical_mean': float(np.mean(chemical_map)),
                'surface_roughness': float(np.std(surface_map)),
                'spectral_intensity_mean': float(np.mean(spectral_data)),
                'spectral_intensity_std': float(np.std(spectral_data))
            }
        }

    def extract_features(self, data: Dict, feature_depth: int) -> Dict:
        """Extract deep features from the data."""
        height, width = 100, 100
        
        # Generate sample feature maps
        feature_maps = []
        for i in range(feature_depth):
            # Create more interesting feature maps with patterns
            x = np.linspace(0, 10, width)
            y = np.linspace(0, 10, height)
            X, Y = np.meshgrid(x, y)
            
            # Different patterns for different features
            if i % 3 == 0:
                feature_map = np.sin(X) * np.cos(Y)
            elif i % 3 == 1:
                feature_map = np.exp(-(X**2 + Y**2) / 20)
            else:
                feature_map = np.random.normal(0, 1, (height, width))
            
            feature_maps.append(feature_map)
        
        # Generate decreasing feature importance
        feature_importance = np.exp(-np.arange(feature_depth) / 2)
        feature_importance /= feature_importance.sum()
        
        return {
            'feature_maps': feature_maps,
            'feature_importance': feature_importance.tolist(),
            'metadata': {
                'depth': feature_depth,
                'dimensions': (height, width)
            }
        }

    def detect_anomalies(self, data: Dict, sensitivity: float, threshold: float) -> Dict:
        """Detect anomalies in the data."""
        height, width = 100, 100
        
        # Generate base pattern
        x = np.linspace(0, 10, width)
        y = np.linspace(0, 10, height)
        X, Y = np.meshgrid(x, y)
        base = np.sin(X) * np.cos(Y)
        
        # Add random anomalies
        anomalies = np.random.choice(
            [0, 1], 
            size=(height, width), 
            p=[0.95, 0.05]
        )
        anomaly_map = base + anomalies * 5 * sensitivity
        
        # Calculate anomaly scores
        anomaly_score = 1 / (1 + np.exp(-sensitivity * (np.abs(anomaly_map) - threshold)))
        
        return {
            'anomaly_map': anomaly_map,
            'anomaly_score': anomaly_score,
            'statistics': {
                'anomaly_count': int(np.sum(anomaly_score > 0.8)),
                'mean_score': float(np.mean(anomaly_score)),
                'max_score': float(np.max(anomaly_score))
            }
        }

    def visualize_results(self, results: Dict):
        """Visualize analysis results based on their type."""
        if 'Chemical Distribution' in results:
            self._visualize_multimodal_results(results)
        elif 'feature_maps' in results:
            self._visualize_feature_results(results)
        elif 'anomaly_map' in results:
            self._visualize_anomaly_results(results)

    def _visualize_multimodal_results(self, results: Dict):
        """Visualize multimodal analysis results."""
        st.subheader("Multimodal Analysis Results")
        
        # Generate unique timestamp for this visualization
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Create tabs for different visualizations
        tabs = st.tabs(["Chemical", "Surface", "Spectral"])
        
        with tabs[0]:
            # Chemical distribution visualization
            fig = px.imshow(
                results['Chemical Distribution'],
                title="Chemical Distribution Map"
            )
            st.plotly_chart(
                fig, 
                use_container_width=True, 
                key=f"chemical_dist_plot_{timestamp}"
            )
            
            # Display statistics
            st.metric(
                "Mean Chemical Distribution",
                f"{results['Statistics']['chemical_mean']:.3f}"
            )
        
        with tabs[1]:
            # Surface topology visualization
            fig = px.imshow(
                results['Surface Topology'],
                title="Surface Topology Map"
            )
            st.plotly_chart(
                fig, 
                use_container_width=True, 
                key=f"surface_topo_plot_{timestamp}"
            )
            
            st.metric(
                "Surface Roughness",
                f"{results['Statistics']['surface_roughness']:.3f}"
            )
        
        with tabs[2]:
            # Spectral features visualization
            wavelengths = np.linspace(300, 800, results['Spectral Features'].shape[-1])
            avg_spectrum = np.mean(results['Spectral Features'].reshape(-1, results['Spectral Features'].shape[-1]), axis=0)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=wavelengths,
                y=avg_spectrum,
                mode='lines',
                name='Average Spectrum'
            ))
            fig.update_layout(
                title="Average Spectral Response",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Intensity"
            )
            st.plotly_chart(
                fig, 
                use_container_width=True, 
                key=f"spectral_plot_{timestamp}"
            )

    def _visualize_feature_results(self, results: Dict):
        """Visualize feature extraction results."""
        st.subheader("Deep Feature Analysis Results")
        
        # Generate unique timestamp for this visualization
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Display feature maps
        n_features = len(results['feature_maps'])
        cols = st.columns(min(3, n_features))
        
        for i, feature_map in enumerate(results['feature_maps']):
            with cols[i % len(cols)]:
                fig = px.imshow(
                    feature_map,
                    title=f"Feature Map {i+1}"
                )
                st.plotly_chart(
                    fig, 
                    use_container_width=True, 
                    key=f"feature_map_{i}_{timestamp}"
                )
        
        # Feature importance plot
        fig = px.bar(
            x=range(1, len(results['feature_importance']) + 1),
            y=results['feature_importance'],
            title="Feature Importance",
            labels={'x': 'Feature', 'y': 'Importance'}
        )
        st.plotly_chart(
            fig, 
            use_container_width=True, 
            key=f"feature_importance_plot_{timestamp}"
        )

    def _visualize_anomaly_results(self, results: Dict):
        """Visualize anomaly detection results."""
        st.subheader("Anomaly Detection Results")
        
        # Generate unique timestamp for this visualization
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Anomaly map visualization
            fig = px.imshow(
                results['anomaly_map'],
                title="Anomaly Map"
            )
            st.plotly_chart(
                fig, 
                use_container_width=True, 
                key=f"anomaly_map_plot_{timestamp}"
            )
        
        with col2:
            # Anomaly score distribution
            fig = px.histogram(
                results['anomaly_score'].flatten(),
                title="Anomaly Score Distribution",
                labels={'value': 'Anomaly Score', 'count': 'Frequency'}
            )
            st.plotly_chart(
                fig, 
                use_container_width=True, 
                key=f"anomaly_hist_plot_{timestamp}"
            )
        
        # Display metrics
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric(
                "Anomalies Detected", 
                results['statistics']['anomaly_count']
            )
        with metric_cols[1]:
            st.metric(
                "Mean Score", 
                f"{results['statistics']['mean_score']:.3f}"
            )
        with metric_cols[2]:
            st.metric(
                "Max Score", 
                f"{results['statistics']['max_score']:.3f}"
            )

class ModelManager:
    def __init__(self):
        self.models = {}
        self.sample_datasets = self._load_sample_datasets()
    
    def _load_sample_datasets(self):
        try:
            return {
                'particle': self._generate_particle_sample(),
                'surface': self._generate_surface_sample(),
                'composition': self._generate_composition_sample(),
                'spectral': self._generate_spectral_sample()
            }
        except Exception as e:
            st.error(f"Failed to load sample datasets: {str(e)}")
            return {}
    
    def _generate_particle_sample(self):
        return {
            'type': 'particle_analysis',
            'data': np.random.rand(100, 100),
            'metadata': {'sample_type': 'Nanoparticle'}
        }
    
    def _generate_surface_sample(self):
        return {
            'type': 'surface_analysis',
            'data': np.random.rand(100, 100),
            'metadata': {'sample_type': 'Surface'}
        }
    
    def _generate_composition_sample(self):
        return {
            'type': 'composition_analysis',
            'data': np.random.rand(100, 100, 3),
            'metadata': {'sample_type': 'Composition'}
        }
    
    def _generate_spectral_sample(self):
        return {
            'type': 'spectral_analysis',
            'data': np.random.rand(100, 100, 50),
            'metadata': {'sample_type': 'Spectral'}
        }

class Dashboard:
    """Main dashboard class for SEHI Analysis."""
    
    def __init__(self):
        """Initialize dashboard components."""
        self.surface_analyzer = SurfaceAnalyzer()
        self.chemical_analyzer = ChemicalAnalyzer()
        
        # Initialize session state directly to 3D Surface
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "3D Surface"

    def main(self):
        """Main dashboard rendering function."""
        st.title("SEHI Analysis Dashboard")
        
        # Navigation with all tools
        pages = {
            "3D Surface": "🔲",
            "Chemical Analysis": "⚗️",
            "Analysis Results": "📊",
            "Sustainability Metrics": "🌱",
            "Multi-Modal Analysis": "🔄",
            "Data Management": "📂",
            "Sound Therapy": "🎶",
            "Point Cloud & Photogrammetry": "📍",
            "Voice Assistant": "🎙️",  # Added new tab
            "Collaboration": "🤝",
            "Settings": "⚙️"
        }
        
        # Navigation buttons
        cols = st.columns(len(pages))
        for idx, (page, icon) in enumerate(pages.items()):
            with cols[idx]:
                if st.button(f"{icon} {page}", use_container_width=True):
                    st.session_state.current_page = page
        
        st.markdown("---")
        
        # Render selected page
        if st.session_state.current_page == "Point Cloud & Photogrammetry":
            from .pages.point_cloud_editor import render_point_cloud_editor
            render_point_cloud_editor()
        elif st.session_state.current_page == "Collaboration":
            from .pages.collaboration import render_collaboration_hub
            render_collaboration_hub()
        elif st.session_state.current_page == "3D Surface":
            self.render_surface_page()
        elif st.session_state.current_page == "Chemical Analysis":
            self.render_chemical_page()
        elif st.session_state.current_page == "Defect Detection":
            render_defect_detection()
        elif st.session_state.current_page == "Analysis Results":
            render_analysis_results()
        elif st.session_state.current_page == "Sustainability Metrics":
            render_sustainability_metrics()
        elif st.session_state.current_page == "Multi-Modal Analysis":
            render_multimodal_analysis()
        elif st.session_state.current_page == "Data Management":
            render_data_management()
        elif st.session_state.current_page == "Sound Therapy":
            render_sound_therapy()
            render_floating_player()
        elif st.session_state.current_page == "Settings":
            render_settings()
        elif st.session_state.current_page == "Voice Assistant":
            from .pages.voice_assistant import render_voice_assistant
            render_voice_assistant()

    def render_surface_page(self):
        """Render 3D Surface Analysis page."""
        # Create layout
        left_col, main_col = st.columns([1, 3])
        
        with left_col:
            st.subheader("Surface Analysis Controls")
            
            # Controls
            resolution = st.slider(
                "Resolution",
                min_value=128,
                max_value=1024,
                value=512,
                step=128,
                help="Higher resolution provides more detailed surface mapping"
            )
            
            noise_reduction = st.slider(
                "Noise Reduction",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                format="%.2f",
                help="Higher values reduce noise but may smooth out fine details"
            )
            
            view_mode = st.selectbox(
                "View Mode",
                ["Height Map", "Roughness Map", "Gradient Map"]
            )
            
            if st.button("Generate Surface", type="primary"):
                with main_col:
                    with st.spinner("Generating surface..."):
                        # Analyze surface
                        results = self.surface_analyzer.analyze_surface(
                            resolution=resolution,
                            noise_reduction=noise_reduction,
                            view_mode=view_mode
                        )
                        
                        # Display statistics
                        cols = st.columns(4)
                        with cols[0]:
                            st.metric("Mean Height", f"{results['stats']['mean_height']:.2f} nm")
                        with cols[1]:
                            st.metric("RMS Roughness", f"{results['stats']['rms_roughness']:.2f} nm")
                        with cols[2]:
                            st.metric("Peak Height", f"{results['stats']['peak_height']:.2f} nm")
                        with cols[3]:
                            st.metric("Surface Area", f"{results['stats']['surface_area']:.2f} μm²")
                        
                        # Create and display plot
                        fig = self.surface_analyzer.create_surface_plot(
                            results['surface_data'],
                            view_mode
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        with main_col:
            if "surface_generate_btn" not in st.session_state:
                st.markdown("""
                    <div style="text-align: center; padding: 40px;">
                        <h3 style="color: #94A3B8;">Welcome to 3D Surface Analysis</h3>
                        <p style="color: #64748B;">
                            Configure parameters and click 'Generate Surface' to begin.
                            The 3D visualization will appear here.
                        </p>
                    </div>
                """, unsafe_allow_html=True)

    def render_chemical_page(self):
        """Render Chemical Analysis page."""
        # Create layout
        left_col, main_col = st.columns([1, 3])
        
        with left_col:
            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
            st.subheader("Chemical Analysis Controls")
            
            # Element selection
            elements = ["Carbon", "Silicon", "Oxygen", "Nitrogen", "Hydrogen"]
            selected_elements = st.multiselect(
                "Select Elements",
                elements,
                default=["Carbon", "Silicon"],
                help="Choose elements to analyze"
            )
            
            # Analysis parameters
            resolution = st.slider(
                "Resolution",
                min_value=128,
                max_value=1024,
                value=512,
                step=128,
                help="Analysis resolution"
            )
            
            sensitivity = st.slider(
                "Sensitivity",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                format="%.2f",
                help="Analysis sensitivity"
            )
            
            # Advanced options
            with st.expander("Advanced Options"):
                analysis_mode = st.selectbox(
                    "Analysis Mode",
                    ["Standard", "High Precision", "Fast Scan"]
                )
                
                background_correction = st.checkbox(
                    "Background Correction",
                    value=True
                )
            
            # Analysis button
            if st.button("Run Analysis", type="primary"):
                if not selected_elements:
                    st.error("Please select at least one element.")
                    return
                    
                with main_col:
                    with st.spinner("Running chemical analysis..."):
                        # Generate sample data
                        data = np.random.normal(0.5, 0.1, (resolution, resolution))
                        
                        # Run analysis
                        results = self.chemical_analyzer.analyze_composition(
                            data, 
                            selected_elements
                        )
                        
                        # Display results
                        tabs = st.tabs(selected_elements)
                        for element, tab in zip(selected_elements, tabs):
                            with tab:
                                # Show element stats
                                stats = results['stats'][element]
                                cols = st.columns(4)
                                with cols[0]:
                                    st.metric("Mean", f"{stats['mean']:.3f}")
                                with cols[1]:
                                    st.metric("Std Dev", f"{stats['std']:.3f}")
                                with cols[2]:
                                    st.metric("Max", f"{stats['max']:.3f}")
                                with cols[3]:
                                    st.metric("Min", f"{stats['min']:.3f}")
                                
                                # Show element distribution
                                fig = self.chemical_analyzer.create_composition_plot(
                                    results, 
                                    element
                                )
                                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Main visualization area
        with main_col:
            if "chemical_analysis_btn" not in st.session_state:
                st.markdown('<div class="visualization-area">', unsafe_allow_html=True)
                st.markdown("""
                    <div style="text-align: center; padding: 40px;">
                        <h3 style="color: #94A3B8;">Chemical Analysis</h3>
                        <p style="color: #64748B;">
                            Select elements and analysis parameters, then click 
                            'Run Analysis' to begin. Results will appear here.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

# Single dashboard instance
dashboard = Dashboard()

# Run the dashboard
if __name__ == "__main__":
    dashboard.main()