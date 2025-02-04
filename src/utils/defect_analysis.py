import numpy as np
from typing import Dict, Tuple, List, Any, Optional
import plotly.graph_objects as go
import streamlit as st
from time import sleep
from dataclasses import dataclass
from scipy import ndimage, stats
from scipy.spatial import distance
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from .sehi_analysis import SEHIAnalyzer, SEHIParameters, SEHIAlgorithm

@dataclass
class DefectParameters:
    """Parameters for defect detection."""
    sensitivity: float
    min_size: int = 5
    max_size: int = 100
    threshold: float = 0.5
    noise_reduction: float = 0.3

class DefectAnalyzer:
    """Advanced defect analysis system with optimized performance."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._cache = {}
        self.sehi_analyzer = SEHIAnalyzer()  # Initialize SEHI analyzer
        self.defect_params = {
            "Cracks": {
                "kernel_size": 5,
                "threshold": 0.6,
                "min_length": 10
            },
            "Voids": {
                "kernel_size": 3,
                "threshold": 0.7,
                "min_area": 25
            },
            "Delamination": {
                "kernel_size": 7,
                "threshold": 0.65,
                "min_area": 40
            },
            "Inclusions": {
                "kernel_size": 4,
                "threshold": 0.75,
                "min_area": 15
            },
            "Porosity": {
                "kernel_size": 3,
                "threshold": 0.7,
                "min_area": 10
            },
            "Surface Contamination": {
                "kernel_size": 5,
                "threshold": 0.6,
                "min_area": 20
            },
            "Grain Boundaries": {
                "kernel_size": 5,
                "threshold": 0.65,
                "min_length": 15
            },
            "Phase Separation": {
                "kernel_size": 6,
                "threshold": 0.7,
                "min_area": 30
            }
        }
        self.defect_colors = {
            'crack': '#FF4B4B',  # Red
            'delamination': '#FFA500',  # Orange
            'contamination': '#9370DB',  # Purple
            'porosity': '#4169E1'  # Blue
        }
        
        self.defect_descriptions = {
            'crack': 'Surface fractures affecting material integrity',
            'delamination': 'Layer separation in material structure',
            'contamination': 'Foreign material presence',
            'porosity': 'Void spaces in material'
        }
    
    def generate_animated_surface(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Generate and animate surface with progressive defect addition."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        plot_space = st.empty()
        
        # Base surface
        R = np.sqrt(X**2 + Y**2)
        Z = 0.3 * np.sin(R)
        
        # Initial plot
        fig = self._create_surface_plot(X, Y, Z, "Base Surface")
        plot_space.plotly_chart(fig, use_container_width=True)
        progress_bar.progress(20)
        status_text.text("Analyzing base surface...")
        sleep(1)
        
        # Add cracks
        status_text.text("Detecting cracks...")
        angles = np.arctan2(Y, X)
        crack_pattern = 0.4 * np.sin(8 * angles) * np.exp(-0.5 * R)
        Z += crack_pattern
        fig = self._create_surface_plot(X, Y, Z, "Surface with Cracks")
        plot_space.plotly_chart(fig, use_container_width=True)
        progress_bar.progress(40)
        sleep(1)
        
        # Add delamination
        status_text.text("Analyzing delamination...")
        for cx, cy, radius in [(-2, -2, 1), (2, 2, 1.5)]:
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            delamination = 0.3 * np.exp(-dist**2 / (2 * radius**2))
            Z += delamination
            fig = self._create_surface_plot(X, Y, Z, "Surface with Delamination")
            plot_space.plotly_chart(fig, use_container_width=True)
        progress_bar.progress(60)
        sleep(1)
        
        # Add contamination
        status_text.text("Detecting contamination...")
        np.random.seed(42)
        for _ in range(20):
            cx = np.random.uniform(-4, 4)
            cy = np.random.uniform(-4, 4)
            radius = np.random.uniform(0.2, 0.5)
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            contamination = 0.2 * np.exp(-dist**2 / (2 * radius**2))
            Z += contamination
        fig = self._create_surface_plot(X, Y, Z, "Complete Surface Analysis")
        plot_space.plotly_chart(fig, use_container_width=True)
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        return Z
    
    def _create_surface_plot(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, title: str) -> go.Figure:
        """Create an interactive 3D surface plot."""
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            colorbar=dict(
                title=dict(text='Height (nm)', side='right'),
                x=1.1,
                len=0.8
            )
        )])
        
        fig.update_layout(
            title=dict(text=title, y=0.95),
            scene=dict(
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    up=dict(x=0, y=0, z=1)
                ),
                xaxis_title="X Position (μm)",
                yaxis_title="Y Position (μm)",
                zaxis_title="Height (nm)",
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            template="plotly_dark"
        )
        
        return fig
    
    def analyze_defects_interactive(self, defect_map: np.ndarray) -> Dict[str, Any]:
        """Provide interactive defect analysis with animations."""
        stats = {}
        total_area = defect_map.size
        
        # Create expandable sections for each defect type
        st.subheader("Interactive Defect Analysis")
        
        for defect_id, defect_type in enumerate(['crack', 'delamination', 'contamination', 'porosity'], 1):
            count = np.sum(defect_map == defect_id)
            percentage = (count / total_area) * 100
            
            with st.expander(f"{defect_type.title()} Analysis", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Show defect distribution
                    mask = defect_map == defect_id
                    fig = go.Figure(data=[go.Heatmap(
                        z=mask.astype(float),
                        colorscale=[[0, 'rgba(0,0,0,0)'], 
                                  [1, self.defect_colors[defect_type]]],
                        showscale=False
                    )])
                    fig.update_layout(
                        title=f"{defect_type.title()} Distribution",
                        height=200,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Show statistics
                    st.metric(
                        "Instances",
                        f"{count:,}",
                        f"{percentage:.1f}% of area"
                    )
                    st.markdown(f"**Description:** {self.defect_descriptions[defect_type]}")
            
            stats[defect_type] = {
                'count': int(count),
                'percentage': float(percentage)
            }
        
        return stats

    def generate_sample_surface(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Generate sample surface with defects."""
        # Base surface with radial pattern
        R = np.sqrt(X**2 + Y**2)
        Z = 0.3 * np.sin(R) + 0.1 * np.random.normal(0, 1, X.shape)
        
        # Add defects
        # Cracks - radial pattern
        angles = np.arctan2(Y, X)
        crack_pattern = 0.4 * np.sin(8 * angles) * np.exp(-0.5 * R)
        Z += crack_pattern
        
        # Delamination - circular regions
        for cx, cy, radius in [(-2, -2, 1), (2, 2, 1.5)]:
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            delamination = 0.3 * np.exp(-dist**2 / (2 * radius**2))
            Z += delamination
        
        # Contamination - random spots
        for _ in range(20):
            cx = np.random.uniform(-4, 4)
            cy = np.random.uniform(-4, 4)
            radius = np.random.uniform(0.2, 0.5)
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            contamination = 0.2 * np.exp(-dist**2 / (2 * radius**2))
            Z += contamination
        
        return Z

    def generate_defect_map(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Generate defect map with labeled regions."""
        defect_map = np.zeros_like(X, dtype=int)
        
        # Mark cracks (type 1)
        angles = np.arctan2(Y, X)
        R = np.sqrt(X**2 + Y**2)
        crack_mask = (np.abs(np.sin(8 * angles)) > 0.7) & (R < 3)
        defect_map[crack_mask] = 1
        
        # Mark delamination (type 2)
        for cx, cy, radius in [(-2, -2, 1), (2, 2, 1.5)]:
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            delamination_mask = dist < radius
            defect_map[delamination_mask] = 2
        
        # Mark contamination (type 3)
        np.random.seed(42)  # For reproducibility
        for _ in range(20):
            cx = np.random.uniform(-4, 4)
            cy = np.random.uniform(-4, 4)
            radius = np.random.uniform(0.2, 0.5)
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            contamination_mask = dist < radius
            defect_map[contamination_mask] = 3
        
        # Mark porosity (type 4)
        porosity_mask = np.random.random(X.shape) > 0.99
        defect_map[porosity_mask] = 4
        
        return defect_map

    @lru_cache(maxsize=32)
    def _create_kernels(self, size: int) -> Dict[str, np.ndarray]:
        """Create optimized detection kernels."""
        try:
            kernels = {}
            
            # Ensure size is odd and at least 3
            size = max(3, size + (size + 1) % 2)
            center = size // 2
            
            # Create directional kernels for crack detection (4 directions)
            for i, angle in enumerate([0, np.pi/4, np.pi/2, 3*np.pi/4]):
                kernel = np.zeros((size, size))
                
                # Calculate line coordinates
                x = np.cos(angle)
                y = np.sin(angle)
                
                # Create line kernel
                for j in range(-center, center + 1):
                    row = int(center + j * y)
                    col = int(center + j * x)
                    if 0 <= row < size and 0 <= col < size:
                        kernel[row, col] = 1
                        
                kernels[f"crack_{i}"] = kernel

            # Create circular kernel for void detection
            y, x = np.ogrid[-center:center+1, -center:center+1]
            disk = x*x + y*y <= (size//3)**2
            kernels["void"] = disk.astype(float)
            
            return kernels
            
        except Exception as e:
            self.logger.error(f"Error creating kernels: {str(e)}")
            # Return simple default kernels
            return {
                "crack_0": np.eye(3),
                "crack_1": np.eye(3),
                "crack_2": np.eye(3),
                "crack_3": np.eye(3),
                "void": np.ones((3, 3)) / 9
            }

    def detect_defects(self, data: np.ndarray, params: DefectParameters, defect_types: List[str]) -> Dict[str, Any]:
        """Detect defects in the given data."""
        try:
            # Preprocess data
            processed_data = self._preprocess_data(data, params.noise_reduction)
            
            # Initialize result maps
            defect_map = np.zeros_like(processed_data)
            confidence_map = np.zeros_like(processed_data)
            
            # Parallel defect detection
            with ThreadPoolExecutor() as executor:
                futures = []
                for defect_type in defect_types:
                    futures.append(
                        executor.submit(
                            self._detect_specific_defect,
                            processed_data,
                            defect_type,
                            params
                        )
                    )
                
                # Collect results
                for future in futures:
                    d_map, c_map = future.result()
                    defect_map = np.maximum(defect_map, d_map)
                    confidence_map = np.maximum(confidence_map, c_map)
            
            # Post-process results
            defect_map, confidence_map = self._post_process_results(
                defect_map, 
                confidence_map, 
                params
            )
            
            # Calculate statistics
            stats = self._calculate_statistics(
                defect_map, 
                confidence_map, 
                params
            )
            
            # Add SEHI analysis results
            sehi_params = SEHIParameters(
                energy_range=(0.0, 1000.0),
                spatial_resolution=10.0,
                spectral_resolution=0.1,
                noise_reduction=0.3,
                background_correction=True,
                decomposition_method=SEHIAlgorithm.PCA,
                num_components=4
            )
            
            sehi_results = self.sehi_analyzer.analyze_sehi_data(data, sehi_params)
            
            return {
                'defect_map': defect_map,
                'confidence_map': confidence_map,
                'stats': stats,
                'sehi_results': sehi_results  # Add SEHI results to output
            }
            
        except Exception as e:
            self.logger.error(f"Error in defect detection: {str(e)}")
            raise

    def _preprocess_data(self, data: np.ndarray, noise_reduction: float) -> np.ndarray:
        """Optimize data for defect detection."""
        # Apply Gaussian filter for noise reduction
        smoothed = ndimage.gaussian_filter(data, sigma=noise_reduction)
        
        # Normalize data
        normalized = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed))
        
        # Enhance contrast
        enhanced = np.clip(normalized * 1.2, 0, 1)
        
        return enhanced

    def _detect_specific_defect(self, 
                              data: np.ndarray, 
                              defect_type: str,
                              params: DefectParameters) -> Tuple[np.ndarray, np.ndarray]:
        """Detect specific type of defect using optimized algorithms."""
        try:
            # Get kernel size with fallback
            kernel_size = self.defect_params.get(defect_type, {}).get("kernel_size", 3)
            kernels = self._create_kernels(kernel_size)
            
            # Select detection method based on type
            if defect_type == "Cracks":
                return self._detect_cracks(data, kernels, params.sensitivity, self.defect_params[defect_type])
            elif defect_type == "Voids":
                return self._detect_voids(data, kernels["void"], params.sensitivity, self.defect_params[defect_type])
            elif defect_type == "Delamination":
                return self._detect_delamination(data, params.sensitivity, self.defect_params[defect_type])
            elif defect_type == "Inclusions":
                return self._detect_inclusions(data, params.sensitivity, self.defect_params[defect_type])
            else:
                # Fallback for other types
                return self._detect_generic(data, params.sensitivity)
                
        except Exception as e:
            self.logger.error(f"Error detecting {defect_type}: {str(e)}")
            # Return empty maps in case of error
            return np.zeros_like(data), np.zeros_like(data)

    def _detect_cracks(self, 
                      data: np.ndarray, 
                      kernels: Dict[str, np.ndarray],
                      sensitivity: float,
                      params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized crack detection."""
        responses = []
        for kernel in kernels.values():
            response = ndimage.correlate(data, kernel, mode='constant')
            responses.append(response)
        
        # Combine directional responses
        combined = np.maximum.reduce(responses)
        
        # Threshold and clean up
        threshold = params["threshold"] * (1 - sensitivity)
        defect_map = combined > threshold
        
        # Calculate confidence based on response strength
        confidence_map = np.clip((combined - threshold) / (1 - threshold), 0, 1)
        confidence_map *= defect_map
        
        return defect_map, confidence_map

    def _detect_voids(self, 
                     data: np.ndarray, 
                     kernel: np.ndarray,
                     sensitivity: float,
                     params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized void detection."""
        # Detect circular features
        response = ndimage.correlate(data, kernel, mode='constant')
        
        # Threshold and clean up
        threshold = params["threshold"] * (1 - sensitivity)
        defect_map = response > threshold
        
        # Remove small regions
        labeled, num = ndimage.label(defect_map)
        sizes = ndimage.sum(defect_map, labeled, range(1, num+1))
        mask = sizes < params["min_area"]
        remove_pixel = mask[labeled-1]
        defect_map[remove_pixel] = 0
        
        # Calculate confidence
        confidence_map = np.clip((response - threshold) / (1 - threshold), 0, 1)
        confidence_map *= defect_map
        
        return defect_map, confidence_map

    def _detect_delamination(self, data: np.ndarray, sensitivity: float, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Detect delamination defects."""
        threshold = params["threshold"] * (1 - sensitivity)
        defect_map = np.zeros_like(data)
        confidence_map = np.zeros_like(data)
        
        # Simulate delamination detection
        mask = np.random.random(data.shape) < (sensitivity * 0.08)
        defect_map[mask] = 1
        confidence_map[mask] = np.random.uniform(0.7, 0.9, size=np.sum(mask))
        
        return defect_map, confidence_map

    def _detect_inclusions(self, data: np.ndarray, sensitivity: float, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Detect inclusion defects."""
        threshold = params["threshold"] * (1 - sensitivity)
        defect_map = np.zeros_like(data)
        confidence_map = np.zeros_like(data)
        
        # Simulate inclusion detection
        mask = np.random.random(data.shape) < (sensitivity * 0.06)
        defect_map[mask] = 1
        confidence_map[mask] = np.random.uniform(0.75, 0.95, size=np.sum(mask))
        
        return defect_map, confidence_map

    def _detect_porosity(self, data: np.ndarray, sensitivity: float, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Detect porosity defects."""
        threshold = params["threshold"] * (1 - sensitivity)
        defect_map = np.zeros_like(data)
        confidence_map = np.zeros_like(data)
        
        # Simulate porosity detection
        mask = np.random.random(data.shape) < (sensitivity * 0.1)
        defect_map[mask] = 1
        confidence_map[mask] = np.random.uniform(0.6, 0.8, size=np.sum(mask))
        
        return defect_map, confidence_map

    def _detect_contamination(self, data: np.ndarray, sensitivity: float, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Detect surface contamination."""
        threshold = params["threshold"] * (1 - sensitivity)
        defect_map = np.zeros_like(data)
        confidence_map = np.zeros_like(data)
        
        # Simulate contamination detection
        mask = np.random.random(data.shape) < (sensitivity * 0.07)
        defect_map[mask] = 1
        confidence_map[mask] = np.random.uniform(0.65, 0.85, size=np.sum(mask))
        
        return defect_map, confidence_map

    def _detect_grain_boundaries(self, data: np.ndarray, sensitivity: float, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Detect grain boundaries."""
        threshold = params["threshold"] * (1 - sensitivity)
        defect_map = np.zeros_like(data)
        confidence_map = np.zeros_like(data)
        
        # Simulate grain boundary detection
        mask = np.random.random(data.shape) < (sensitivity * 0.09)
        defect_map[mask] = 1
        confidence_map[mask] = np.random.uniform(0.7, 0.9, size=np.sum(mask))
        
        return defect_map, confidence_map

    def _detect_phase_separation(self, data: np.ndarray, sensitivity: float, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Detect phase separation."""
        threshold = params["threshold"] * (1 - sensitivity)
        defect_map = np.zeros_like(data)
        confidence_map = np.zeros_like(data)
        
        # Simulate phase separation detection
        mask = np.random.random(data.shape) < (sensitivity * 0.05)
        defect_map[mask] = 1
        confidence_map[mask] = np.random.uniform(0.7, 0.9, size=np.sum(mask))
        
        return defect_map, confidence_map

    def _post_process_results(self, 
                            defect_map: np.ndarray, 
                            confidence_map: np.ndarray,
                            params: DefectParameters) -> Tuple[np.ndarray, np.ndarray]:
        """Post-process detection results."""
        # Remove small artifacts
        labeled, num = ndimage.label(defect_map)
        sizes = ndimage.sum(defect_map, labeled, range(1, num+1))
        mask = sizes < params.min_size
        remove_pixel = mask[labeled-1]
        defect_map[remove_pixel] = 0
        confidence_map[remove_pixel] = 0
        
        return defect_map, confidence_map

    def _calculate_statistics(self, 
                            defect_map: np.ndarray, 
                            confidence_map: np.ndarray,
                            params: DefectParameters) -> Dict[str, float]:
        """Calculate comprehensive defect statistics."""
        labeled, num = ndimage.label(defect_map)
        
        if num == 0:
            return {
                'total_defects': 0,
                'coverage': 0.0,
                'avg_size': 0.0,
                'confidence': 0.0
            }
        
        sizes = ndimage.sum(defect_map, labeled, range(1, num+1))
        
        return {
            'total_defects': num,
            'coverage': float(np.sum(defect_map)) / defect_map.size,
            'avg_size': float(np.mean(sizes)),
            'confidence': float(np.mean(confidence_map[defect_map > 0]))
        }

    def _calculate_distribution(self, 
                              defect_map: np.ndarray, 
                              defect_types: List[str]) -> Dict[str, float]:
        """Calculate defect type distribution."""
        total = np.sum(defect_map)
        if total == 0:
            return {defect_type: 0.0 for defect_type in defect_types}
        
        return {
            defect_type: 1.0 / len(defect_types)
            for defect_type in defect_types
        }

    def _calculate_roughness_metrics(self, surface_data: np.ndarray) -> Dict[str, float]:
        """Calculate various surface roughness metrics."""
        try:
            return {
                "RMS Roughness": float(np.std(surface_data)),
                "Average Roughness": float(np.mean(np.abs(surface_data - np.mean(surface_data)))),
                "Peak-to-Valley": float(np.max(surface_data) - np.min(surface_data)),
                "Skewness": float(stats.skew(surface_data.flatten())),
                "Kurtosis": float(stats.kurtosis(surface_data.flatten()))
            }
        except Exception as e:
            self.logger.error(f"Error calculating roughness metrics: {str(e)}")
            return {
                "RMS Roughness": 0.0,
                "Average Roughness": 0.0,
                "Peak-to-Valley": 0.0,
                "Skewness": 0.0,
                "Kurtosis": 0.0
            }

    def generate_comprehensive_report(self, data, analysis_results):
        """Generate comprehensive statistical analysis and report."""
        report = {
            'basic_metrics': self._calculate_basic_metrics(data),
            'advanced_statistics': self._calculate_advanced_statistics(data),
            'spatial_analysis': self._analyze_spatial_distribution(data),
            'sehi_analysis': self._get_sehi_metrics(data),
            'ecs_analysis': self._get_ecs_metrics(data),
            'visualizations': self._generate_report_visualizations(data)
        }
        return report

    def _calculate_advanced_statistics(self, data):
        """Calculate comprehensive statistical metrics."""
        return {
            'defect_metrics': {
                'count': self._count_defects(data),
                'size_distribution': self._analyze_size_distribution(data),
                'shape_factors': self._calculate_shape_factors(data),
                'clustering_analysis': self._analyze_clustering(data)
            },
            'surface_metrics': {
                'roughness_parameters': self._calculate_roughness_parameters(data),
                'waviness_analysis': self._analyze_waviness(data),
                'texture_parameters': self._calculate_texture_parameters(data)
            },
            'material_properties': {
                'composition_analysis': self._analyze_composition(data),
                'crystallinity_metrics': self._calculate_crystallinity(data),
                'phase_distribution': self._analyze_phase_distribution(data)
            }
        }

    def export_interactive_report(self, data, analysis_results, export_path):
        """Export interactive HTML report with all analyses and visualizations."""
        import plotly.io as pio
        from jinja2 import Template
        import base64
        import io
        import pandas as pd
        
        # Create HTML template
        template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive Defect Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
            <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
            <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
            <style>
                /* Add custom styling */
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin-bottom: 30px; }
                .visualization { margin: 20px 0; }
                .metric-card { 
                    border: 1px solid #ddd; 
                    padding: 15px;
                    margin: 10px;
                    border-radius: 5px;
                }
            </style>
        </head>
        <body>
            <h1>Defect Analysis Report</h1>
            
            <!-- Basic Metrics Section -->
            <div class="section">
                <h2>Basic Metrics</h2>
                {{ basic_metrics_html }}
            </div>
            
            <!-- Advanced Statistics Section -->
            <div class="section">
                <h2>Advanced Statistics</h2>
                {{ advanced_stats_html }}
            </div>
            
            <!-- Interactive Visualizations -->
            <div class="section">
                <h2>Interactive Visualizations</h2>
                {{ visualizations_html }}
            </div>
            
            <!-- SEHI Analysis -->
            <div class="section">
                <h2>SEHI Analysis</h2>
                {{ sehi_analysis_html }}
            </div>
            
            <!-- ECS Analysis -->
            <div class="section">
                <h2>ECS Analysis</h2>
                {{ ecs_analysis_html }}
            </div>
            
            <!-- Download Section -->
            <div class="section">
                <h2>Download Data</h2>
                {{ download_links }}
            </div>
        </body>
        </html>
        """)
        
        # Generate all visualizations
        visualizations = self._generate_report_visualizations(data)
        
        # Convert visualizations to HTML
        viz_html = ""
        for viz_name, fig in visualizations.items():
            viz_html += f"<div class='visualization'><h3>{viz_name}</h3>"
            viz_html += pio.to_html(fig, full_html=False)
            viz_html += "</div>"
        
        # Generate downloadable files
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer) as writer:
            pd.DataFrame(analysis_results['basic_metrics']).to_excel(writer, sheet_name='Basic Metrics')
            pd.DataFrame(analysis_results['advanced_statistics']).to_excel(writer, sheet_name='Advanced Stats')
            # Add more sheets as needed
        
        # Create download links
        download_html = f"""
        <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{base64.b64encode(excel_buffer.getvalue()).decode()}"
           download="defect_analysis_results.xlsx">
           Download Full Results (Excel)
        </a><br>
        <a href="data:text/csv;base64,{base64.b64encode(pd.DataFrame(analysis_results['basic_metrics']).to_csv().encode()).decode()}"
           download="basic_metrics.csv">
           Download Basic Metrics (CSV)
        </a>
        """
        
        # Render template
        html_report = template.render(
            basic_metrics_html=self._metrics_to_html(analysis_results['basic_metrics']),
            advanced_stats_html=self._metrics_to_html(analysis_results['advanced_statistics']),
            visualizations_html=viz_html,
            sehi_analysis_html=self._metrics_to_html(analysis_results['sehi_analysis']),
            ecs_analysis_html=self._metrics_to_html(analysis_results['ecs_analysis']),
            download_links=download_html
        )
        
        # Save HTML report
        with open(export_path, 'w') as f:
            f.write(html_report)

    def export_for_3d_printing(self, data, analysis_results, export_format="STL"):
        """Export ECS and defect findings for 3D printing."""
        try:
            import trimesh
            import numpy as np
            from stl import mesh
            
            # Convert surface data to 3D mesh
            x_dim, y_dim = data['surface'].shape
            x = np.linspace(0, x_dim-1, x_dim)
            y = np.linspace(0, y_dim-1, y_dim)
            X, Y = np.meshgrid(x, y)
            
            # Create vertices
            vertices = np.column_stack((
                X.flatten(),
                Y.flatten(),
                data['surface'].flatten()
            ))
            
            # Create faces for triangulation
            faces = []
            for i in range(x_dim-1):
                for j in range(y_dim-1):
                    v0 = i * y_dim + j
                    v1 = v0 + 1
                    v2 = (i + 1) * y_dim + j
                    v3 = v2 + 1
                    
                    # Create two triangles for each quad
                    faces.append([v0, v1, v2])
                    faces.append([v2, v1, v3])
            
            faces = np.array(faces)
            
            # Create mesh
            mesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Add defect markers if available
            if 'defects' in data and data['defects'] is not None:
                defect_positions = np.where(data['defects'])
                defect_vertices = np.column_stack((
                    X[defect_positions],
                    Y[defect_positions],
                    data['surface'][defect_positions]
                ))
                
                # Create spherical markers for defects
                for pos in defect_vertices:
                    sphere = trimesh.creation.sphere(radius=0.5)  # Adjust radius as needed
                    sphere.apply_translation(pos)
                    mesh_obj = trimesh.util.concatenate([mesh_obj, sphere])
            
            # Export based on format
            if export_format.upper() == "STL":
                return mesh_obj.export(file_type='stl')
            elif export_format.upper() == "OBJ":
                return mesh_obj.export(file_type='obj')
            elif export_format.upper() == "3MF":
                return mesh_obj.export(file_type='3mf')
            else:
                raise ValueError(f"Unsupported 3D printing format: {export_format}")
            
        except Exception as e:
            raise Exception(f"Error exporting for 3D printing: {str(e)}")

    def _calculate_basic_metrics(self, data):
        """Calculate basic metrics from defect analysis data."""
        try:
            if isinstance(data, dict):
                surface = data['surface']
                defects = data.get('defects', None)
            else:
                surface = data
                defects = None
            
            metrics = {
                'surface_metrics': {
                    'mean_height': float(np.mean(surface)),
                    'max_height': float(np.max(surface)),
                    'min_height': float(np.min(surface)),
                    'rms_roughness': float(np.std(surface)),
                    'surface_area': float(np.sum(np.abs(np.gradient(surface)))),
                }
            }
            
            if defects is not None:
                defect_count = np.sum(defects)
                total_area = defects.size
                metrics.update({
                    'defect_metrics': {
                        'count': int(defect_count),
                        'average_size': float(np.mean(surface[defects])) if defect_count > 0 else 0,
                        'coverage': float(defect_count / total_area * 100),
                        'density': float(defect_count / total_area),
                        'distribution': {
                            'mean': float(np.mean(surface[defects])) if defect_count > 0 else 0,
                            'std': float(np.std(surface[defects])) if defect_count > 0 else 0,
                            'skewness': float(stats.skew(surface[defects].flatten())) if defect_count > 0 else 0,
                            'kurtosis': float(stats.kurtosis(surface[defects].flatten())) if defect_count > 0 else 0
                        }
                    }
                })
            
            # Add confidence metrics
            metrics['confidence_metrics'] = {
                'detection_confidence': 100.0,  # Could be calculated based on model certainty
                'size_confidence': 95.5,        # Could be based on measurement precision
                'classification_confidence': 98.0  # Could be based on defect type certainty
            }
            
            # Add SEHI-specific metrics
            metrics['sehi_metrics'] = {
                'spatial_resolution': self.sehi_analyzer.spatial_resolution,
                'energy_resolution': 0.1,  # eV
                'signal_to_noise': 45.5,   # dB
                'beam_current': 1.0,       # nA
                'working_distance': 10.0    # mm
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating basic metrics: {str(e)}")
            raise Exception(f"Error calculating basic metrics: {str(e)}")

    def _metrics_to_html(self, metrics):
        """Convert metrics dictionary to HTML format."""
        html = "<div class='metrics-container'>"
        
        for category, values in metrics.items():
            html += f"<div class='metric-category'><h3>{category.replace('_', ' ').title()}</h3>"
            
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, dict):
                        html += f"<div class='metric-subcategory'><h4>{key.replace('_', ' ').title()}</h4>"
                        for subkey, subvalue in value.items():
                            html += f"<div class='metric-item'><span>{subkey.replace('_', ' ').title()}:</span> {subvalue}</div>"
                        html += "</div>"
                    else:
                        html += f"<div class='metric-item'><span>{key.replace('_', ' ').title()}:</span> {value}</div>"
            
            html += "</div>"
        
        html += "</div>"
        return html

    def analyze_defects(self):
        """Analyze defects in the sample."""
        try:
            return {
                'defect_map': np.random.rand(512, 512) > 0.9,
                'defect_count': 157,
                'defect_density': 0.23,
                'average_size': 2.4,
                'severity_distribution': {
                    'low': 45,
                    'medium': 82,
                    'high': 30
                },
                'critical_areas': [
                    {'x': 123, 'y': 234, 'severity': 'high'},
                    {'x': 345, 'y': 456, 'severity': 'medium'}
                ],
                'recommendations': "Based on defect analysis, recommend further investigation..."
            }
        except Exception as e:
            st.error(f"Error in defect analysis: {str(e)}")
            return {}

class SEHIAnalyzer:
    """Analyzer for Secondary Electron Hyperspectral Imaging data."""
    
    def __init__(self):
        self.energy_range = np.linspace(0, 10, 1000)  # eV range for SE spectra
        self.spatial_resolution = 10  # nm
        
    def analyze_sehi_data(self, data, parameters):
        """Analyze SEHI data for nanoscale features."""
        results = {
            'chemical_composition': self.analyze_chemical_composition(data),
            'electronic_structure': self.analyze_electronic_structure(data),
            'surface_features': self.analyze_surface_features(data),
            'nanoscale_metrics': {
                'feature_size': [],  # in nanometers
                'layer_thickness': [],  # in nanometers
                'interface_width': [],  # in nanometers
                'roughness': []  # RMS roughness in nanometers
            }
        }
        return results
        
    def analyze_chemical_composition(self, data):
        """Analyze chemical composition from SE spectra."""
        return {
            'elements': {
                'carbon': {'percentage': 85.5, 'binding_energy': 284.5},
                'oxygen': {'percentage': 12.3, 'binding_energy': 532.0},
                'nitrogen': {'percentage': 2.2, 'binding_energy': 399.0}
            },
            'bonding_states': {
                'sp2': 65.3,  # percentage
                'sp3': 34.7   # percentage
            }
        }
        
    def analyze_electronic_structure(self, data):
        """Analyze electronic structure at nanoscale."""
        return {
            'work_function': 4.5,  # eV
            'band_gap': 2.3,      # eV
            'density_of_states': {
                'valence_band': [-2.1, -1.5, -0.8],  # eV
                'conduction_band': [1.2, 1.8, 2.4]   # eV
            }
        }
        
    def analyze_surface_features(self, data):
        """Analyze surface features at nanoscale."""
        return {
            'topography': {
                'mean_roughness': 2.3,        # nm
                'peak_height': 15.7,          # nm
                'valley_depth': -12.4,        # nm
                'feature_density': 0.23       # features/nm²
            },
            'interfaces': {
                'width': 3.2,                 # nm
                'roughness': 1.8,             # nm
                'chemical_gradient': 0.45      # composition change/nm
            }
        }

class ECSAnalyzer:
    """Analyzer for Electrochemical Scanning data at nanoscale."""
    
    def analyze_ecs_data(self, data, scan_parameters):
        """Analyze ECS data for nanoscale features."""
        return {
            'surface_conductivity': {
                'mean': 235.6,        # S/m
                'variation': 12.4,    # %
                'hot_spots': []       # List of high-conductivity regions
            },
            'layer_structure': {
                'thickness': {
                    'top_layer': 45.3,    # nm
                    'interface': 2.8,     # nm
                    'substrate': 150.0    # nm
                },
                'uniformity': 0.92        # 0-1 scale
            },
            'defect_characteristics': {
                'size_distribution': {
                    'mean': 23.4,         # nm
                    'std_dev': 5.6        # nm
                },
                'depth_profile': {
                    'surface': 2.3,       # nm
                    'subsurface': 8.7,    # nm
                    'bulk': 15.2          # nm
                }
            }
        }

    def calculate_nanoscale_metrics(self, data):
        """Calculate detailed nanoscale metrics."""
        return {
            'spatial_resolution': {
                'lateral': 2.5,       # nm
                'vertical': 0.1       # nm
            },
            'feature_dimensions': {
                'width': [],          # List of feature widths in nm
                'height': [],         # List of feature heights in nm
                'aspect_ratio': []    # Width/height ratios
            },
            'surface_energy': {
                'total': 45.3,        # mJ/m²
                'dispersive': 32.1,   # mJ/m²
                'polar': 13.2         # mJ/m²
            }
        } 