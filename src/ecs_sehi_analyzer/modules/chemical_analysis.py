import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from scipy import ndimage
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

@dataclass
class ChemicalAnalysisConfig:
    resolution: int = 512
    noise_reduction: float = 0.5
    peak_threshold: float = 0.1
    cluster_count: int = 3
    confidence_level: float = 0.95

class ChemicalAnalyzer:
    """Chemical composition analysis for SEHI data"""
    
    def __init__(self, config: Optional[ChemicalAnalysisConfig] = None):
        self.config = config or ChemicalAnalysisConfig()
        self._initialize_analysis_tools()
    
    def _initialize_analysis_tools(self):
        """Initialize analysis components"""
        self.kmeans = KMeans(
            n_clusters=self.config.cluster_count,
            random_state=42
        )
    
    def analyze(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Perform chemical analysis on input data
        
        Args:
            data: 2D numpy array of chemical composition data
            
        Returns:
            Dictionary containing analysis results
        """
        # Preprocess data
        processed_data = self._preprocess_data(data)
        
        # Perform analysis
        basic_stats = self._calculate_basic_statistics(processed_data)
        composition = self._analyze_composition(processed_data)
        peaks = self._find_chemical_peaks(processed_data)
        clusters = self._identify_chemical_clusters(processed_data)
        
        return {
            "basic_statistics": basic_stats,
            "composition_analysis": composition,
            "chemical_peaks": peaks,
            "chemical_clusters": clusters,
            "visualization_data": self._prepare_visualization_data(processed_data)
        }
    
    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """Preprocess raw data"""
        # Apply noise reduction
        if self.config.noise_reduction > 0:
            data = ndimage.gaussian_filter(
                data,
                sigma=self.config.noise_reduction
            )
        
        # Normalize data
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        return data
    
    def _calculate_basic_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate basic statistical measures"""
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
            "skewness": float(self._calculate_skewness(data)),
            "kurtosis": float(self._calculate_kurtosis(data))
        }
    
    def _analyze_composition(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze chemical composition distribution"""
        hist, bins = np.histogram(data, bins=50, density=True)
        
        return {
            "histogram": hist.tolist(),
            "bin_edges": bins.tolist(),
            "distribution_type": self._determine_distribution_type(data),
            "uniformity_index": float(self._calculate_uniformity(data))
        }
    
    def _find_chemical_peaks(self, data: np.ndarray) -> Dict[str, List[float]]:
        """Identify chemical composition peaks"""
        hist, bins = np.histogram(data, bins=50)
        peak_indices, _ = find_peaks(hist, height=self.config.peak_threshold * np.max(hist))
        
        return {
            "peak_positions": bins[peak_indices].tolist(),
            "peak_intensities": hist[peak_indices].tolist(),
            "peak_count": len(peak_indices)
        }
    
    def _identify_chemical_clusters(self, data: np.ndarray) -> Dict[str, Any]:
        """Identify distinct chemical regions using clustering"""
        # Reshape data for clustering
        X = data.reshape(-1, 1)
        
        # Perform clustering
        labels = self.kmeans.fit_predict(X)
        centers = self.kmeans.cluster_centers_
        
        return {
            "cluster_centers": centers.flatten().tolist(),
            "cluster_populations": [int(sum(labels == i)) for i in range(self.config.cluster_count)],
            "cluster_map": labels.reshape(data.shape).tolist()
        }
    
    def _prepare_visualization_data(self, data: np.ndarray) -> Dict[str, Any]:
        """Prepare data for visualization"""
        return {
            "chemical_map": data.tolist(),
            "gradient_magnitude": self._calculate_gradients(data).tolist(),
            "cluster_boundaries": self._find_cluster_boundaries(data).tolist()
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate distribution skewness"""
        return float(np.mean(((data - np.mean(data)) / np.std(data)) ** 3))
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate distribution kurtosis"""
        return float(np.mean(((data - np.mean(data)) / np.std(data)) ** 4) - 3)
    
    def _determine_distribution_type(self, data: np.ndarray) -> str:
        """Determine the type of chemical distribution"""
        skewness = self._calculate_skewness(data)
        kurtosis = self._calculate_kurtosis(data)
        
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return "Normal"
        elif skewness > 1:
            return "Right-skewed"
        elif skewness < -1:
            return "Left-skewed"
        else:
            return "Mixed"
    
    def _calculate_uniformity(self, data: np.ndarray) -> float:
        """Calculate chemical uniformity index"""
        return float(1.0 - np.std(data) / np.mean(data))
    
    def _calculate_gradients(self, data: np.ndarray) -> np.ndarray:
        """Calculate chemical composition gradients"""
        grad_y, grad_x = np.gradient(data)
        return np.sqrt(grad_x**2 + grad_y**2)
    
    def _find_cluster_boundaries(self, data: np.ndarray) -> np.ndarray:
        """Find boundaries between chemical clusters"""
        labels = self._identify_chemical_clusters(data)["cluster_map"]
        return ndimage.generic_gradient_magnitude(labels, ndimage.sobel)
    
    def generate_visualization(self, data: np.ndarray, plot_type: str = "2D") -> go.Figure:
        """
        Generate interactive visualization of chemical analysis
        
        Args:
            data: Chemical composition data
            plot_type: Type of plot ("2D" or "3D")
            
        Returns:
            Plotly figure object
        """
        if plot_type == "2D":
            fig = go.Figure(data=go.Heatmap(
                z=data,
                colorscale='Viridis',
                colorbar=dict(title="Chemical Composition")
            ))
        else:
            x = np.arange(data.shape[0])
            y = np.arange(data.shape[1])
            fig = go.Figure(data=[go.Surface(
                x=x, y=y, z=data,
                colorscale='Viridis',
                colorbar=dict(title="Chemical Composition")
            )])
            
        fig.update_layout(
            title="Chemical Composition Analysis",
            template="plotly_dark",
            height=600
        )
        
        return fig

    def create_3d_surface(self, data: np.ndarray, params: Dict) -> go.Figure:
        """Create interactive 3D surface plot with advanced controls"""
        fig = go.Figure(data=[
            go.Surface(
                z=data,
                colorscale='Viridis',
                contours={
                    "z": {"show": True, "usecolormap": True}
                },
                lighting={
                    "ambient": 0.4,
                    "diffuse": 0.6,
                    "specular": 0.2
                }
            )
        ])
        
        fig.update_layout(
            scene=dict(
                xaxis_title="X (μm)",
                yaxis_title="Y (μm)",
                zaxis_title="Intensity (a.u.)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.5)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            coloraxis_colorbar=dict(
                title="Intensity",
                thickness=20,
                len=0.75
            )
        )
        return fig

    def create_correlation_matrix(self, spectral_data: np.ndarray) -> go.Figure:
        """Generate correlation matrix visualization for spectral data"""
        corr_matrix = np.corrcoef(spectral_data.reshape(-1, spectral_data.shape[2]).T)
        
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(
                    title="Correlation",
                    thickness=20,
                    len=0.75
                )
            )
        )
        
        fig.update_layout(
            title="Spectral Feature Correlation Matrix",
            xaxis_title="Wavelength Index",
            yaxis_title="Wavelength Index",
            height=600
        )
        return fig
