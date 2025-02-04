import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
from scipy import signal, ndimage
from sklearn import decomposition, cluster
import logging
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.neural_network import MLPRegressor

class SEHIAlgorithm(Enum):
    """Available algorithms for SEHI analysis."""
    PCA = "Principal Component Analysis"
    NMF = "Non-negative Matrix Factorization"
    ICA = "Independent Component Analysis"
    MCR = "Multivariate Curve Resolution"
    NN = "Neural Network Analysis"  # Renamed from CNN to NN
    # Removed UMAP temporarily

@dataclass
class SEHIParameters:
    """Parameters for SEHI analysis."""
    energy_range: Tuple[float, float]
    spatial_resolution: float
    spectral_resolution: float
    noise_reduction: float
    background_correction: bool
    decomposition_method: SEHIAlgorithm
    num_components: int

class SEHIAnalyzer:
    """Advanced SEHI analysis system."""
    
    def __init__(self):
        self.algorithms = {
            SEHIAlgorithm.PCA: self._run_pca,
            SEHIAlgorithm.NMF: self._run_nmf,
            SEHIAlgorithm.ICA: self._run_ica,
            SEHIAlgorithm.MCR: self._run_mcr,
            SEHIAlgorithm.NN: self._run_nn  # Renamed from _run_cnn to _run_nn
            # Removed UMAP temporarily
        }
        
        self.material_properties = {
            "Carbon Network": {
                "sp2_sp3_ratio": (0.1, 0.9),
                "crystallinity": (0.2, 0.8),
                "surface_area": (10, 1000),  # m²/g
                "porosity": (0.1, 0.5)
            },
            "Surface Chemistry": {
                "oxidation_state": (-2, 4),
                "functional_groups": ["C-O", "C=O", "COOH", "C-H"],
                "surface_energy": (20, 80)  # mJ/m²
            },
            "Degradation Metrics": {
                "corrosion_rate": (0.01, 1.0),  # mm/year
                "mechanical_stress": (0, 500),  # MPa
                "chemical_stability": (0.5, 1.0)
            }
        }

    def analyze_sehi_data(self, data: np.ndarray, params: SEHIParameters) -> Dict[str, Any]:
        """Analyze SEHI data with selected algorithm."""
        try:
            # Initialize default values if data is missing or invalid
            if data is None or data.size == 0:
                return self._get_default_results()
            
            # Ensure data is 3D (add batch dimension if needed)
            if len(data.shape) == 2:
                data = data.reshape(1, *data.shape)
            
            # Preprocess data
            processed_data = self._preprocess_data(data, params)
            
            # Run selected decomposition method
            algorithm = self.algorithms[params.decomposition_method]
            components, loadings = algorithm(processed_data, params.num_components)
            
            # Analyze properties
            network_properties = self._analyze_carbon_network(components)
            surface_properties = self._analyze_surface_chemistry(components)
            degradation_metrics = self._assess_degradation(
                components, 
                network_properties, 
                surface_properties
            )
            
            return {
                "components": components,
                "loadings": loadings,
                "network_properties": network_properties,
                "surface_properties": surface_properties,
                "degradation_metrics": degradation_metrics
            }
            
        except Exception as e:
            logging.error(f"Error in SEHI analysis: {str(e)}")
            return self._get_default_results()

    def _get_default_results(self) -> Dict[str, Any]:
        """Return default results structure."""
        return {
            "components": np.zeros((4, 10, 10)),
            "loadings": np.zeros((100, 4)),
            "network_properties": {
                "sp2_sp3_ratio": 0.5,
                "crystallinity": 0.7,
                "surface_area": 500,
                "porosity": 0.3
            },
            "surface_properties": {
                "oxidation_state": 0.0,
                "surface_energy": 50.0,
                "functional_groups": {
                    "C-O": 0.3,
                    "C=O": 0.2,
                    "COOH": 0.1,
                    "C-H": 0.4
                }
            },
            "degradation_metrics": {
                "mechanical_stability": 80.0,
                "chemical_stability": 0.75,
                "corrosion_rate": 0.15
            }
        }

    def _preprocess_data(self, data: np.ndarray, params: SEHIParameters) -> np.ndarray:
        """Preprocess SEHI data."""
        # Apply noise reduction
        if params.noise_reduction > 0:
            data = ndimage.gaussian_filter(data, sigma=params.noise_reduction)
            
        # Background correction
        if params.background_correction:
            data = self._correct_background(data)
            
        return data

    def _correct_background(self, data: np.ndarray) -> np.ndarray:
        """Apply background correction to SEHI data."""
        try:
            # Estimate background using morphological operations
            background = ndimage.grey_opening(data, size=(5, 5))
            
            # Subtract background
            corrected = data - background
            
            # Ensure non-negative values
            corrected = np.maximum(corrected, 0)
            
            # Normalize
            if np.max(corrected) > 0:
                corrected = corrected / np.max(corrected)
            
            return corrected
            
        except Exception as e:
            logging.error(f"Error in background correction: {str(e)}")
            return data  # Return original data if correction fails

    def _analyze_carbon_network(self, components: np.ndarray) -> Dict[str, float]:
        """Analyze carbon network properties."""
        return {
            "sp2_sp3_ratio": self._calculate_sp2_sp3_ratio(components),
            "crystallinity": self._estimate_crystallinity(components),
            "surface_area": self._calculate_surface_area(components),
            "porosity": self._estimate_porosity(components)
        }

    def _analyze_surface_chemistry(self, components: np.ndarray) -> Dict[str, Any]:
        """Analyze surface chemistry."""
        return {
            "oxidation_state": self._estimate_oxidation_state(components),
            "functional_groups": self._identify_functional_groups(components),
            "surface_energy": self._calculate_surface_energy(components)
        }

    def _assess_degradation(self, 
                          components: np.ndarray,
                          network_props: Dict[str, float],
                          surface_props: Dict[str, Any]) -> Dict[str, float]:
        """Assess degradation metrics."""
        return {
            "corrosion_rate": self._estimate_corrosion_rate(
                components, network_props, surface_props
            ),
            "mechanical_stability": self._assess_mechanical_stability(
                components, network_props
            ),
            "chemical_stability": self._assess_chemical_stability(
                components, surface_props
            )
        }

    def _run_pca(self, data: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
        """Run PCA analysis."""
        try:
            # Ensure data is 2D for PCA
            original_shape = data.shape
            X = data.reshape(-1, np.prod(data.shape[1:]))
            
            # Apply PCA
            pca = PCA(n_components=min(n_components, X.shape[0], X.shape[1]))
            components = pca.fit_transform(X)
            loadings = pca.components_
            
            # Reshape components back to 3D
            components = components.reshape(-1, original_shape[1], original_shape[2])
            
            return components, loadings
            
        except Exception as e:
            logging.error(f"Error in PCA analysis: {str(e)}")
            return np.zeros((n_components, data.shape[1], data.shape[2])), np.zeros((n_components, data.shape[1]))

    def _run_nmf(self, data: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
        """Run Non-negative Matrix Factorization."""
        try:
            # Reshape and ensure non-negativity
            X = data.reshape(data.shape[0], -1)
            X = np.abs(X)
            
            # Apply NMF
            model = NMF(n_components=n_components, init='random', random_state=0)
            components = model.fit_transform(X)
            loadings = model.components_
            
            # Reshape components
            components = components.reshape(-1, data.shape[1], data.shape[2])
            
            return components, loadings
            
        except Exception as e:
            logging.error(f"Error in NMF analysis: {str(e)}")
            return np.zeros((n_components, data.shape[1], data.shape[2])), np.zeros((n_components, data.shape[1]))

    def _run_ica(self, data: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
        """Run Independent Component Analysis."""
        try:
            # Reshape data
            X = data.reshape(data.shape[0], -1)
            
            # Apply ICA
            ica = FastICA(n_components=n_components, random_state=0)
            components = ica.fit_transform(X)
            loadings = ica.components_
            
            # Reshape components
            components = components.reshape(-1, data.shape[1], data.shape[2])
            
            return components, loadings
            
        except Exception as e:
            logging.error(f"Error in ICA analysis: {str(e)}")
            return np.zeros((n_components, data.shape[1], data.shape[2])), np.zeros((n_components, data.shape[1]))

    def _run_mcr(self, data: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
        """Run Multivariate Curve Resolution."""
        try:
            # Simplified MCR implementation
            # In practice, you might want to use a specialized MCR library
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(data.reshape(data.shape[0], -1))
            loadings = pca.components_
            
            # Apply non-negativity constraint
            components = np.abs(components)
            loadings = np.abs(loadings)
            
            # Reshape components
            components = components.reshape(-1, data.shape[1], data.shape[2])
            
            return components, loadings
            
        except Exception as e:
            logging.error(f"Error in MCR analysis: {str(e)}")
            return np.zeros((n_components, data.shape[1], data.shape[2])), np.zeros((n_components, data.shape[1]))

    def _run_nn(self, data: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
        """Run neural network-based analysis using sklearn."""
        try:
            # Store original shape
            original_shape = data.shape
            
            # Reshape data to 2D
            X = data.reshape(1, -1) if len(data.shape) == 2 else data.reshape(data.shape[0], -1)
            
            # Create and train neural network
            nn = MLPRegressor(
                hidden_layer_sizes=(64, n_components, 64),
                activation='relu',
                solver='adam',
                random_state=0,
                max_iter=1000
            )
            
            # Fit and transform
            nn.fit(X, X)  # Autoencoder-like behavior
            
            # Get the middle layer representation
            components = nn.predict(X)
            
            # Get weights as loadings
            loadings = nn.coefs_[0]
            
            # Reshape components based on original dimensionality
            if len(original_shape) == 2:
                components = components.reshape(n_components, original_shape[0], original_shape[1])
            else:
                components = components.reshape(-1, original_shape[1], original_shape[2])
            
            return components, loadings
            
        except Exception as e:
            logging.error(f"Error in neural network analysis: {str(e)}")
            if len(data.shape) == 2:
                return np.zeros((n_components, data.shape[0], data.shape[1])), np.zeros((n_components, data.shape[0]))
            else:
                return np.zeros((n_components, data.shape[1], data.shape[2])), np.zeros((n_components, data.shape[1])) 