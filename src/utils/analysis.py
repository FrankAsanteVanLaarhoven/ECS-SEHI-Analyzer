import streamlit as st
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

# Try importing optional dependencies
try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    st.warning("scikit-learn not installed. Advanced analysis will be limited.")

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    st.warning("hdbscan not installed. Advanced clustering will be limited.")

class SEHIAnalyzer:
    """Basic analysis tools for SEHI data with graceful fallbacks."""
    
    def __init__(self):
        self.available_methods = ['basic']
        
        if HAS_SKLEARN:
            self.available_methods.extend(['pca', 'kmeans', 'dbscan'])
        if HAS_HDBSCAN:
            self.available_methods.append('hdbscan')

    def analyze_data(self, data: np.ndarray, method: str = 'basic') -> Optional[Dict[str, Any]]:
        """Analyze data using available methods."""
        try:
            if data is None:
                return None

            if method not in self.available_methods:
                st.warning(f"Method {method} not available. Using basic analysis.")
                method = 'basic'

            if method == 'basic':
                return self._basic_analysis(data)
            elif method == 'pca' and HAS_SKLEARN:
                return self._pca_analysis(data)
            elif method == 'kmeans' and HAS_SKLEARN:
                return self._kmeans_analysis(data)
            elif method == 'dbscan' and HAS_SKLEARN:
                return self._dbscan_analysis(data)
            elif method == 'hdbscan' and HAS_HDBSCAN:
                return self._hdbscan_analysis(data)

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return None

    def _basic_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform basic statistical analysis."""
        try:
            return {
                'mean': np.mean(data, axis=0),
                'std': np.std(data, axis=0),
                'min': np.min(data, axis=0),
                'max': np.max(data, axis=0),
                'shape': data.shape
            }
        except Exception as e:
            st.error(f"Basic analysis failed: {str(e)}")
            return {}

    def _pca_analysis(self, data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Perform PCA if sklearn is available."""
        if not HAS_SKLEARN:
            return None
            
        try:
            pca = PCA(n_components=min(2, data.shape[1]))
            transformed = pca.fit_transform(data)
            return {
                'transformed': transformed,
                'explained_variance': pca.explained_variance_ratio_,
                'n_components': pca.n_components_
            }
        except Exception as e:
            st.error(f"PCA analysis failed: {str(e)}")
            return None

    def _kmeans_analysis(self, data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Perform K-means clustering if sklearn is available."""
        if not HAS_SKLEARN:
            return None
            
        try:
            kmeans = KMeans(n_clusters=min(3, data.shape[0]))
            labels = kmeans.fit_predict(data)
            return {
                'labels': labels,
                'centers': kmeans.cluster_centers_,
                'n_clusters': kmeans.n_clusters
            }
        except Exception as e:
            st.error(f"K-means analysis failed: {str(e)}")
            return None

    def _dbscan_analysis(self, data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Perform DBSCAN clustering if sklearn is available."""
        if not HAS_SKLEARN:
            return None
            
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            labels = dbscan.fit_predict(data)
            return {
                'labels': labels,
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'noise': np.sum(labels == -1)
            }
        except Exception as e:
            st.error(f"DBSCAN analysis failed: {str(e)}")
            return None

    def _hdbscan_analysis(self, data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Perform HDBSCAN clustering if available."""
        if not HAS_HDBSCAN:
            return None
            
        try:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
            labels = clusterer.fit_predict(data)
            return {
                'labels': labels,
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'noise': np.sum(labels == -1),
                'probabilities': clusterer.probabilities_
            }
        except Exception as e:
            st.error(f"HDBSCAN analysis failed: {str(e)}")
            return None 