import numpy as np
from scipy import stats

class SEHIAnalyzer:
    def calculate_basic_stats(self, data):
        """Compute fundamental statistical metrics"""
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'skewness': stats.skew(data.flatten()),
            'kurtosis': stats.kurtosis(data.flatten())
        }

    def detect_phase_boundaries(self, data, threshold=0.5):
        """Identify material phase boundaries using gradient analysis"""
        grad_x, grad_y = np.gradient(data)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return magnitude > threshold * np.max(magnitude)
