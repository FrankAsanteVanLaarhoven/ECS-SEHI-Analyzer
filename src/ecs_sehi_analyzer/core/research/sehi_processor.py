import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SEHIData:
    """SEHI data structure"""
    image: np.ndarray
    metadata: Dict
    timestamp: float
    resolution: Tuple[int, int] = (512, 512)

class SEHIDegradationPredictor(nn.Module):
    def __init__(self, input_dim: int = 512*512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 512*512)
        return self.layers(x)

class SEHIProcessor:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = SEHIDegradationPredictor().to(self.device)
        self.current_data: Optional[SEHIData] = None
        
    def process_image(self, data: SEHIData) -> Dict:
        """Process SEHI image data"""
        self.current_data = data
        processed_data = self._preprocess_image(data.image)
        
        with torch.no_grad():
            tensor_data = torch.tensor(processed_data).float().to(self.device)
            prediction = self.model(tensor_data)
            
        critical_areas = self._identify_critical_areas(processed_data)
        
        return {
            "degradation_score": float(prediction.cpu().numpy()),
            "critical_areas": critical_areas,
            "resolution": data.resolution,
            "analysis_timestamp": data.timestamp
        }
    
    def analyze_temporal_changes(self, image_series: list[SEHIData]) -> Dict:
        """Analyze temporal changes in material degradation"""
        degradation_scores = []
        timestamps = []
        
        for data in image_series:
            result = self.process_image(data)
            degradation_scores.append(result["degradation_score"])
            timestamps.append(data.timestamp)
            
        degradation_rate = np.polyfit(timestamps, degradation_scores, 1)[0]
        
        return {
            "degradation_rate": degradation_rate,
            "trend": "accelerating" if degradation_rate > 0.1 else "stable",
            "scores": degradation_scores,
            "timestamps": timestamps
        }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess SEHI image data"""
        # Normalize
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        # Apply Gaussian filter for noise reduction
        from scipy.ndimage import gaussian_filter
        image = gaussian_filter(image, sigma=1.0)
        
        return image
    
    def _identify_critical_areas(self, processed_data: np.ndarray) -> np.ndarray:
        """Identify critical areas in the material"""
        # Threshold for critical degradation
        threshold = np.mean(processed_data) + 2 * np.std(processed_data)
        return np.where(processed_data > threshold, 1, 0) 