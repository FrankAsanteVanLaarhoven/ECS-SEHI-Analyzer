from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from enum import Enum

class InterferenceType(Enum):
    CONSTRUCTIVE = "constructive"
    DESTRUCTIVE = "destructive"
    PARTIAL = "partial"

@dataclass
class InterferenceConfig:
    """Interference calculation configuration"""
    wavelength: float = 633e-9  # meters
    resolution: Tuple[int, int] = (1024, 1024)
    pixel_size: float = 8e-6    # meters
    propagation_distance: float = 0.1  # meters
    metadata: Dict = field(default_factory=dict)

class InterferenceCalculator:
    def __init__(self, config: Optional[InterferenceConfig] = None):
        self.config = config or InterferenceConfig()
        
    def calculate_interference(self,
                             wave1: np.ndarray,
                             wave2: np.ndarray,
                             phase_difference: float = 0.0) -> Dict:
        """Calculate interference pattern"""
        try:
            if wave1.shape != wave2.shape:
                raise ValueError("Wave shapes must match")
                
            # Calculate complex amplitudes
            k = 2 * np.pi / self.config.wavelength
            phase1 = np.angle(wave1)
            phase2 = np.angle(wave2) + phase_difference
            
            # Combine waves
            combined = wave1 + wave2 * np.exp(1j * phase_difference)
            intensity = np.abs(combined)**2
            
            # Analyze interference type
            max_intensity = np.max(intensity)
            min_intensity = np.min(intensity)
            contrast = (max_intensity - min_intensity) / (max_intensity + min_intensity)
            
            if contrast > 0.8:
                interference_type = InterferenceType.CONSTRUCTIVE
            elif contrast < 0.2:
                interference_type = InterferenceType.DESTRUCTIVE
            else:
                interference_type = InterferenceType.PARTIAL
                
            return {
                "success": True,
                "intensity": intensity,
                "contrast": contrast,
                "interference_type": interference_type.value,
                "phase_difference": phase_difference
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def propagate_wave(self,
                      initial_wave: np.ndarray,
                      distance: Optional[float] = None) -> Dict:
        """Propagate wave through space"""
        try:
            distance = distance or self.config.propagation_distance
            
            # Calculate propagation phase
            k = 2 * np.pi / self.config.wavelength
            x = np.linspace(-self.config.resolution[0]/2, 
                           self.config.resolution[0]/2,
                           self.config.resolution[0]) * self.config.pixel_size
            y = np.linspace(-self.config.resolution[1]/2,
                           self.config.resolution[1]/2,
                           self.config.resolution[1]) * self.config.pixel_size
            X, Y = np.meshgrid(x, y)
            R = np.sqrt(X**2 + Y**2 + distance**2)
            
            # Apply propagation
            propagated = initial_wave * np.exp(1j * k * R)
            
            return {
                "success": True,
                "wave": propagated,
                "distance": distance
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def calculate_coherence(self,
                          wave1: np.ndarray,
                          wave2: np.ndarray) -> Dict:
        """Calculate coherence between two waves"""
        try:
            # Calculate correlation
            correlation = np.sum(wave1 * np.conj(wave2)) / np.sqrt(
                np.sum(np.abs(wave1)**2) * np.sum(np.abs(wave2)**2)
            )
            
            coherence = np.abs(correlation)
            phase = np.angle(correlation)
            
            return {
                "success": True,
                "coherence": coherence,
                "phase": phase
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            } 