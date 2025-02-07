from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from enum import Enum
import plotly.graph_objects as go

class RenderMode(Enum):
    AMPLITUDE = "amplitude"
    PHASE = "phase"
    INTENSITY = "intensity"
    COMBINED = "combined"

@dataclass
class RenderConfig:
    """Hologram rendering configuration"""
    resolution: Tuple[int, int] = (1024, 1024)
    wavelength: float = 633e-9  # meters
    pixel_size: float = 8e-6    # meters
    render_mode: RenderMode = RenderMode.INTENSITY
    metadata: Dict = field(default_factory=dict)

class HologramRenderer:
    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()
        
    def render_hologram(self,
                       hologram_data: np.ndarray,
                       mode: Optional[RenderMode] = None) -> Dict:
        """Render hologram visualization"""
        try:
            mode = mode or self.config.render_mode
            
            if mode == RenderMode.AMPLITUDE:
                fig = self._render_amplitude(hologram_data)
            elif mode == RenderMode.PHASE:
                fig = self._render_phase(hologram_data)
            elif mode == RenderMode.INTENSITY:
                fig = self._render_intensity(hologram_data)
            elif mode == RenderMode.COMBINED:
                fig = self._render_combined(hologram_data)
            else:
                raise ValueError(f"Unknown render mode: {mode}")
                
            return {
                "success": True,
                "figure": fig
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def _render_amplitude(self, data: np.ndarray) -> go.Figure:
        """Render amplitude visualization"""
        fig = go.Figure(data=go.Heatmap(
            z=np.abs(data),
            colorscale="Viridis",
            colorbar=dict(title="Amplitude")
        ))
        
        fig.update_layout(
            title="Hologram Amplitude",
            xaxis_title="X (pixels)",
            yaxis_title="Y (pixels)"
        )
        
        return fig 