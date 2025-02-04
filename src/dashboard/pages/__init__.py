"""Dashboard pages initialization."""
from .surface_analysis import render_surface_analysis
from .chemical_analysis import render_chemical_analysis
from .defect_detection import render_defect_detection
from .analysis_results import render_analysis_results
from .sustainability_metrics import render_sustainability_metrics
from .multimodal_analysis import render_multimodal_analysis
from .landing import render_landing_page, handle_landing_messages

__all__ = [
    'render_surface_analysis',
    'render_chemical_analysis',
    'render_defect_detection',
    'render_analysis_results',
    'render_sustainability_metrics',
    'render_multimodal_analysis',
    'render_landing_page',
    'handle_landing_messages'
]
