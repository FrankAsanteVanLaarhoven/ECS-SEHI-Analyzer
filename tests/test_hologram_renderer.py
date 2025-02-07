import pytest
import numpy as np
from src.ecs_sehi_analyzer.core.hologram.hologram_renderer import (
    HologramRenderer,
    RenderConfig,
    RenderMode
)

@pytest.fixture
def renderer():
    return HologramRenderer()

@pytest.fixture
def sample_hologram():
    size = 32  # Smaller size for testing
    return np.exp(1j * np.random.random((size, size)))

def test_amplitude_rendering(renderer, sample_hologram):
    result = renderer.render_hologram(sample_hologram, RenderMode.AMPLITUDE)
    
    assert result["success"]
    assert "figure" in result
    # Check figure properties
    fig = result["figure"]
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"

def test_phase_rendering(renderer, sample_hologram):
    result = renderer.render_hologram(sample_hologram, RenderMode.PHASE)
    
    assert result["success"]
    assert "figure" in result
    fig = result["figure"]
    assert "Phase" in fig.layout.title.text

def test_intensity_rendering(renderer, sample_hologram):
    result = renderer.render_hologram(sample_hologram, RenderMode.INTENSITY)
    
    assert result["success"]
    assert "figure" in result
    fig = result["figure"]
    assert "Intensity" in fig.layout.title.text

def test_combined_rendering(renderer, sample_hologram):
    result = renderer.render_hologram(sample_hologram, RenderMode.COMBINED)
    
    assert result["success"]
    assert "figure" in result
    fig = result["figure"]
    # Combined mode should have multiple subplots
    assert len(fig.data) > 1

def test_invalid_render_mode(renderer, sample_hologram):
    with pytest.raises(ValueError):
        renderer.render_hologram(sample_hologram, "invalid_mode")

def test_config_options():
    config = RenderConfig(
        resolution=(512, 512),
        wavelength=532e-9,  # Green laser
        pixel_size=5e-6
    )
    renderer = HologramRenderer(config)
    
    assert renderer.config.resolution == (512, 512)
    assert renderer.config.wavelength == 532e-9
    assert renderer.config.pixel_size == 5e-6

def test_invalid_hologram_data(renderer):
    invalid_data = np.random.random(3)  # 1D array
    result = renderer.render_hologram(invalid_data, RenderMode.AMPLITUDE)
    
    assert not result["success"]
    assert "error" in result

def test_render_quality(renderer, sample_hologram):
    config = RenderConfig(render_quality="low")
    renderer = HologramRenderer(config)
    
    result = renderer.render_hologram(sample_hologram, RenderMode.AMPLITUDE)
    assert result["success"]