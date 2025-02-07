import pytest
import numpy as np
from src.ecs_sehi_analyzer.core.hologram.hologram_engine import (
    HologramEngine,
    HologramConfig
)

@pytest.fixture
def engine():
    return HologramEngine()

@pytest.fixture
def sample_data():
    # Create synthetic 3D data
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    return Z

@pytest.fixture
def sample_4d_data():
    # Create synthetic 4D data (time series of 3D data)
    data = []
    for t in range(10):
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2) - t*0.1)
        data.append(Z)
    return np.array(data)

def test_hologram_creation(engine, sample_data):
    result = engine.create_hologram(sample_data)
    assert "layers" in result
    assert result["layers"] == engine.config.depth_layers
    assert len(engine.layers) == engine.config.depth_layers

def test_4d_hologram_creation(engine, sample_4d_data):
    result = engine.create_hologram(sample_4d_data)
    assert "layers" in result
    assert result["layers"] == engine.config.depth_layers
    assert len(engine.layers) == engine.config.depth_layers

def test_invalid_data_shape(engine):
    invalid_data = np.random.rand(10)  # 1D data
    with pytest.raises(ValueError):
        engine.create_hologram(invalid_data)

def test_depth_layer_generation(engine, sample_data):
    engine.create_hologram(sample_data)
    
    # Check layer properties
    for layer in engine.layers:
        assert layer.shape == sample_data.shape
        assert np.all(layer >= 0)  # Non-negative values
        assert np.all(layer <= 1)  # Normalized values

def test_config_update(engine, sample_data):
    # Update configuration
    new_config = HologramConfig(
        depth_layers=8,
        opacity_threshold=0.2
    )
    engine = HologramEngine(new_config)
    
    result = engine.create_hologram(sample_data)
    assert result["layers"] == 8
    assert len(engine.layers) == 8 