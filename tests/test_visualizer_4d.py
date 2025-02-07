import pytest
import numpy as np
from src.ecs_sehi_analyzer.core.visualization.visualizer_4d import (
    DataVisualizer4D,
    VisualizationConfig
)

@pytest.fixture
def visualizer():
    return DataVisualizer4D()

@pytest.fixture
def sample_data():
    # Create synthetic 4D data (time, x, y, features)
    t, x, y = 10, 50, 50
    data = np.random.normal(0, 1, (t, x, y, 1))
    timestamps = np.linspace(0, 10, t)
    return data, timestamps.tolist()

def test_visualization_creation(visualizer, sample_data):
    data, timestamps = sample_data
    fig = visualizer.create_4d_visualization(data, timestamps)
    assert fig is not None
    assert len(fig.frames) == len(timestamps)
    assert visualizer.figure is not None

def test_invalid_data_shape(visualizer):
    invalid_data = np.random.normal(0, 1, (10, 50, 50))  # Missing dimension
    timestamps = np.linspace(0, 10, 10).tolist()
    
    with pytest.raises(ValueError):
        visualizer.create_4d_visualization(invalid_data, timestamps)

def test_config_update(visualizer, sample_data):
    data, timestamps = sample_data
    
    # Create with custom config
    config = VisualizationConfig(
        colorscale="Plasma",
        opacity=0.5,
        surface_count=20
    )
    visualizer = DataVisualizer4D(config)
    
    fig = visualizer.create_4d_visualization(data, timestamps)
    assert fig is not None
    assert visualizer.config.colorscale == "Plasma"
    assert visualizer.config.opacity == 0.5
    assert visualizer.config.surface_count == 20 