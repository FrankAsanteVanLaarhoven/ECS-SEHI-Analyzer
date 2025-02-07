import pytest
import numpy as np
from src.ecs_sehi_analyzer.core.quantum.visualizer import (
    QuantumVisualizer,
    VisualizerConfig,
    VisualizationType
)

@pytest.fixture
def visualizer():
    return QuantumVisualizer()

@pytest.fixture
def sample_state():
    # Create normalized single-qubit state
    return np.array([1/np.sqrt(2), 1/np.sqrt(2)])

@pytest.fixture
def sample_process():
    return {
        "type": "sample",
        "num_qubits": 2,
        "operations": [
            {"gate": "h", "qubits": [0]},
            {"gate": "cx", "qubits": [0, 1]}
        ]
    }

def test_bloch_sphere_visualization(visualizer, sample_state):
    result = visualizer.visualize_state(
        sample_state,
        VisualizationType.BLOCH_SPHERE
    )
    
    assert result["success"]
    assert "figure" in result
    assert result["metadata"]["type"] == "bloch_sphere"
    assert result["metadata"]["state_dim"] == 2

def test_invalid_bloch_sphere_state(visualizer):
    # Test with invalid 3-dimensional state
    invalid_state = np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])
    
    result = visualizer.visualize_state(
        invalid_state,
        VisualizationType.BLOCH_SPHERE
    )
    
    assert not result["success"]
    assert "error" in result

def test_density_matrix_visualization(visualizer, sample_state):
    result = visualizer.visualize_state(
        sample_state,
        VisualizationType.DENSITY_MATRIX
    )
    
    assert result["success"]
    assert "figure" in result
    assert result["metadata"]["type"] == "density_matrix"

def test_state_vector_visualization(visualizer):
    # Test with 3-qubit state
    state = np.array([1/np.sqrt(8)] * 8)
    
    result = visualizer.visualize_state(
        state,
        VisualizationType.STATE_VECTOR
    )
    
    assert result["success"]
    assert "figure" in result
    assert result["metadata"]["state_dim"] == 8

def test_circuit_diagram_visualization(visualizer, sample_process):
    result = visualizer.visualize_process(
        sample_process,
        VisualizationType.CIRCUIT_DIAGRAM
    )
    
    assert result["success"]
    assert "figure" in result

def test_process_matrix_visualization(visualizer, sample_process):
    result = visualizer.visualize_process(
        sample_process,
        VisualizationType.PROCESS_MATRIX
    )
    
    assert result["success"]
    assert "figure" in result

def test_visualization_history(visualizer, sample_state):
    # Perform multiple visualizations
    for vis_type in [
        VisualizationType.BLOCH_SPHERE,
        VisualizationType.DENSITY_MATRIX,
        VisualizationType.STATE_VECTOR
    ]:
        visualizer.visualize_state(sample_state, vis_type)
    
    assert len(visualizer.visualization_history) == 3
    
    for record in visualizer.visualization_history:
        assert "type" in record
        assert "state_dim" in record
        assert "params" in record

def test_invalid_visualization_type(visualizer, sample_state):
    with pytest.raises(ValueError):
        visualizer.visualize_state(
            sample_state,
            "invalid_type"
        )

def test_config_options():
    config = VisualizerConfig(
        resolution=200,
        colormap="plasma",
        animation_frames=100,
        render_quality="medium"
    )
    visualizer = QuantumVisualizer(config)
    
    assert visualizer.config.resolution == 200
    assert visualizer.config.colormap == "plasma"
    assert visualizer.config.animation_frames == 100
    assert visualizer.config.render_quality == "medium"

def test_custom_visualization_params(visualizer, sample_state):
    params = {
        "opacity": 0.5,
        "colorscale": "Viridis",
        "showlegend": True
    }
    
    result = visualizer.visualize_state(
        sample_state,
        VisualizationType.BLOCH_SPHERE,
        params
    )
    
    assert result["success"]
    assert result["metadata"]["params"] == params

def test_state_normalization(visualizer):
    # Test with unnormalized state
    state = np.array([2.0, 2.0])
    
    result = visualizer.visualize_state(
        state,
        VisualizationType.BLOCH_SPHERE
    )
    
    assert result["success"]
    # Verify the state was normalized
    assert np.abs(np.linalg.norm(visualizer.current_state) - 1.0) < 1e-6 