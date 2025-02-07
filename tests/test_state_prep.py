import pytest
import numpy as np
from src.ecs_sehi_analyzer.core.quantum.state_prep import (
    QuantumStatePreparator,
    StateConfig,
    StateType
)

@pytest.fixture
def preparator():
    return QuantumStatePreparator()

def test_computational_basis_state(preparator):
    params = {"basis_state": 1}
    result = preparator.prepare_state(StateType.COMPUTATIONAL, params)
    
    assert result["success"]
    assert len(result["state"]) == 2**preparator.config.num_qubits
    assert result["fidelity"] > 0.99

def test_bell_state(preparator):
    result = preparator.prepare_state(StateType.BELL)
    
    assert result["success"]
    state = result["state"]
    # Check if it's a valid Bell state
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
    fidelity = np.abs(np.vdot(state, expected))**2
    assert fidelity > 0.99

def test_ghz_state(preparator):
    config = StateConfig(num_qubits=3)
    preparator = QuantumStatePreparator(config)
    
    result = preparator.prepare_state(StateType.GHZ)
    
    assert result["success"]
    state = result["state"]
    # Check if it's a valid GHZ state
    expected = np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)
    fidelity = np.abs(np.vdot(state, expected))**2
    assert fidelity > 0.99

def test_w_state(preparator):
    config = StateConfig(num_qubits=3)
    preparator = QuantumStatePreparator(config)
    
    result = preparator.prepare_state(StateType.W)
    
    assert result["success"]
    state = result["state"]
    # Check W state properties
    probabilities = np.abs(state)**2
    assert np.sum(probabilities) ≈ 1.0
    assert len(np.where(probabilities > 0.3)[0]) == 3

def test_cluster_state(preparator):
    result = preparator.prepare_state(StateType.CLUSTER)
    
    assert result["success"]
    assert result["fidelity"] > 0.99
    # Verify entanglement properties
    state = result["state"]
    assert len(state) == 2**preparator.config.num_qubits

def test_custom_state(preparator):
    custom_state = np.array([1, 1j]) / np.sqrt(2)
    params = {"state_vector": custom_state}
    
    result = preparator.prepare_state(StateType.CUSTOM, params)
    
    assert result["success"]
    assert np.allclose(result["state"], custom_state, atol=1e-6)
    assert result["fidelity"] > 0.99

def test_invalid_custom_state(preparator):
    result = preparator.prepare_state(StateType.CUSTOM)
    assert not result["success"]
    assert "error" in result

def test_state_normalization(preparator):
    # Test with unnormalized state
    custom_state = np.array([2, 2])
    params = {"state_vector": custom_state}
    
    result = preparator.prepare_state(StateType.CUSTOM, params)
    
    assert result["success"]
    prepared_state = result["state"]
    assert np.abs(np.sum(np.abs(prepared_state)**2) - 1.0) < 1e-6

def test_config_options():
    config = StateConfig(
        num_qubits=4,
        fidelity_threshold=0.95,
        optimization_steps=200
    )
    preparator = QuantumStatePreparator(config)
    
    assert preparator.config.num_qubits == 4
    assert preparator.config.fidelity_threshold == 0.95
    assert preparator.config.optimization_steps == 200

def test_fidelity_calculation(preparator):
    state1 = np.array([1, 0])
    state2 = np.array([1, 0])
    
    preparator.target_state = state1
    fidelity = preparator._calculate_fidelity(state2)
    
    assert np.abs(fidelity - 1.0) < 1e-6 