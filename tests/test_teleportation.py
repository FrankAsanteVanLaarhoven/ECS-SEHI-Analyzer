import pytest
import numpy as np
from src.ecs_sehi_analyzer.core.quantum.teleportation import (
    QuantumTeleporter,
    TeleportationConfig,
    TeleportationState
)

@pytest.fixture
def teleporter():
    return QuantumTeleporter()

@pytest.fixture
def sample_state():
    # Create normalized state vector
    state = np.array([1, 1j]) / np.sqrt(2)
    return state

def test_teleportation(teleporter, sample_state):
    result = teleporter.teleport_state(sample_state, "target_1")
    
    assert result["success"]
    assert result["fidelity"] > 0.9
    assert "measurements" in result
    assert result["state"] == TeleportationState.CORRECTED.value

def test_entanglement_creation(teleporter):
    success = teleporter._create_entangled_pair()
    assert success
    assert teleporter.state == TeleportationState.ENTANGLED

def test_bell_measurement(teleporter, sample_state):
    teleporter._create_entangled_pair()
    teleporter._prepare_input_state(sample_state)
    
    measurements = teleporter._bell_measurement()
    assert "q0" in measurements
    assert "q1" in measurements
    assert teleporter.state == TeleportationState.MEASURED

def test_corrections(teleporter):
    measurements = {"q0": 1, "q1": 0}
    teleporter._create_entangled_pair()
    teleporter._apply_corrections(measurements)
    
    assert teleporter.state == TeleportationState.CORRECTED

def test_invalid_state(teleporter):
    invalid_state = np.array([1, 0, 0])  # Wrong dimensions
    
    result = teleporter.teleport_state(invalid_state, "target_1")
    assert not result["success"]
    assert "error" in result
    assert result["state"] == TeleportationState.ERROR.value

def test_fidelity_calculation(teleporter):
    state1 = np.array([1, 0])
    state2 = np.array([1, 0])  # Same state
    fidelity = teleporter._calculate_fidelity(state1, state2)
    assert np.abs(fidelity - 1.0) < 1e-6
    
    state3 = np.array([0, 1])  # Orthogonal state
    fidelity = teleporter._calculate_fidelity(state1, state3)
    assert np.abs(fidelity) < 1e-6

def test_teleportation_history(teleporter, sample_state):
    # Perform multiple teleportations
    for _ in range(3):
        teleporter.teleport_state(sample_state, "target_1")
    
    assert len(teleporter.teleportation_history) == 3
    for result in teleporter.teleportation_history:
        assert "success" in result
        assert "fidelity" in result

def test_config_options():
    config = TeleportationConfig(
        num_qubits=4,
        fidelity_threshold=0.98,
        error_correction=False
    )
    teleporter = QuantumTeleporter(config)
    
    assert teleporter.config.num_qubits == 4
    assert teleporter.config.fidelity_threshold == 0.98
    assert not teleporter.config.error_correction 