import pytest
import numpy as np
from src.ecs_sehi_analyzer.core.quantum.noise import (
    NoiseSimulator,
    NoiseConfig,
    NoiseType
)

@pytest.fixture
def simulator():
    return NoiseSimulator()

def test_noise_model_creation(simulator):
    for noise_type in NoiseType:
        model = simulator.create_noise_model(noise_type)
        assert model is not None
        assert len(model.basis_gates) > 0

def test_depolarizing_noise(simulator):
    model = simulator.create_noise_model(NoiseType.DEPOLARIZING)
    
    # Check that error probabilities are set
    assert model.noise_instructions
    assert "u3" in model.noise_instructions
    assert "cx" in model.noise_instructions

def test_thermal_noise(simulator):
    config = NoiseConfig(
        t1=20e-6,  # Short T1
        t2=30e-6   # Short T2
    )
    simulator = NoiseSimulator(config)
    model = simulator.create_noise_model(NoiseType.THERMAL)
    
    analysis = simulator.analyze_noise_impact()
    assert analysis["coherence_time"] == 20e-6  # Should be min(T1, T2)
    assert "thermal_relaxation" in [c["type"] for c in analysis["error_channels"]]

def test_readout_noise(simulator):
    config = NoiseConfig(
        error_probabilities={"measurement": 0.1}
    )
    simulator = NoiseSimulator(config)
    model = simulator.create_noise_model(NoiseType.READOUT)
    
    # Check that readout error is set
    assert model.readout_errors

def test_noise_analysis(simulator):
    simulator.create_noise_model(NoiseType.DEPOLARIZING)
    analysis = simulator.analyze_noise_impact()
    
    assert "error_rates" in analysis
    assert "coherence_time" in analysis
    assert "gate_fidelities" in analysis
    assert "error_channels" in analysis

def test_gate_fidelities(simulator):
    config = NoiseConfig(
        error_probabilities={
            "single_qubit": 0.01,
            "two_qubit": 0.05,
            "measurement": 0.02
        }
    )
    simulator = NoiseSimulator(config)
    
    fidelities = simulator._calculate_gate_fidelities()
    assert fidelities["single_qubit"] == 0.99
    assert fidelities["two_qubit"] == 0.95
    assert fidelities["measurement"] == 0.98

def test_error_channels(simulator):
    config = NoiseConfig(
        t1=10e-6,  # Very short T1
        error_probabilities={"two_qubit": 0.1}  # High gate error
    )
    simulator = NoiseSimulator(config)
    
    channels = simulator._identify_error_channels()
    assert len(channels) == 2  # Should identify both issues
    assert any(c["type"] == "thermal_relaxation" for c in channels)
    assert any(c["type"] == "two_qubit_gate" for c in channels)

def test_custom_noise(simulator):
    model = simulator.create_noise_model(NoiseType.CUSTOM)
    
    # Custom noise should include multiple error types
    assert model.noise_instructions  # Has quantum errors
    assert model.readout_errors     # Has readout errors 