import pytest
import numpy as np
from src.ecs_sehi_analyzer.core.quantum.circuit import (
    QuantumCircuitEngine,
    CircuitConfig,
    GateType
)

@pytest.fixture
def circuit_engine():
    return QuantumCircuitEngine()

def test_circuit_initialization(circuit_engine):
    circuit_engine.initialize_circuit()
    assert circuit_engine.circuit is not None
    assert circuit_engine.circuit.num_qubits == circuit_engine.config.num_qubits

def test_gate_addition(circuit_engine):
    circuit_engine.initialize_circuit()
    
    # Add Hadamard gate
    circuit_engine.add_gate(GateType.HADAMARD, [0])
    
    # Add CNOT gate
    circuit_engine.add_gate(GateType.CNOT, [0, 1])
    
    # Add Phase gate
    circuit_engine.add_gate(GateType.PHASE, [0], [np.pi/2])
    
    assert len(circuit_engine.circuit.data) == 3

def test_circuit_execution(circuit_engine):
    circuit_engine.initialize_circuit()
    circuit_engine.add_gate(GateType.HADAMARD, [0])
    circuit_engine.add_gate(GateType.MEASUREMENT, [0])
    
    result = circuit_engine.execute_circuit()
    
    assert "counts" in result
    assert "success_rate" in result
    assert "quantum_state" in result
    assert "circuit_depth" in result

def test_invalid_gate_parameters(circuit_engine):
    circuit_engine.initialize_circuit()
    
    # Test CNOT with wrong number of qubits
    with pytest.raises(ValueError):
        circuit_engine.add_gate(GateType.CNOT, [0])
    
    # Test Phase gate without parameters
    with pytest.raises(ValueError):
        circuit_engine.add_gate(GateType.PHASE, [0])

def test_circuit_statistics(circuit_engine):
    circuit_engine.initialize_circuit()
    circuit_engine.add_gate(GateType.HADAMARD, [0])
    circuit_engine.add_gate(GateType.CNOT, [0, 1])
    circuit_engine.add_gate(GateType.MEASUREMENT, [0, 1])
    
    stats = circuit_engine.get_circuit_statistics()
    
    assert "num_qubits" in stats
    assert "depth" in stats
    assert "size" in stats
    assert "num_measurements" in stats

def test_circuit_reset(circuit_engine):
    circuit_engine.initialize_circuit()
    circuit_engine.add_gate(GateType.HADAMARD, [0])
    
    # Execute and record results
    circuit_engine.execute_circuit()
    assert len(circuit_engine.measurement_results) == 1
    
    # Reset circuit
    circuit_engine.reset_circuit()
    assert len(circuit_engine.measurement_results) == 0
    assert len(circuit_engine.circuit.data) == 0 