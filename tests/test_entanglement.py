import pytest
import numpy as np
from datetime import datetime
from src.ecs_sehi_analyzer.core.quantum.entanglement import (
    QuantumEntanglementEngine,
    EntanglementNode,
    SyncState,
    EntanglementConfig,
    EntanglementType,
    QuantumChannel
)

@pytest.fixture
def engine():
    return QuantumEntanglementEngine()

@pytest.fixture
def sample_nodes(engine):
    nodes = []
    for i in range(2):
        node = engine.create_node(f"node_{i}")
        nodes.append(node)
    return nodes

@pytest.fixture
def qubit_pairs():
    return [(0, 1), (2, 3)]

def test_node_creation(engine):
    node = engine.create_node("test_node")
    assert isinstance(node, EntanglementNode)
    assert node.node_id == "test_node"
    assert node.sync_state == SyncState.READY

def test_duplicate_node(engine):
    engine.create_node("test_node")
    with pytest.raises(ValueError):
        engine.create_node("test_node")

def test_entanglement_establishment(engine, sample_nodes):
    result = engine.establish_entanglement("node_0", "node_1")
    assert result is True
    
    node_a = engine.nodes["node_0"]
    node_b = engine.nodes["node_1"]
    
    assert "node_1" in node_a.peers
    assert "node_0" in node_b.peers
    assert node_a.sync_state == SyncState.ENTANGLED
    assert node_b.sync_state == SyncState.ENTANGLED

def test_data_sync(engine, sample_nodes):
    engine.establish_entanglement("node_0", "node_1")
    
    test_data = {"key": "value"}
    result = engine.sync_data("node_0", "node_1", test_data)
    
    assert result["status"] == "success"
    assert isinstance(result["synced_at"], datetime)

def test_invalid_sync(engine):
    with pytest.raises(ValueError):
        engine.sync_data("invalid_a", "invalid_b", {})

def test_bell_pair_creation(engine, qubit_pairs):
    result = engine.create_entanglement(
        qubit_pairs,
        EntanglementType.BELL_PAIR
    )
    
    assert result["success"]
    assert result["fidelity"] > 0.9
    assert result["type"] == "bell_pair"
    assert result["pairs"] == qubit_pairs

def test_ghz_state_creation(engine, qubit_pairs):
    result = engine.create_entanglement(
        qubit_pairs,
        EntanglementType.GHZ
    )
    
    assert result["success"]
    assert result["fidelity"] > 0.9
    assert result["type"] == "ghz"

def test_cluster_state_creation(engine, qubit_pairs):
    result = engine.create_entanglement(
        qubit_pairs,
        EntanglementType.CLUSTER
    )
    
    assert result["success"]
    assert result["fidelity"] > 0.9
    assert result["type"] == "cluster"

def test_invalid_pairs(engine):
    invalid_pairs = [(0, 1), (1, 2)]  # Overlapping qubits
    
    result = engine.create_entanglement(
        invalid_pairs,
        EntanglementType.BELL_PAIR
    )
    
    assert not result["success"]
    assert "error" in result

def test_fidelity_tracking(engine, qubit_pairs):
    # Create multiple entanglements
    for _ in range(3):
        engine.create_entanglement(
            qubit_pairs,
            EntanglementType.BELL_PAIR
        )
    
    assert len(engine.fidelity_history) == 3
    assert all(0 <= f <= 1 for f in engine.fidelity_history)

def test_quantum_channel(engine):
    channel = engine.secure_channel("target_id")
    
    assert isinstance(channel, QuantumChannel)
    assert channel.source_id == id(engine)
    assert channel.target_id == "target_id"
    
    # Test data transmission
    data = {"test": "data"}
    transmitted = channel.send(data)
    received = channel.receive(transmitted)
    
    assert received == data

def test_noise_effects(engine, qubit_pairs):
    # Set high noise level
    config = EntanglementConfig(noise_level=0.1)
    engine = QuantumEntanglementEngine(config)
    
    result = engine.create_entanglement(
        qubit_pairs,
        EntanglementType.BELL_PAIR
    )
    
    # High noise should reduce fidelity
    assert result["fidelity"] < 0.99

def test_entanglement_config():
    config = EntanglementConfig(
        num_qubits=6,
        fidelity_threshold=0.98,
        max_distance=500.0
    )
    engine = QuantumEntanglementEngine(config)
    
    assert engine.config.num_qubits == 6
    assert engine.config.fidelity_threshold == 0.98
    assert engine.config.max_distance == 500.0 