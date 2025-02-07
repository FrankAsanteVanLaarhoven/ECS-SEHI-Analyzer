import pytest
import time
import numpy as np
from datetime import datetime
from src.ecs_sehi_analyzer.core.collaboration.realtime_editor import (
    CollaborationHub,
    ResearchDocument,
    CollaborationSession
)
from src.ecs_sehi_analyzer.core.quantum.collaboration import (
    QuantumCollaborator,
    CollaborationConfig,
    CollaborationMode
)

@pytest.fixture
def hub():
    return CollaborationHub()

@pytest.fixture
def document():
    return ResearchDocument()

@pytest.fixture
def collaborator():
    return QuantumCollaborator()

@pytest.fixture
def sample_public_key():
    return b"sample_key_123"

def test_session_creation(hub):
    session_id = hub.create_session("test_user")
    assert session_id in hub.active_sessions
    assert hub.active_sessions[session_id].owner == "test_user"

def test_document_versioning(document):
    # Test document updates
    result = document.update_content({"data": "test"}, "user1")
    assert result["version"] == 1
    assert "user1" in document.contributors
    
    # Test version retrieval
    content = document.get_version(1)
    assert content["data"] == "test"

def test_collaboration_workflow(hub):
    # Create session
    session_id = hub.create_session("user1")
    
    # Join session
    channel = hub.join_session(session_id, "user2")
    assert "user2" in hub.active_sessions[session_id].participants
    
    # Update document
    result = hub.update_document(session_id, {"data": "test"}, "user2")
    assert result["version"] == 1
    
    # Verify document
    document = hub.documents[session_id]
    assert document.content["data"] == "test"
    assert "user2" in document.contributors 

def test_collaborator_registration(collaborator, sample_public_key):
    result = collaborator.register_collaborator("alice", sample_public_key)
    
    assert result["success"]
    assert "alice" in collaborator.collaborators
    assert collaborator.collaborators["alice"]["public_key"] == sample_public_key

def test_duplicate_registration(collaborator, sample_public_key):
    collaborator.register_collaborator("alice", sample_public_key)
    
    with pytest.raises(ValueError):
        collaborator.register_collaborator("alice", sample_public_key)

def test_collaboration_initiation(collaborator, sample_public_key):
    # Register collaborator first
    collaborator.register_collaborator("bob", sample_public_key)
    
    # Test each collaboration mode
    for mode in CollaborationMode:
        result = collaborator.initiate_collaboration("bob", mode)
        assert result["success"]
        assert result["mode"] == mode.value
        assert "session_id" in result

def test_invalid_collaboration(collaborator):
    result = collaborator.initiate_collaboration(
        "invalid_id",
        CollaborationMode.SECURE_COMPUTE
    )
    assert not result["success"]
    assert "error" in result

def test_quantum_state_sharing(collaborator, sample_public_key):
    collaborator.register_collaborator("charlie", sample_public_key)
    
    # Share state securely
    state = np.array([1, 0]) / np.sqrt(2)
    result = collaborator.share_quantum_state(state, "charlie", secure=True)
    
    assert result["success"]
    assert result["fidelity"] > 0.9

def test_collaborative_circuit(collaborator, sample_public_key):
    # Register multiple collaborators
    collaborators = ["alice", "bob", "charlie"]
    for c in collaborators:
        collaborator.register_collaborator(c, sample_public_key)
        
    # Execute collaborative circuit
    circuit_data = {
        "operations": [
            {"gate": "h", "qubits": [0]},
            {"gate": "cx", "qubits": [0, 1]}
        ]
    }
    
    result = collaborator.execute_collaborative_circuit(
        circuit_data,
        collaborators
    )
    
    assert result["success"]
    assert "results" in result
    assert result["collaborators"] == collaborators

def test_session_history(collaborator, sample_public_key):
    collaborator.register_collaborator("alice", sample_public_key)
    
    # Create multiple sessions
    for mode in CollaborationMode:
        collaborator.initiate_collaboration("alice", mode)
        
    assert len(collaborator.session_history) == len(CollaborationMode)
    
    for session in collaborator.session_history:
        assert session["collaborator_id"] == "alice"
        assert session["status"] == "active"
        assert isinstance(session["started_at"], datetime)

def test_config_options():
    config = CollaborationConfig(
        num_parties=3,
        security_level="medium",
        max_qubit_transfer=50
    )
    collaborator = QuantumCollaborator(config)
    
    assert collaborator.config.num_parties == 3
    assert collaborator.config.security_level == "medium"
    assert collaborator.config.max_qubit_transfer == 50

def test_secure_compute_setup(collaborator, sample_public_key):
    collaborator.register_collaborator("alice", sample_public_key)
    
    result = collaborator.initiate_collaboration(
        "alice",
        CollaborationMode.SECURE_COMPUTE
    )
    
    assert result["success"]
    assert "channel_id" in result
    assert result["resources_allocated"] 