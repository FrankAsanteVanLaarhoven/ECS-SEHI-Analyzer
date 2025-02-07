import pytest
import json
from src.ecs_sehi_analyzer.core.quantum.encryption import (
    QuantumEncryptionEngine,
    SecureQuantumChannel,
    QuantumEncryptionError
)

@pytest.fixture
def encryption_engine():
    return QuantumEncryptionEngine()

@pytest.fixture
def sample_data():
    return {
        "material_id": "TEST-001",
        "analysis_results": {
            "degradation_score": 0.75,
            "confidence": 0.92
        }
    }

def test_encryption_decryption(encryption_engine, sample_data):
    # Test basic encryption/decryption
    data = json.dumps(sample_data).encode()
    ciphertext, shared_secret = encryption_engine.encrypt_data(data)
    decrypted = encryption_engine.decrypt_data(ciphertext)
    
    assert json.loads(decrypted.decode()) == sample_data

def test_secure_channel(encryption_engine, sample_data):
    # Create two encryption engines for bidirectional communication
    alice_engine = QuantumEncryptionEngine()
    bob_engine = QuantumEncryptionEngine()
    
    # Create secure channels
    alice_channel = alice_engine.secure_channel(bob_engine.public_key)
    bob_channel = bob_engine.secure_channel(alice_engine.public_key)
    
    # Test bidirectional communication
    encrypted_data = alice_channel.send(sample_data)
    received_data = bob_channel.receive(encrypted_data)
    
    assert received_data == sample_data

def test_invalid_decryption(encryption_engine):
    with pytest.raises(QuantumEncryptionError):
        encryption_engine.decrypt_data(b"invalid_ciphertext")

def test_channel_id_consistency(encryption_engine):
    recipient_key = QuantumEncryptionEngine().public_key
    channel1 = encryption_engine.secure_channel(recipient_key)
    channel2 = encryption_engine.secure_channel(recipient_key)
    
    assert channel1.channel_id == channel2.channel_id 