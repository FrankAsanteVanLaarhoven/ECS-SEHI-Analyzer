import pytest
import numpy as np
from src.ecs_sehi_analyzer.core.quantum.encryption import (
    QuantumEncryptionEngine,
    EncryptionConfig,
    EncryptionProtocol
)

@pytest.fixture
def engine():
    return QuantumEncryptionEngine()

def test_encryption_decryption(engine):
    message = "test message"
    target_id = "test_target"
    
    # Encrypt
    result = engine.encrypt_message(message, target_id)
    assert result["success"]
    assert "encrypted_data" in result
    
    # Decrypt
    decrypt_result = engine.decrypt_message(
        result["encrypted_data"],
        target_id
    )
    assert decrypt_result["success"]
    assert decrypt_result["decrypted_data"].decode() == message

def test_protocols(engine):
    message = "test message"
    target_id = "test_target"
    
    for protocol in EncryptionProtocol:
        result = engine.encrypt_message(message, target_id, protocol)
        assert result["success"]
        assert result["protocol"] == protocol.value
        
        decrypt_result = engine.decrypt_message(
            result["encrypted_data"],
            target_id
        )
        assert decrypt_result["success"]

def test_key_generation(engine):
    target_id = "test_target"
    
    # Generate key using BB84
    key = engine._get_session_key(target_id, EncryptionProtocol.BB84)
    assert len(key) == 32  # Fernet key length
    
    # Key should be cached
    assert target_id in engine.session_keys
    assert engine.session_keys[target_id] == key

def test_error_handling(engine):
    # Invalid protocol
    with pytest.raises(ValueError):
        engine._get_session_key("test", "invalid_protocol")
        
    # Invalid source for decryption
    result = engine.decrypt_message(b"invalid", "invalid_source")
    assert not result["success"]
    assert "error" in result

def test_post_processing(engine):
    key_bits = np.random.randint(0, 2, 100)
    processed_key = engine._post_process_key(key_bits)
    
    # Check privacy amplification
    expected_length = int(len(key_bits) * engine.config.privacy_amplification)
    assert len(processed_key) >= expected_length

def test_sift_key(engine):
    bits = np.array([1, 0, 1, 1, 0])
    bases = np.array([0, 1, 0, 1, 1])
    result = {"counts": {}}  # Mock measurement result
    
    sifted_key = engine._sift_key(bits, bases, result)
    assert isinstance(sifted_key, np.ndarray)

def test_config_options(engine):
    config = EncryptionConfig(
        key_length=512,
        error_threshold=0.1,
        privacy_amplification=0.7
    )
    engine = QuantumEncryptionEngine(config)
    
    assert engine.config.key_length == 512
    assert engine.config.error_threshold == 0.1
    assert engine.config.privacy_amplification == 0.7 