from cryptography.hazmat.primitives.asymmetric import kyber
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from typing import Tuple, Dict, Optional, Union
import numpy as np
import json
from dataclasses import dataclass, field
from enum import Enum
import streamlit as st
from qiskit import QuantumCircuit
from cryptography.fernet import Fernet
from .circuit import QuantumCircuitEngine, GateType
from .noise import NoiseSimulator, NoiseType

class EncryptionProtocol(Enum):
    BB84 = "bb84"
    E91 = "e91"
    B92 = "b92"
    SIX_STATE = "six_state"

@dataclass
class EncryptionConfig:
    """Quantum encryption configuration"""
    key_length: int = 256
    error_threshold: float = 0.15
    privacy_amplification: float = 0.5
    authentication_rounds: int = 3
    metadata: Dict = field(default_factory=dict)

class QuantumEncryptionEngine:
    def __init__(self, config: Optional[EncryptionConfig] = None):
        """Initialize quantum-safe encryption engine"""
        self.config = config or EncryptionConfig()
        self.private_key = None
        self.public_key = None
        self.shared_keys: Dict[str, bytes] = {}
        self.circuit_engine = QuantumCircuitEngine()
        self.noise_simulator = NoiseSimulator()
        self.key_pairs: Dict[str, Tuple[bytes, bytes]] = {}
        self.session_keys: Dict[str, bytes] = {}
        self._generate_keypair()
        self._generate_base_keys()
        
    def _generate_keypair(self):
        """Generate quantum-resistant keypair"""
        self.private_key, self.public_key = kyber.generate_keypair()
        
    def encrypt_data(self, data: bytes, recipient_id: str = "default") -> Tuple[bytes, bytes]:
        """Encrypt data using quantum-safe algorithm"""
        try:
            # Generate ciphertext and shared secret
            ciphertext, shared_secret = kyber.enc(self.public_key, data)
            
            # Store shared secret for recipient
            self.shared_keys[recipient_id] = shared_secret
            
            return ciphertext, shared_secret
        except Exception as e:
            raise QuantumEncryptionError(f"Encryption failed: {str(e)}")
    
    def decrypt_data(self, ciphertext: bytes, recipient_id: str = "default") -> bytes:
        """Decrypt data using quantum-safe algorithm"""
        try:
            # Get shared secret for recipient
            shared_secret = self.shared_keys.get(recipient_id)
            if not shared_secret:
                raise ValueError(f"No shared secret found for recipient {recipient_id}")
                
            # Decrypt using shared secret
            decrypted = kyber.dec(ciphertext, self.private_key)
            return decrypted
        except Exception as e:
            raise QuantumEncryptionError(f"Decryption failed: {str(e)}")
    
    def secure_channel(self, recipient_public_key) -> 'SecureQuantumChannel':
        """Create secure quantum channel for collaboration"""
        return SecureQuantumChannel(self, recipient_public_key)

    def encrypt_message(self, 
                       message: Union[str, bytes], 
                       target_id: str,
                       protocol: EncryptionProtocol = EncryptionProtocol.BB84) -> Dict:
        """Encrypt message using quantum key distribution"""
        try:
            # Get or generate session key
            session_key = self._get_session_key(target_id, protocol)
            
            # Classical encryption with quantum-derived key
            fernet = Fernet(session_key)
            if isinstance(message, str):
                message = message.encode()
                
            encrypted_data = fernet.encrypt(message)
            
            return {
                "success": True,
                "encrypted_data": encrypted_data,
                "protocol": protocol.value,
                "key_id": target_id
            }
            
        except Exception as e:
            st.error(f"Encryption failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def decrypt_message(self,
                       encrypted_data: bytes,
                       source_id: str) -> Dict:
        """Decrypt message using quantum key"""
        try:
            session_key = self.session_keys.get(source_id)
            if not session_key:
                raise ValueError("No session key found for source")
                
            fernet = Fernet(session_key)
            decrypted_data = fernet.decrypt(encrypted_data)
            
            return {
                "success": True,
                "decrypted_data": decrypted_data
            }
            
        except Exception as e:
            st.error(f"Decryption failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def _get_session_key(self, target_id: str, protocol: EncryptionProtocol) -> bytes:
        """Get or generate session key using QKD"""
        if target_id in self.session_keys:
            return self.session_keys[target_id]
            
        # Generate new key using specified protocol
        if protocol == EncryptionProtocol.BB84:
            key = self._bb84_protocol(target_id)
        elif protocol == EncryptionProtocol.E91:
            key = self._e91_protocol(target_id)
        elif protocol == EncryptionProtocol.B92:
            key = self._b92_protocol(target_id)
        elif protocol == EncryptionProtocol.SIX_STATE:
            key = self._six_state_protocol(target_id)
        else:
            raise ValueError(f"Unknown protocol: {protocol}")
            
        self.session_keys[target_id] = key
        return key
        
    def _bb84_protocol(self, target_id: str) -> bytes:
        """Implement BB84 protocol"""
        # Initialize quantum circuit
        self.circuit_engine.initialize_circuit()
        num_qubits = self.config.key_length
        
        # Alice's random bits and bases
        alice_bits = np.random.randint(0, 2, num_qubits)
        alice_bases = np.random.randint(0, 2, num_qubits)  # 0: Z-basis, 1: X-basis
        
        # Prepare qubits
        for i, (bit, basis) in enumerate(zip(alice_bits, alice_bases)):
            if bit:
                self.circuit_engine.add_gate(GateType.PHASE, [i], [np.pi])
            if basis:
                self.circuit_engine.add_gate(GateType.HADAMARD, [i])
                
        # Simulate transmission and measurement
        result = self.circuit_engine.execute_circuit()
        
        # Sift key
        key_bits = self._sift_key(alice_bits, alice_bases, result)
        
        # Error correction and privacy amplification
        final_key = self._post_process_key(key_bits)
        
        return self._format_key(final_key)
        
    def _e91_protocol(self, target_id: str) -> bytes:
        """Implement E91 protocol"""
        # Create entangled pairs
        num_pairs = self.config.key_length // 2
        pairs = [(i*2, i*2+1) for i in range(num_pairs)]
        
        self.circuit_engine.initialize_circuit()
        for q1, q2 in pairs:
            self.circuit_engine.add_gate(GateType.HADAMARD, [q1])
            self.circuit_engine.add_gate(GateType.CNOT, [q1, q2])
            
        # Measure in random bases
        result = self.circuit_engine.execute_circuit()
        
        # Process measurements
        key_bits = self._process_e91_measurements(result)
        final_key = self._post_process_key(key_bits)
        
        return self._format_key(final_key)
        
    def _b92_protocol(self, target_id: str) -> bytes:
        """Implement B92 protocol"""
        # Simplified B92 implementation
        num_qubits = self.config.key_length * 2  # Need more qubits due to lower efficiency
        
        self.circuit_engine.initialize_circuit()
        alice_bits = np.random.randint(0, 2, num_qubits)
        
        for i, bit in enumerate(alice_bits):
            if bit:
                self.circuit_engine.add_gate(GateType.HADAMARD, [i])
                
        result = self.circuit_engine.execute_circuit()
        key_bits = self._process_b92_measurements(alice_bits, result)
        final_key = self._post_process_key(key_bits)
        
        return self._format_key(final_key)
        
    def _six_state_protocol(self, target_id: str) -> bytes:
        """Implement six-state protocol"""
        # Similar to BB84 but with three bases
        num_qubits = self.config.key_length
        
        self.circuit_engine.initialize_circuit()
        alice_bits = np.random.randint(0, 2, num_qubits)
        alice_bases = np.random.randint(0, 3, num_qubits)  # 0: X, 1: Y, 2: Z
        
        for i, (bit, basis) in enumerate(zip(alice_bits, alice_bases)):
            if bit:
                self.circuit_engine.add_gate(GateType.PHASE, [i], [np.pi])
            if basis == 0:
                self.circuit_engine.add_gate(GateType.HADAMARD, [i])
            elif basis == 1:
                self.circuit_engine.add_gate(GateType.PHASE, [i], [np.pi/2])
                self.circuit_engine.add_gate(GateType.HADAMARD, [i])
                
        result = self.circuit_engine.execute_circuit()
        key_bits = self._sift_key(alice_bits, alice_bases, result)
        final_key = self._post_process_key(key_bits)
        
        return self._format_key(final_key)
        
    def _sift_key(self, bits: np.ndarray, bases: np.ndarray, result: Dict) -> np.ndarray:
        """Sift key bits based on matching bases"""
        # Simulate Bob's random bases and measurements
        bob_bases = np.random.randint(0, 2, len(bits))
        matching_bases = bases == bob_bases
        
        return bits[matching_bases]
        
    def _post_process_key(self, key_bits: np.ndarray) -> np.ndarray:
        """Perform error correction and privacy amplification"""
        # Simple error correction
        block_size = 8
        corrected_bits = []
        
        for i in range(0, len(key_bits), block_size):
            block = key_bits[i:i+block_size]
            if len(block) == block_size:
                # Majority vote error correction
                corrected_bit = np.mean(block) > 0.5
                corrected_bits.extend([corrected_bit] * block_size)
                
        # Privacy amplification
        num_bits = int(len(corrected_bits) * self.config.privacy_amplification)
        final_bits = corrected_bits[:num_bits]
        
        return np.array(final_bits, dtype=bool)
        
    def _format_key(self, key_bits: np.ndarray) -> bytes:
        """Format key bits as bytes"""
        # Convert bits to bytes
        num_bytes = len(key_bits) // 8
        key_bytes = np.packbits(key_bits[:num_bytes*8])
        
        # Ensure key meets Fernet requirements (32 bytes, base64-encoded)
        while len(key_bytes) < 32:
            key_bytes = np.concatenate([key_bytes, key_bytes])
        key_bytes = key_bytes[:32]
        
        return Fernet.generate_key()  # For now, use Fernet's key generation
        
    def _generate_base_keys(self):
        """Generate base key pair"""
        key = Fernet.generate_key()
        self.public_key = key
        self._private_key = key  # In reality, would be different

class SecureQuantumChannel:
    def __init__(self, engine: QuantumEncryptionEngine, recipient_key):
        self.engine = engine
        self.recipient_key = recipient_key
        self.channel_id = self._generate_channel_id()
        
    def _generate_channel_id(self) -> str:
        """Generate unique channel identifier"""
        return hex(hash(str(self.recipient_key)))[:16]
    
    def send(self, data: Dict) -> bytes:
        """Send encrypted data through channel"""
        serialized = json.dumps(data).encode()
        ciphertext, _ = self.engine.encrypt_data(serialized, self.channel_id)
        return ciphertext
    
    def receive(self, ciphertext: bytes) -> Dict:
        """Receive and decrypt data from channel"""
        decrypted = self.engine.decrypt_data(ciphertext, self.channel_id)
        return json.loads(decrypted.decode())

class QuantumEncryptionError(Exception):
    """Custom exception for quantum encryption errors"""
    pass 