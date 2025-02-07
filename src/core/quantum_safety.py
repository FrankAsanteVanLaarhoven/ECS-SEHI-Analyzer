from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend

class QuantumSafeEncryptor:
    def __init__(self, algorithm=hashes.SHA3_256()):
        self.algorithm = algorithm
    
    def derive_key(self, salt=None):
        """Derive quantum-safe encryption key"""
        return HKDF(
            algorithm=self.algorithm,
            length=32,
            salt=salt,
            info=b'ecs-sehi-encryption',
            backend=default_backend()
        )
