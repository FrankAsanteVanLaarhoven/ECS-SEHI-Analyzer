from cryptography.fernet import Fernet
import hashlib

class QuantumSafeEncryption:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
        self.post_quantum_salt = hashlib.shake_256().digest(32)

    def encrypt_data(self, data: bytes) -> bytes:
        return self.cipher.encrypt(data)

    def decrypt_data(self, token: bytes) -> bytes:
        return self.cipher.decrypt(token)
