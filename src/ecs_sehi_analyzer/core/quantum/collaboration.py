from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from enum import Enum
import streamlit as st
from datetime import datetime
from .encryption import QuantumEncryptionEngine, EncryptionProtocol
from .teleportation import QuantumTeleporter
from .entanglement import QuantumEntanglementEngine, EntanglementType
from .circuit import QuantumCircuitEngine
from .noise import NoiseSimulator

class CollaborationMode(Enum):
    SECURE_COMPUTE = "secure_compute"
    STATE_SHARING = "state_sharing"
    DISTRIBUTED_SENSING = "distributed_sensing"
    BLIND_COMPUTE = "blind_compute"

@dataclass
class CollaborationConfig:
    """Quantum collaboration configuration"""
    num_parties: int = 2
    security_level: str = "high"  # high, medium, low
    max_qubit_transfer: int = 100
    timeout: float = 3600  # seconds
    metadata: Dict = field(default_factory=dict)

class QuantumCollaborator:
    def __init__(self, config: Optional[CollaborationConfig] = None):
        self.config = config or CollaborationConfig()
        self.encryption = QuantumEncryptionEngine()
        self.teleporter = QuantumTeleporter()
        self.entanglement = QuantumEntanglementEngine()
        self.circuit = QuantumCircuitEngine()
        self.noise = NoiseSimulator()
        
        self.collaborators: Dict[str, Dict] = {}
        self.shared_states: Dict[str, np.ndarray] = {}
        self.session_history: List[Dict] = []
        
    def register_collaborator(self, collaborator_id: str, public_key: bytes) -> Dict:
        """Register new collaborator"""
        if collaborator_id in self.collaborators:
            raise ValueError(f"Collaborator {collaborator_id} already exists")
            
        self.collaborators[collaborator_id] = {
            "public_key": public_key,
            "registered_at": datetime.now(),
            "last_active": datetime.now(),
            "shared_resources": []
        }
        
        return {
            "success": True,
            "collaborator_id": collaborator_id,
            "timestamp": datetime.now()
        }
        
    def initiate_collaboration(self, 
                             collaborator_id: str,
                             mode: CollaborationMode,
                             params: Optional[Dict] = None) -> Dict:
        """Start collaborative quantum computation"""
        try:
            if collaborator_id not in self.collaborators:
                raise ValueError(f"Unknown collaborator: {collaborator_id}")
                
            # Create secure channel
            channel = self.encryption.secure_channel(
                self.collaborators[collaborator_id]["public_key"]
            )
            
            # Initialize based on mode
            if mode == CollaborationMode.SECURE_COMPUTE:
                result = self._setup_secure_compute(collaborator_id, channel, params)
            elif mode == CollaborationMode.STATE_SHARING:
                result = self._setup_state_sharing(collaborator_id, channel, params)
            elif mode == CollaborationMode.DISTRIBUTED_SENSING:
                result = self._setup_distributed_sensing(collaborator_id, channel, params)
            elif mode == CollaborationMode.BLIND_COMPUTE:
                result = self._setup_blind_compute(collaborator_id, channel, params)
            else:
                raise ValueError(f"Unknown collaboration mode: {mode}")
                
            # Record session
            session = {
                "collaborator_id": collaborator_id,
                "mode": mode.value,
                "started_at": datetime.now(),
                "status": "active",
                "params": params
            }
            self.session_history.append(session)
            
            return {
                "success": True,
                "session_id": len(self.session_history) - 1,
                "mode": mode.value,
                **result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def share_quantum_state(self, 
                          state: np.ndarray,
                          target_id: str,
                          secure: bool = True) -> Dict:
        """Share quantum state with collaborator"""
        try:
            if secure:
                # Use quantum teleportation
                result = self.teleporter.teleport_state(state, target_id)
            else:
                # Direct state transfer (for testing)
                self.shared_states[target_id] = state
                result = {
                    "success": True,
                    "fidelity": 1.0
                }
                
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def execute_collaborative_circuit(self,
                                   circuit_data: Dict,
                                   collaborators: List[str]) -> Dict:
        """Execute quantum circuit collaboratively"""
        try:
            # Verify collaborators
            for collab_id in collaborators:
                if collab_id not in self.collaborators:
                    raise ValueError(f"Unknown collaborator: {collab_id}")
                    
            # Create entangled resource states
            entanglement_result = self.entanglement.create_entanglement(
                [(0, i+1) for i in range(len(collaborators))],
                EntanglementType.GHZ
            )
            
            if not entanglement_result["success"]:
                raise ValueError("Failed to create entangled states")
                
            # Execute circuit
            self.circuit.initialize_circuit()
            # Add circuit operations from circuit_data
            result = self.circuit.execute_circuit()
            
            return {
                "success": True,
                "results": result,
                "collaborators": collaborators
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def render_collaboration_interface(self):
        """Render Streamlit collaboration interface"""
        st.markdown("### 🤝 Quantum Collaboration")
        
        # Collaborator management
        st.markdown("#### Collaborators")
        for collab_id, info in self.collaborators.items():
            st.markdown(f"""
            - **{collab_id}**
              - Registered: {info['registered_at']}
              - Last Active: {info['last_active']}
            """)
            
        # New collaboration
        st.markdown("#### New Collaboration")
        collab_id = st.text_input("Collaborator ID")
        mode = st.selectbox(
            "Collaboration Mode",
            [m.value for m in CollaborationMode]
        )
        
        if st.button("Start Collaboration"):
            result = self.initiate_collaboration(
                collab_id,
                CollaborationMode(mode)
            )
            
            if result["success"]:
                st.success(f"Collaboration started! Session ID: {result['session_id']}")
            else:
                st.error(f"Failed to start collaboration: {result.get('error')}")
                
        # Session history
        if self.session_history:
            st.markdown("#### Session History")
            for i, session in enumerate(self.session_history):
                st.markdown(f"""
                Session {i}:
                - Collaborator: {session['collaborator_id']}
                - Mode: {session['mode']}
                - Started: {session['started_at']}
                - Status: {session['status']}
                """)
                
    def _setup_secure_compute(self, collaborator_id: str, channel, params: Dict) -> Dict:
        """Setup secure multi-party computation"""
        # Initialize quantum resources
        self.circuit.initialize_circuit()
        
        # Create entangled states for computation
        self.entanglement.create_entanglement(
            [(0, 1)],
            EntanglementType.BELL_PAIR
        )
        
        return {
            "channel_id": channel.channel_id,
            "resources_allocated": True
        }
        
    def _setup_state_sharing(self, collaborator_id: str, channel, params: Dict) -> Dict:
        """Setup quantum state sharing"""
        # Prepare teleportation resources
        self.teleporter = QuantumTeleporter()
        
        return {
            "channel_id": channel.channel_id,
            "max_state_size": self.config.max_qubit_transfer
        }
        
    def _setup_distributed_sensing(self, collaborator_id: str, channel, params: Dict) -> Dict:
        """Setup distributed quantum sensing"""
        # Initialize GHZ states for sensing
        self.entanglement.create_entanglement(
            [(i, i+1) for i in range(self.config.num_parties-1)],
            EntanglementType.GHZ
        )
        
        return {
            "channel_id": channel.channel_id,
            "sensor_ids": list(range(self.config.num_parties))
        }
        
    def _setup_blind_compute(self, collaborator_id: str, channel, params: Dict) -> Dict:
        """Setup blind quantum computation"""
        # Create trap qubits and dummy operations
        self.circuit.initialize_circuit()
        
        return {
            "channel_id": channel.channel_id,
            "computation_id": hex(hash(datetime.now().isoformat()))[:16]
        } 