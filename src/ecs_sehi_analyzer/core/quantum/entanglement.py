from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from datetime import datetime
import streamlit as st
from ..quantum.encryption import QuantumEncryptionEngine
from enum import Enum
from qiskit import QuantumCircuit
from .circuit import QuantumCircuitEngine, GateType
from .noise import NoiseSimulator, NoiseType

class SyncState(Enum):
    READY = "ready"
    SYNCING = "syncing"
    ENTANGLED = "entangled"
    ERROR = "error"

class EntanglementType(Enum):
    BELL_PAIR = "bell_pair"
    GHZ = "ghz"
    CLUSTER = "cluster"
    CUSTOM = "custom"

@dataclass
class EntanglementNode:
    """Quantum entanglement node"""
    node_id: str
    public_key: bytes
    sync_state: SyncState = SyncState.READY
    last_sync: Optional[datetime] = None
    peers: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

@dataclass
class EntanglementConfig:
    """Quantum entanglement configuration"""
    num_qubits: int = 4
    fidelity_threshold: float = 0.95
    max_distance: float = 1000.0  # meters
    noise_level: float = 0.01
    metadata: Dict = field(default_factory=dict)

class QuantumEntanglementEngine:
    def __init__(self, config: Optional[EntanglementConfig] = None):
        self.encryption = QuantumEncryptionEngine()
        self.nodes: Dict[str, EntanglementNode] = {}
        self.sync_channels: Dict[str, Dict] = {}
        self.entanglement_pairs: List[Tuple[str, str]] = []
        self.config = config or EntanglementConfig()
        self.circuit_engine = QuantumCircuitEngine()
        self.noise_simulator = NoiseSimulator()
        self.entangled_pairs: List[Tuple[int, int]] = []
        self.fidelity_history: List[float] = []
        
    def create_node(self, node_id: str) -> EntanglementNode:
        """Create new entanglement node"""
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists")
            
        # Generate quantum-safe keys
        public_key = self.encryption.public_key
        
        node = EntanglementNode(
            node_id=node_id,
            public_key=public_key
        )
        
        self.nodes[node_id] = node
        return node
        
    def establish_entanglement(self, node_a: str, node_b: str) -> bool:
        """Establish quantum entanglement between nodes"""
        if node_a not in self.nodes or node_b not in self.nodes:
            raise ValueError("Invalid node IDs")
            
        # Create secure sync channel
        channel_id = f"{node_a}_{node_b}"
        self.sync_channels[channel_id] = {
            "created_at": datetime.now(),
            "last_sync": None,
            "state": SyncState.READY
        }
        
        # Update node states
        self.nodes[node_a].peers.append(node_b)
        self.nodes[node_b].peers.append(node_a)
        self.nodes[node_a].sync_state = SyncState.ENTANGLED
        self.nodes[node_b].sync_state = SyncState.ENTANGLED
        
        # Record entanglement pair
        self.entanglement_pairs.append((node_a, node_b))
        
        return True
        
    def sync_data(self, node_a: str, node_b: str, data: Dict) -> Dict:
        """Synchronize data between entangled nodes"""
        channel_id = f"{node_a}_{node_b}"
        if channel_id not in self.sync_channels:
            raise ValueError("No entanglement channel exists")
            
        # Update sync states
        self.nodes[node_a].sync_state = SyncState.SYNCING
        self.nodes[node_b].sync_state = SyncState.SYNCING
        
        try:
            # Encrypt data for sync
            encrypted_data = self.encryption.encrypt_data(
                str(data).encode(),
                channel_id
            )
            
            # Simulate quantum sync
            synced_data = self._quantum_sync(encrypted_data)
            
            # Update sync timestamp
            now = datetime.now()
            self.sync_channels[channel_id]["last_sync"] = now
            self.nodes[node_a].last_sync = now
            self.nodes[node_b].last_sync = now
            
            # Reset sync states
            self.nodes[node_a].sync_state = SyncState.ENTANGLED
            self.nodes[node_b].sync_state = SyncState.ENTANGLED
            
            return {"status": "success", "synced_at": now}
            
        except Exception as e:
            self.nodes[node_a].sync_state = SyncState.ERROR
            self.nodes[node_b].sync_state = SyncState.ERROR
            return {"status": "error", "message": str(e)}
            
    def render_entanglement_interface(self):
        """Render Streamlit interface for quantum entanglement"""
        st.markdown("### 🔄 Quantum Entanglement")
        
        # Node management
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Create Node")
            node_id = st.text_input("Node ID")
            if st.button("Create Node"):
                try:
                    node = self.create_node(node_id)
                    st.success(f"Node {node_id} created")
                except ValueError as e:
                    st.error(str(e))
                    
        with col2:
            st.markdown("#### Establish Entanglement")
            if len(self.nodes) >= 2:
                node_a = st.selectbox("Node A", list(self.nodes.keys()))
                node_b = st.selectbox("Node B", 
                                    [n for n in self.nodes.keys() if n != node_a])
                
                if st.button("Entangle Nodes"):
                    try:
                        self.establish_entanglement(node_a, node_b)
                        st.success("Nodes entangled successfully")
                    except Exception as e:
                        st.error(f"Entanglement failed: {str(e)}")
                        
        # Node status
        st.markdown("#### Node Status")
        for node_id, node in self.nodes.items():
            with st.expander(f"Node {node_id}"):
                st.markdown(f"**Status:** {node.sync_state.value}")
                st.markdown(f"**Peers:** {', '.join(node.peers)}")
                if node.last_sync:
                    st.markdown(f"**Last Sync:** {node.last_sync}")
                    
    def _quantum_sync(self, data: bytes) -> bytes:
        """Simulate quantum synchronization"""
        # Add quantum noise
        noise = np.random.normal(0, 0.1, len(data))
        noisy_data = np.frombuffer(data, dtype=np.uint8) + noise
        
        # Quantum error correction
        corrected_data = np.clip(noisy_data, 0, 255).astype(np.uint8)
        
        return bytes(corrected_data)

    def create_entanglement(self, 
                          qubit_pairs: List[Tuple[int, int]],
                          entanglement_type: EntanglementType) -> Dict:
        """Create quantum entanglement between qubit pairs"""
        try:
            self.circuit_engine.initialize_circuit()
            
            if entanglement_type == EntanglementType.BELL_PAIR:
                success = self._create_bell_pairs(qubit_pairs)
            elif entanglement_type == EntanglementType.GHZ:
                success = self._create_ghz_state(qubit_pairs)
            elif entanglement_type == EntanglementType.CLUSTER:
                success = self._create_cluster_state(qubit_pairs)
            else:
                raise ValueError(f"Unknown entanglement type: {entanglement_type}")
                
            # Add noise effects
            noise_model = self.noise_simulator.create_noise_model(NoiseType.CUSTOM)
            self.circuit_engine.circuit.noise_model = noise_model
            
            # Execute circuit
            result = self.circuit_engine.execute_circuit()
            fidelity = self._calculate_entanglement_fidelity(result)
            self.fidelity_history.append(fidelity)
            
            return {
                "success": success and fidelity >= self.config.fidelity_threshold,
                "fidelity": fidelity,
                "pairs": qubit_pairs,
                "type": entanglement_type.value
            }
            
        except Exception as e:
            st.error(f"Entanglement creation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def secure_channel(self, target_id: str) -> "QuantumChannel":
        """Create secure quantum channel"""
        return QuantumChannel(
            source_id=id(self),
            target_id=target_id,
            noise_model=self.noise_simulator.noise_model
        )
        
    def _create_bell_pairs(self, pairs: List[Tuple[int, int]]) -> bool:
        """Create Bell pairs"""
        try:
            for q1, q2 in pairs:
                self.circuit_engine.add_gate(GateType.HADAMARD, [q1])
                self.circuit_engine.add_gate(GateType.CNOT, [q1, q2])
            return True
        except Exception:
            return False
            
    def _create_ghz_state(self, pairs: List[Tuple[int, int]]) -> bool:
        """Create GHZ state"""
        try:
            # First qubit in first pair is control
            control = pairs[0][0]
            self.circuit_engine.add_gate(GateType.HADAMARD, [control])
            
            # CNOT to all other qubits
            for pair in pairs:
                for target in pair:
                    if target != control:
                        self.circuit_engine.add_gate(GateType.CNOT, [control, target])
            return True
        except Exception:
            return False
            
    def _create_cluster_state(self, pairs: List[Tuple[int, int]]) -> bool:
        """Create cluster state"""
        try:
            # Hadamard on all qubits
            all_qubits = list(set([q for pair in pairs for q in pair]))
            for q in all_qubits:
                self.circuit_engine.add_gate(GateType.HADAMARD, [q])
                
            # CZ between pairs
            for q1, q2 in pairs:
                self.circuit_engine.add_gate(GateType.PHASE, [q1], [np.pi])
                self.circuit_engine.add_gate(GateType.CNOT, [q1, q2])
                self.circuit_engine.add_gate(GateType.PHASE, [q1], [np.pi])
            return True
        except Exception:
            return False
            
    def _calculate_entanglement_fidelity(self, result: Dict) -> float:
        """Calculate entanglement fidelity"""
        # Simplified fidelity calculation
        counts = result["counts"]
        total_shots = sum(counts.values())
        
        # For Bell pairs, expect |00⟩ and |11⟩ with equal probability
        expected_states = ["00", "11"]
        correct_counts = sum(counts.get(state, 0) for state in expected_states)
        
        return correct_counts / total_shots if total_shots > 0 else 0.0
        
    def _render_fidelity_history(self):
        """Render fidelity history visualization"""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=self.fidelity_history,
            mode='lines+markers',
            name='Fidelity'
        ))
        
        fig.update_layout(
            title="Entanglement Fidelity History",
            xaxis_title="Attempt",
            yaxis_title="Fidelity",
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True)

class QuantumChannel:
    def __init__(self, source_id: str, target_id: str, noise_model: Optional[NoiseModel] = None):
        self.source_id = source_id
        self.target_id = target_id
        self.noise_model = noise_model
        
    def send(self, data: Dict) -> Dict:
        """Send quantum data through channel"""
        # Apply noise model if present
        if self.noise_model:
            # Simulate noise effects
            pass
        return data
        
    def receive(self, data: Dict) -> Dict:
        """Receive quantum data from channel"""
        return data 