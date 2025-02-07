from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from enum import Enum
import streamlit as st
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from .circuit import QuantumCircuitEngine, GateType

class StateType(Enum):
    COMPUTATIONAL = "computational"
    BELL = "bell"
    GHZ = "ghz"
    W = "w"
    CLUSTER = "cluster"
    CUSTOM = "custom"

@dataclass
class StateConfig:
    """Quantum state preparation configuration"""
    num_qubits: int = 2
    fidelity_threshold: float = 0.99
    optimization_steps: int = 100
    metadata: Dict = field(default_factory=dict)

class QuantumStatePreparator:
    def __init__(self, config: Optional[StateConfig] = None):
        self.config = config or StateConfig()
        self.circuit = QuantumCircuitEngine()
        self.target_state: Optional[np.ndarray] = None
        self.prepared_state: Optional[np.ndarray] = None
        self.preparation_fidelity: float = 0.0
        
    def prepare_state(self, 
                     state_type: StateType,
                     params: Optional[Dict] = None) -> Dict:
        """Prepare quantum state"""
        try:
            if state_type == StateType.COMPUTATIONAL:
                state = self._prepare_computational(params)
            elif state_type == StateType.BELL:
                state = self._prepare_bell(params)
            elif state_type == StateType.GHZ:
                state = self._prepare_ghz(params)
            elif state_type == StateType.W:
                state = self._prepare_w(params)
            elif state_type == StateType.CLUSTER:
                state = self._prepare_cluster(params)
            elif state_type == StateType.CUSTOM:
                if not params or "state_vector" not in params:
                    raise ValueError("Custom state requires state vector parameter")
                state = self._prepare_custom(params["state_vector"])
            else:
                raise ValueError(f"Unknown state type: {state_type}")
                
            return {
                "success": True,
                "state": state,
                "fidelity": self._calculate_fidelity(state)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def _prepare_computational(self, params: Optional[Dict]) -> np.ndarray:
        """Prepare computational basis state"""
        basis_state = params.get("basis_state", 0)
        self.circuit.initialize_circuit()
        
        # Convert to binary and apply X gates
        binary = format(basis_state, f'0{self.config.num_qubits}b')
        for i, bit in enumerate(binary):
            if bit == '1':
                self.circuit.add_gate(GateType.PAULI_X, [i])
                
        result = self.circuit.execute_circuit()
        return result["statevector"]
        
    def _prepare_bell(self, params: Optional[Dict]) -> np.ndarray:
        """Prepare Bell state"""
        self.circuit.initialize_circuit()
        
        # Apply Hadamard and CNOT
        self.circuit.add_gate(GateType.HADAMARD, [0])
        self.circuit.add_gate(GateType.CNOT, [0, 1])
        
        result = self.circuit.execute_circuit()
        return result["statevector"]
        
    def _prepare_ghz(self, params: Optional[Dict]) -> np.ndarray:
        """Prepare GHZ state"""
        self.circuit.initialize_circuit()
        
        # Apply Hadamard and CNOT
        self.circuit.add_gate(GateType.HADAMARD, [0])
        for i in range(1, self.config.num_qubits):
            self.circuit.add_gate(GateType.CNOT, [0, i])
            
        result = self.circuit.execute_circuit()
        return result["statevector"]
        
    def _prepare_w(self, params: Optional[Dict]) -> np.ndarray:
        """Prepare W state"""
        self.circuit.initialize_circuit()
        
        # Apply rotations and CNOTs
        self.circuit.add_gate(
            GateType.ROTATION,
            [0],
            [np.arccos(np.sqrt(1/self.config.num_qubits)), 0, 0]
        )
        
        for i in range(1, self.config.num_qubits):
            angle = np.arccos(np.sqrt(1/(self.config.num_qubits-i)))
            self.circuit.add_gate(GateType.CNOT, [i-1, i])
            self.circuit.add_gate(
                GateType.ROTATION,
                [i],
                [angle, 0, 0]
            )
            
        result = self.circuit.execute_circuit()
        return result["statevector"]
        
    def _prepare_cluster(self, params: Optional[Dict]) -> np.ndarray:
        """Prepare cluster state"""
        self.circuit.initialize_circuit()
        
        # Apply Hadamard to all qubits
        for i in range(self.config.num_qubits):
            self.circuit.add_gate(GateType.HADAMARD, [i])
            
        # Apply CZ gates between neighbors
        for i in range(self.config.num_qubits-1):
            self.circuit.add_gate(GateType.PHASE, [i], [np.pi])
            self.circuit.add_gate(GateType.CNOT, [i, i+1])
            self.circuit.add_gate(GateType.PHASE, [i], [np.pi])
            
        result = self.circuit.execute_circuit()
        return result["statevector"]
        
    def _prepare_custom(self, state_vector: np.ndarray) -> np.ndarray:
        """Prepare custom state"""
        self.target_state = state_vector / np.linalg.norm(state_vector)
        
        # Convert to Qiskit Statevector
        target_statevector = Statevector(self.target_state)
        
        # Initialize circuit to this state
        self.circuit.circuit.initialize(
            target_statevector.data,
            self.circuit.circuit.qubits
        )
        
        result = self.circuit.execute_circuit()
        return result["statevector"]
        
    def _calculate_fidelity(self, state: np.ndarray) -> float:
        """Calculate state fidelity"""
        if self.target_state is None or state is None:
            return 0.0
            
        overlap = np.abs(np.vdot(self.target_state, state))
        return overlap**2
        
    def render_state_interface(self):
        """Render Streamlit state preparation interface"""
        st.markdown("### 🔮 Quantum State Preparation")
        
        # State selection
        state_type = st.selectbox(
            "Select State Type",
            [s.value for s in StateType]
        )
        
        # State parameters
        params = {}
        if state_type == StateType.CUSTOM:
            st.markdown("#### Custom State Parameters")
            state_vector_str = st.text_area(
                "State Vector (comma-separated complex numbers)",
                "1,0,0,0"
            )
            try:
                state_vector = np.array(
                    [complex(x) for x in state_vector_str.split(",")]
                )
                params["state_vector"] = state_vector
            except ValueError:
                st.error("Invalid state vector format")
                return
                
        # Prepare state button
        if st.button("Prepare State"):
            result = self.prepare_state(StateType(state_type), params)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Fidelity",
                    f"{result['fidelity']:.4f}"
                )
            with col2:
                st.metric(
                    "Circuit Depth",
                    self.circuit.circuit.depth()
                )
                
            # State visualization
            if result['state'] is not None:
                st.markdown("#### Prepared State")
                self._render_state_visualization(result['state'])
                
    def _render_state_visualization(self, state: np.ndarray):
        """Render quantum state visualization"""
        import plotly.graph_objects as go
        
        # Create bar chart for state amplitudes
        amplitudes = np.abs(state)
        phases = np.angle(state)
        
        fig = go.Figure()
        
        # Amplitude bars
        fig.add_trace(go.Bar(
            name="Amplitude",
            x=[f"|{i:0{self.config.num_qubits}b}⟩" for i in range(len(amplitudes))],
            y=amplitudes,
            marker_color=phases,
            marker_colorscale="HSL",
            marker_showscale=True,
            marker_colorbar_title="Phase"
        ))
        
        fig.update_layout(
            title="Quantum State Visualization",
            xaxis_title="Basis State",
            yaxis_title="Amplitude",
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True) 