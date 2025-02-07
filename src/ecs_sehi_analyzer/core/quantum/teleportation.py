from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from enum import Enum
import streamlit as st
from qiskit import QuantumCircuit
from .circuit import QuantumCircuitEngine, GateType
from .entanglement import QuantumEntanglementEngine, EntanglementType

class TeleportationState(Enum):
    READY = "ready"
    ENTANGLED = "entangled"
    MEASURED = "measured"
    CORRECTED = "corrected"
    ERROR = "error"

@dataclass
class TeleportationConfig:
    """Quantum teleportation configuration"""
    num_qubits: int = 3  # Minimum needed for teleportation
    fidelity_threshold: float = 0.95
    max_distance: float = 1000.0  # meters
    error_correction: bool = True
    metadata: Dict = field(default_factory=dict)

class QuantumTeleporter:
    def __init__(self, config: Optional[TeleportationConfig] = None):
        self.config = config or TeleportationConfig()
        self.circuit_engine = QuantumCircuitEngine()
        self.entanglement_engine = QuantumEntanglementEngine()
        self.state: TeleportationState = TeleportationState.READY
        self.teleportation_history: List[Dict] = []
        
    def teleport_state(self, state_vector: np.ndarray, target_id: str) -> Dict:
        """Teleport quantum state to target"""
        try:
            # Initialize circuit
            self.circuit_engine.initialize_circuit()
            
            # Create entangled pair between qubits 1 and 2
            success = self._create_entangled_pair()
            if not success:
                raise ValueError("Failed to create entangled pair")
                
            # Prepare input state on qubit 0
            self._prepare_input_state(state_vector)
            
            # Perform Bell measurement on qubits 0 and 1
            measurement_results = self._bell_measurement()
            
            # Apply corrections on target qubit based on measurements
            if self.config.error_correction:
                self._apply_corrections(measurement_results)
                
            # Verify teleported state
            final_state = self.circuit_engine._get_quantum_state()
            fidelity = self._calculate_fidelity(state_vector, final_state)
            
            # Record teleportation
            result = {
                "success": fidelity >= self.config.fidelity_threshold,
                "fidelity": fidelity,
                "target_id": target_id,
                "measurements": measurement_results,
                "state": self.state.value
            }
            
            self.teleportation_history.append(result)
            return result
            
        except Exception as e:
            self.state = TeleportationState.ERROR
            return {
                "success": False,
                "error": str(e),
                "state": self.state.value
            }
            
    def render_teleportation_interface(self):
        """Render Streamlit teleportation interface"""
        st.markdown("### 📡 Quantum Teleportation")
        
        # State preparation
        st.markdown("#### Input State")
        state_vector_str = st.text_area(
            "State Vector (comma-separated complex numbers)",
            "1,0"
        )
        
        try:
            state_vector = np.array(
                [complex(x) for x in state_vector_str.split(",")]
            )
            state_vector = state_vector / np.linalg.norm(state_vector)
        except ValueError:
            st.error("Invalid state vector format")
            return
            
        # Target selection
        target_id = st.text_input("Target ID", "target_1")
        
        # Teleport button
        if st.button("Teleport State"):
            result = self.teleport_state(state_vector, target_id)
            
            if result["success"]:
                st.success("State teleported successfully!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Fidelity",
                        f"{result['fidelity']:.4f}"
                    )
                with col2:
                    st.metric(
                        "State",
                        result['state']
                    )
                    
                # Show teleportation history
                if self.teleportation_history:
                    self._render_history()
            else:
                st.error(f"Teleportation failed: {result.get('error', 'Unknown error')}")
                
    def _create_entangled_pair(self) -> bool:
        """Create entangled pair for teleportation"""
        try:
            # Create Bell pair between qubits 1 and 2
            self.circuit_engine.add_gate(GateType.HADAMARD, [1])
            self.circuit_engine.add_gate(GateType.CNOT, [1, 2])
            
            self.state = TeleportationState.ENTANGLED
            return True
        except Exception:
            return False
            
    def _prepare_input_state(self, state_vector: np.ndarray):
        """Prepare input state on first qubit"""
        try:
            # Initialize qubit 0 to desired state
            self.circuit_engine.circuit.initialize(
                state_vector,
                [0]
            )
        except Exception as e:
            raise ValueError(f"Failed to prepare input state: {str(e)}")
            
    def _bell_measurement(self) -> Dict[str, int]:
        """Perform Bell measurement on first two qubits"""
        try:
            # Apply CNOT and H before measurement
            self.circuit_engine.add_gate(GateType.CNOT, [0, 1])
            self.circuit_engine.add_gate(GateType.HADAMARD, [0])
            
            # Measure qubits 0 and 1
            self.circuit_engine.add_gate(GateType.MEASUREMENT, [0, 1])
            
            result = self.circuit_engine.execute_circuit()
            self.state = TeleportationState.MEASURED
            
            # Parse measurement results
            counts = result["counts"]
            most_frequent = max(counts.items(), key=lambda x: x[1])[0]
            return {
                "q0": int(most_frequent[0]),
                "q1": int(most_frequent[1])
            }
            
        except Exception as e:
            raise ValueError(f"Bell measurement failed: {str(e)}")
            
    def _apply_corrections(self, measurements: Dict[str, int]):
        """Apply correction operations based on measurements"""
        try:
            # Apply X gate if q1 measurement is 1
            if measurements["q1"]:
                self.circuit_engine.add_gate(GateType.PHASE, [2], [np.pi])
                
            # Apply Z gate if q0 measurement is 1
            if measurements["q0"]:
                self.circuit_engine.add_gate(GateType.HADAMARD, [2])
                self.circuit_engine.add_gate(GateType.PHASE, [2], [np.pi])
                self.circuit_engine.add_gate(GateType.HADAMARD, [2])
                
            self.state = TeleportationState.CORRECTED
            
        except Exception as e:
            raise ValueError(f"Correction operations failed: {str(e)}")
            
    def _calculate_fidelity(self, target_state: np.ndarray, final_state: np.ndarray) -> float:
        """Calculate fidelity between target and teleported states"""
        overlap = np.abs(np.vdot(target_state, final_state))
        return overlap**2
        
    def _render_history(self):
        """Render teleportation history visualization"""
        import plotly.graph_objects as go
        
        # Create fidelity history plot
        fidelities = [r["fidelity"] for r in self.teleportation_history]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=fidelities,
            mode='lines+markers',
            name='Fidelity'
        ))
        
        fig.update_layout(
            title="Teleportation Fidelity History",
            xaxis_title="Attempt",
            yaxis_title="Fidelity",
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True) 