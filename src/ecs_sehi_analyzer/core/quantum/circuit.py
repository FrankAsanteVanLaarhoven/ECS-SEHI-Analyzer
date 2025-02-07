from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from enum import Enum
import streamlit as st
import plotly.graph_objects as go
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

class GateType(Enum):
    HADAMARD = "h"
    CNOT = "cx"
    PHASE = "p"
    ROTATION = "r"
    MEASUREMENT = "measure"

@dataclass
class CircuitConfig:
    """Quantum circuit configuration"""
    num_qubits: int = 4
    num_classical_bits: int = 4
    max_depth: int = 100
    optimization_level: int = 2
    shots: int = 1000
    backend: str = "qasm_simulator"
    noise_model: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)

class QuantumCircuitEngine:
    def __init__(self, config: Optional[CircuitConfig] = None):
        self.config = config or CircuitConfig()
        self.circuit: Optional[QuantumCircuit] = None
        self.measurement_results: List[Dict] = []
        self.backend = Aer.get_backend(self.config.backend)
        
    def initialize_circuit(self):
        """Initialize quantum circuit"""
        qr = QuantumRegister(self.config.num_qubits, 'q')
        cr = ClassicalRegister(self.config.num_classical_bits, 'c')
        self.circuit = QuantumCircuit(qr, cr)
        
    def add_gate(self, gate_type: GateType, qubits: List[int], 
                 params: Optional[List[float]] = None):
        """Add quantum gate to circuit"""
        if not self.circuit:
            self.initialize_circuit()
            
        if gate_type == GateType.HADAMARD:
            for qubit in qubits:
                self.circuit.h(qubit)
                
        elif gate_type == GateType.CNOT:
            if len(qubits) != 2:
                raise ValueError("CNOT gate requires 2 qubits")
            self.circuit.cx(qubits[0], qubits[1])
            
        elif gate_type == GateType.PHASE:
            if not params or len(params) != 1:
                raise ValueError("Phase gate requires 1 parameter")
            for qubit in qubits:
                self.circuit.p(params[0], qubit)
                
        elif gate_type == GateType.ROTATION:
            if not params or len(params) != 3:
                raise ValueError("Rotation gate requires 3 parameters")
            for qubit in qubits:
                self.circuit.r(params[0], params[1], params[2], qubit)
                
        elif gate_type == GateType.MEASUREMENT:
            for i, qubit in enumerate(qubits):
                self.circuit.measure(qubit, i)
                
    def execute_circuit(self) -> Dict:
        """Execute quantum circuit"""
        if not self.circuit:
            raise ValueError("Circuit not initialized")
            
        # Add measurements if not present
        if not any(op.name == 'measure' for op in self.circuit.data):
            self.add_gate(GateType.MEASUREMENT, list(range(self.config.num_qubits)))
            
        # Execute circuit
        job = execute(
            self.circuit,
            self.backend,
            shots=self.config.shots,
            optimization_level=self.config.optimization_level,
            noise_model=self.config.noise_model
        )
        
        result = job.result()
        counts = result.get_counts(self.circuit)
        
        # Record results
        measurement_result = {
            "counts": counts,
            "success_rate": self._calculate_success_rate(counts),
            "quantum_state": self._get_quantum_state(),
            "circuit_depth": self.circuit.depth()
        }
        
        self.measurement_results.append(measurement_result)
        return measurement_result
        
    def render_circuit_interface(self):
        """Render Streamlit circuit interface"""
        st.markdown("### ⚛️ Quantum Circuit")
        
        # Circuit visualization
        if self.circuit:
            st.markdown("#### Circuit Diagram")
            st.code(str(self.circuit))
            
            # Measurement results
            if self.measurement_results:
                latest_result = self.measurement_results[-1]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Success Rate",
                        f"{latest_result['success_rate']:.2%}"
                    )
                with col2:
                    st.metric(
                        "Circuit Depth",
                        latest_result['circuit_depth']
                    )
                    
                # Measurement distribution
                st.markdown("#### Measurement Distribution")
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(latest_result['counts'].keys()),
                        y=list(latest_result['counts'].values())
                    )
                ])
                
                fig.update_layout(
                    title="Measurement Outcomes",
                    xaxis_title="State",
                    yaxis_title="Count"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
    def _calculate_success_rate(self, counts: Dict[str, int]) -> float:
        """Calculate circuit success rate"""
        total_shots = sum(counts.values())
        # Consider the most frequent outcome as "success"
        max_count = max(counts.values())
        return max_count / total_shots if total_shots > 0 else 0.0
        
    def _get_quantum_state(self) -> np.ndarray:
        """Get quantum state vector"""
        statevector_backend = Aer.get_backend('statevector_simulator')
        statevector = execute(self.circuit, statevector_backend).result().get_statevector()
        return np.array(statevector)
        
    def get_circuit_statistics(self) -> Dict:
        """Get circuit statistics"""
        if not self.circuit:
            return {}
            
        return {
            "num_qubits": self.circuit.num_qubits,
            "depth": self.circuit.depth(),
            "size": len(self.circuit.data),
            "num_measurements": sum(1 for op in self.circuit.data if op.name == 'measure'),
            "success_rates": [r["success_rate"] for r in self.measurement_results]
        }
        
    def reset_circuit(self):
        """Reset quantum circuit"""
        self.initialize_circuit()
        self.measurement_results = [] 