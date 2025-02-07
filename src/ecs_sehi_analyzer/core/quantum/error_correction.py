from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from enum import Enum
import streamlit as st
from datetime import datetime
from .circuit import QuantumCircuitEngine, GateType
from .noise import NoiseSimulator, NoiseType

class CorrectionCode(Enum):
    BIT_FLIP = "bit_flip"
    PHASE_FLIP = "phase_flip"
    SHOR = "shor"
    STEANE = "steane"
    SURFACE = "surface"

class ErrorType(Enum):
    BIT = "bit"
    PHASE = "phase"
    COMBINED = "combined"
    CUSTOM = "custom"

@dataclass
class CorrectionConfig:
    """Quantum error correction configuration"""
    code_distance: int = 3
    measurement_rounds: int = 5
    error_threshold: float = 0.01
    syndrome_method: str = "standard"  # standard, fault_tolerant
    metadata: Dict = field(default_factory=dict)

class QuantumErrorCorrector:
    def __init__(self, config: Optional[CorrectionConfig] = None):
        self.config = config or CorrectionConfig()
        self.circuit = QuantumCircuitEngine()
        self.noise = NoiseSimulator()
        
        self.correction_history: List[Dict] = []
        self.syndrome_data: Dict[str, np.ndarray] = {}
        self.logical_states: Dict[str, np.ndarray] = {}
        
    def encode_state(self,
                    state: np.ndarray,
                    code: CorrectionCode,
                    params: Optional[Dict] = None) -> Dict:
        """Encode quantum state using error correction code"""
        try:
            # Initialize encoding circuit
            self.circuit.initialize_circuit()
            
            if code == CorrectionCode.BIT_FLIP:
                result = self._encode_bit_flip(state)
            elif code == CorrectionCode.PHASE_FLIP:
                result = self._encode_phase_flip(state)
            elif code == CorrectionCode.SHOR:
                result = self._encode_shor(state)
            elif code == CorrectionCode.STEANE:
                result = self._encode_steane(state)
            elif code == CorrectionCode.SURFACE:
                result = self._encode_surface(state)
            else:
                raise ValueError(f"Unknown correction code: {code}")
                
            # Record encoding
            record = {
                "timestamp": datetime.now(),
                "code": code.value,
                "state_size": len(state),
                "result": result
            }
            self.correction_history.append(record)
            
            return {
                "success": True,
                "encoded_state": result["state"],
                "ancilla_qubits": result["ancilla"],
                "logical_mapping": result["mapping"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def detect_errors(self,
                     state: np.ndarray,
                     error_type: ErrorType = ErrorType.COMBINED) -> Dict:
        """Detect quantum errors in state"""
        try:
            # Perform syndrome measurements
            syndromes = []
            for _ in range(self.config.measurement_rounds):
                if error_type == ErrorType.BIT:
                    syndrome = self._measure_bit_syndrome(state)
                elif error_type == ErrorType.PHASE:
                    syndrome = self._measure_phase_syndrome(state)
                elif error_type == ErrorType.COMBINED:
                    syndrome = self._measure_combined_syndrome(state)
                else:
                    raise ValueError(f"Unknown error type: {error_type}")
                    
                syndromes.append(syndrome)
                
            # Analyze syndrome data
            syndrome_array = np.array(syndromes)
            error_locations = self._analyze_syndromes(syndrome_array)
            
            return {
                "success": True,
                "error_detected": len(error_locations) > 0,
                "error_locations": error_locations,
                "syndrome_data": syndrome_array
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def correct_errors(self,
                      state: np.ndarray,
                      error_data: Dict,
                      code: CorrectionCode) -> Dict:
        """Apply error correction"""
        try:
            if not error_data["error_detected"]:
                return {"success": True, "state": state, "corrections_applied": 0}
                
            # Apply corrections based on code
            if code == CorrectionCode.BIT_FLIP:
                result = self._correct_bit_flip(state, error_data)
            elif code == CorrectionCode.PHASE_FLIP:
                result = self._correct_phase_flip(state, error_data)
            elif code == CorrectionCode.SHOR:
                result = self._correct_shor(state, error_data)
            else:
                raise ValueError(f"Unsupported correction for code: {code}")
                
            return {
                "success": True,
                "corrected_state": result["state"],
                "corrections_applied": len(error_data["error_locations"]),
                "fidelity": result["fidelity"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def _encode_bit_flip(self, state: np.ndarray) -> Dict:
        """Encode state using bit-flip code"""
        # Create 3-qubit encoding
        self.circuit.add_gate(GateType.CNOT, [0, 1])
        self.circuit.add_gate(GateType.CNOT, [0, 2])
        
        result = self.circuit.execute_circuit()
        encoded_state = result["statevector"]
        
        return {
            "state": encoded_state,
            "ancilla": [1, 2],
            "mapping": {0: "logical_0", 1: "logical_1"}
        }
        
    def _encode_phase_flip(self, state: np.ndarray) -> Dict:
        """Encode state using phase-flip code"""
        # Apply Hadamard gates
        for i in range(3):
            self.circuit.add_gate(GateType.HADAMARD, [i])
            
        # Apply CNOT gates
        self.circuit.add_gate(GateType.CNOT, [0, 1])
        self.circuit.add_gate(GateType.CNOT, [0, 2])
        
        result = self.circuit.execute_circuit()
        encoded_state = result["statevector"]
        
        return {
            "state": encoded_state,
            "ancilla": [1, 2],
            "mapping": {0: "logical_0", 1: "logical_1"}
        }
        
    def _measure_bit_syndrome(self, state: np.ndarray) -> np.ndarray:
        """Measure bit-flip syndrome"""
        # Add ancilla qubits for syndrome measurement
        self.circuit.add_gate(GateType.CNOT, [0, 3])
        self.circuit.add_gate(GateType.CNOT, [1, 3])
        self.circuit.add_gate(GateType.CNOT, [1, 4])
        self.circuit.add_gate(GateType.CNOT, [2, 4])
        
        # Measure ancilla qubits
        self.circuit.add_gate(GateType.MEASUREMENT, [3, 4])
        
        result = self.circuit.execute_circuit()
        return np.array(list(result["counts"].keys())[0])
        
    def _analyze_syndromes(self, syndromes: np.ndarray) -> List[int]:
        """Analyze syndrome measurements"""
        # Majority vote on syndrome bits
        syndrome_votes = np.sum(syndromes, axis=0) > len(syndromes)/2
        
        # Identify error locations
        error_locations = []
        if syndrome_votes[0] and not syndrome_votes[1]:
            error_locations.append(0)
        elif syndrome_votes[0] and syndrome_votes[1]:
            error_locations.append(1)
        elif not syndrome_votes[0] and syndrome_votes[1]:
            error_locations.append(2)
            
        return error_locations
        
    def _correct_bit_flip(self, state: np.ndarray, error_data: Dict) -> Dict:
        """Apply bit-flip corrections"""
        corrected_state = state.copy()
        
        # Apply X gates at error locations
        for location in error_data["error_locations"]:
            self.circuit.add_gate(GateType.PAULI_X, [location])
            
        result = self.circuit.execute_circuit()
        corrected_state = result["statevector"]
        
        # Calculate fidelity with original logical state
        fidelity = np.abs(np.vdot(state[::3], corrected_state[::3]))**2
        
        return {
            "state": corrected_state,
            "fidelity": fidelity
        }
        
    def render_correction_interface(self):
        """Render Streamlit error correction interface"""
        st.markdown("### 🔧 Quantum Error Correction")
        
        # Code selection
        code = st.selectbox(
            "Error Correction Code",
            [c.value for c in CorrectionCode]
        )
        
        # Error type selection
        error_type = st.selectbox(
            "Error Type",
            [e.value for e in ErrorType]
        )
        
        # State preparation
        st.markdown("#### Input State")
        state_input = st.text_area(
            "State Vector (comma-separated complex numbers)",
            "1,0"
        )
        
        try:
            state = np.array([complex(x) for x in state_input.split(",")])
            state = state / np.linalg.norm(state)
        except ValueError:
            st.error("Invalid state vector format")
            return
            
        # Encode and detect errors
        if st.button("Encode and Detect Errors"):
            # Encode state
            encode_result = self.encode_state(
                state,
                CorrectionCode(code)
            )
            
            if encode_result["success"]:
                st.success("State encoded successfully")
                
                # Detect errors
                detect_result = self.detect_errors(
                    encode_result["encoded_state"],
                    ErrorType(error_type)
                )
                
                if detect_result["success"]:
                    if detect_result["error_detected"]:
                        st.warning(
                            f"Errors detected at qubits: {detect_result['error_locations']}"
                        )
                        
                        # Offer correction
                        if st.button("Apply Correction"):
                            correct_result = self.correct_errors(
                                encode_result["encoded_state"],
                                detect_result,
                                CorrectionCode(code)
                            )
                            
                            if correct_result["success"]:
                                st.success(f"""
                                Errors corrected!
                                - Fidelity: {correct_result['fidelity']:.3f}
                                - Corrections applied: {correct_result['corrections_applied']}
                                """)
                            else:
                                st.error(f"Correction failed: {correct_result['error']}")
                    else:
                        st.success("No errors detected")
                else:
                    st.error(f"Error detection failed: {detect_result['error']}")
            else:
                st.error(f"Encoding failed: {encode_result['error']}") 