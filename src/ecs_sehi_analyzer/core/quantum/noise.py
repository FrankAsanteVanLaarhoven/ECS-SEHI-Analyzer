from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from enum import Enum
import streamlit as st
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import (
    depolarizing_error,
    thermal_relaxation_error,
    readout_error
)

class NoiseType(Enum):
    DEPOLARIZING = "depolarizing"
    THERMAL = "thermal"
    READOUT = "readout"
    CUSTOM = "custom"

@dataclass
class NoiseConfig:
    """Quantum noise configuration"""
    error_probabilities: Dict[str, float] = field(default_factory=lambda: {
        "single_qubit": 0.001,
        "two_qubit": 0.01,
        "measurement": 0.05
    })
    t1: float = 50e-6  # T1 relaxation time (microseconds)
    t2: float = 70e-6  # T2 dephasing time (microseconds)
    temperature: float = 0.02  # Kelvin
    gate_times: Dict[str, float] = field(default_factory=lambda: {
        "single_qubit": 20e-9,  # 20 ns
        "two_qubit": 100e-9     # 100 ns
    })

class NoiseSimulator:
    def __init__(self, config: Optional[NoiseConfig] = None):
        self.config = config or NoiseConfig()
        self.noise_model = NoiseModel()
        self.error_rates: Dict[str, List[float]] = {
            "single_qubit": [],
            "two_qubit": [],
            "measurement": []
        }
        
    def create_noise_model(self, noise_type: NoiseType) -> NoiseModel:
        """Create quantum noise model"""
        self.noise_model = NoiseModel()
        
        if noise_type == NoiseType.DEPOLARIZING:
            self._add_depolarizing_noise()
        elif noise_type == NoiseType.THERMAL:
            self._add_thermal_noise()
        elif noise_type == NoiseType.READOUT:
            self._add_readout_noise()
        elif noise_type == NoiseType.CUSTOM:
            self._add_custom_noise()
            
        return self.noise_model
        
    def analyze_noise_impact(self, num_shots: int = 1000) -> Dict:
        """Analyze impact of noise on quantum system"""
        analysis = {
            "error_rates": self.error_rates,
            "coherence_time": self._estimate_coherence_time(),
            "gate_fidelities": self._calculate_gate_fidelities(),
            "error_channels": self._identify_error_channels()
        }
        
        return analysis
        
    def render_noise_interface(self):
        """Render Streamlit noise simulation interface"""
        st.markdown("### 🌊 Quantum Noise Simulation")
        
        # Noise type selection
        noise_type = st.selectbox(
            "Select Noise Type",
            [n.value for n in NoiseType]
        )
        
        # Noise parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Error Probabilities")
            single_qubit = st.slider(
                "Single Qubit Error",
                0.0, 0.1, self.config.error_probabilities["single_qubit"],
                format="%.4f"
            )
            two_qubit = st.slider(
                "Two Qubit Error",
                0.0, 0.2, self.config.error_probabilities["two_qubit"],
                format="%.4f"
            )
            
        with col2:
            st.markdown("#### Coherence Times")
            t1 = st.number_input(
                "T1 (µs)",
                0.0, 1000.0, self.config.t1 * 1e6,
                format="%.1f"
            )
            t2 = st.number_input(
                "T2 (µs)",
                0.0, 1000.0, self.config.t2 * 1e6,
                format="%.1f"
            )
            
        # Update configuration
        if st.button("Update Noise Model"):
            self.config.error_probabilities.update({
                "single_qubit": single_qubit,
                "two_qubit": two_qubit
            })
            self.config.t1 = t1 * 1e-6
            self.config.t2 = t2 * 1e-6
            
            # Create new noise model
            self.create_noise_model(NoiseType(noise_type))
            
            # Show analysis
            analysis = self.analyze_noise_impact()
            self._render_noise_analysis(analysis)
            
    def _add_depolarizing_noise(self):
        """Add depolarizing noise to model"""
        # Single qubit depolarizing error
        error_1q = depolarizing_error(
            self.config.error_probabilities["single_qubit"],
            1
        )
        self.noise_model.add_all_qubit_quantum_error(error_1q, ["u1", "u2", "u3"])
        
        # Two qubit depolarizing error
        error_2q = depolarizing_error(
            self.config.error_probabilities["two_qubit"],
            2
        )
        self.noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])
        
    def _add_thermal_noise(self):
        """Add thermal relaxation noise"""
        # Single qubit thermal error
        error_1q = thermal_relaxation_error(
            self.config.t1,
            self.config.t2,
            self.config.gate_times["single_qubit"]
        )
        self.noise_model.add_all_qubit_quantum_error(error_1q, ["u1", "u2", "u3"])
        
        # Two qubit thermal error
        error_2q = thermal_relaxation_error(
            self.config.t1,
            self.config.t2,
            self.config.gate_times["two_qubit"]
        )
        self.noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])
        
    def _add_readout_noise(self):
        """Add measurement readout noise"""
        # Readout error probabilities
        prob_0_1 = self.config.error_probabilities["measurement"]  # P(1|0)
        prob_1_0 = self.config.error_probabilities["measurement"]  # P(0|1)
        
        error = readout_error([[1 - prob_0_1, prob_0_1],
                             [prob_1_0, 1 - prob_1_0]])
        self.noise_model.add_all_qubit_readout_error(error)
        
    def _add_custom_noise(self):
        """Add custom noise model"""
        # Combine multiple noise sources
        self._add_depolarizing_noise()
        self._add_thermal_noise()
        self._add_readout_noise()
        
    def _estimate_coherence_time(self) -> float:
        """Estimate system coherence time"""
        return min(self.config.t1, self.config.t2)
        
    def _calculate_gate_fidelities(self) -> Dict[str, float]:
        """Calculate gate fidelities"""
        return {
            "single_qubit": 1 - self.config.error_probabilities["single_qubit"],
            "two_qubit": 1 - self.config.error_probabilities["two_qubit"],
            "measurement": 1 - self.config.error_probabilities["measurement"]
        }
        
    def _identify_error_channels(self) -> List[Dict]:
        """Identify dominant error channels"""
        channels = []
        
        # Check thermal relaxation
        if self.config.t1 < 100e-6:  # Less than 100µs
            channels.append({
                "type": "thermal_relaxation",
                "severity": "high",
                "timescale": self.config.t1
            })
            
        # Check gate errors
        if self.config.error_probabilities["two_qubit"] > 0.05:
            channels.append({
                "type": "two_qubit_gate",
                "severity": "high",
                "rate": self.config.error_probabilities["two_qubit"]
            })
            
        return channels
        
    def _render_noise_analysis(self, analysis: Dict):
        """Render noise analysis visualization"""
        import plotly.graph_objects as go
        
        # Gate fidelities chart
        fidelities = analysis["gate_fidelities"]
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(fidelities.keys()),
                y=list(fidelities.values()),
                text=[f"{v:.3%}" for v in fidelities.values()],
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            title="Gate Fidelities",
            xaxis_title="Gate Type",
            yaxis_title="Fidelity",
            yaxis_range=[0.8, 1.0]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Error channels
        st.markdown("#### Error Channels")
        for channel in analysis["error_channels"]:
            st.markdown(f"""
            - **{channel['type'].replace('_', ' ').title()}**
              - Severity: {channel['severity']}
              - {'Timescale' if 'timescale' in channel else 'Rate'}: 
                {channel.get('timescale', channel.get('rate'))}
            """) 