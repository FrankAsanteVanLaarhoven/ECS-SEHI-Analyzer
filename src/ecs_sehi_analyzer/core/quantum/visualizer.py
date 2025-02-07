from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from enum import Enum
import streamlit as st
import plotly.graph_objects as go
from .circuit import QuantumCircuitEngine
from .state_prep import QuantumStatePreparator

class VisualizationType(Enum):
    BLOCH_SPHERE = "bloch_sphere"
    DENSITY_MATRIX = "density_matrix"
    CIRCUIT_DIAGRAM = "circuit_diagram"
    STATE_VECTOR = "state_vector"
    PROCESS_MATRIX = "process_matrix"

@dataclass
class VisualizerConfig:
    """Quantum visualization configuration"""
    resolution: int = 100
    colormap: str = "viridis"
    animation_frames: int = 50
    interactive: bool = True
    render_quality: str = "high"
    metadata: Dict = field(default_factory=dict)

class QuantumVisualizer:
    def __init__(self, config: Optional[VisualizerConfig] = None):
        self.config = config or VisualizerConfig()
        self.circuit = QuantumCircuitEngine()
        self.state_prep = QuantumStatePreparator()
        
        self.current_state: Optional[np.ndarray] = None
        self.visualization_history: List[Dict] = []
        
    def visualize_state(self, 
                       state: np.ndarray,
                       vis_type: VisualizationType,
                       params: Optional[Dict] = None) -> Dict:
        """Visualize quantum state"""
        try:
            self.current_state = state
            
            if vis_type == VisualizationType.BLOCH_SPHERE:
                fig = self._create_bloch_sphere(state, params)
            elif vis_type == VisualizationType.DENSITY_MATRIX:
                fig = self._create_density_matrix(state, params)
            elif vis_type == VisualizationType.STATE_VECTOR:
                fig = self._create_state_vector(state, params)
            else:
                raise ValueError(f"Unsupported visualization type: {vis_type}")
                
            # Record visualization
            record = {
                "type": vis_type.value,
                "state_dim": len(state),
                "params": params
            }
            self.visualization_history.append(record)
            
            return {
                "success": True,
                "figure": fig,
                "metadata": record
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def visualize_process(self,
                         process_data: Dict,
                         vis_type: VisualizationType,
                         params: Optional[Dict] = None) -> Dict:
        """Visualize quantum process"""
        try:
            if vis_type == VisualizationType.CIRCUIT_DIAGRAM:
                fig = self._create_circuit_diagram(process_data, params)
            elif vis_type == VisualizationType.PROCESS_MATRIX:
                fig = self._create_process_matrix(process_data, params)
            else:
                raise ValueError(f"Unsupported process visualization: {vis_type}")
                
            return {
                "success": True,
                "figure": fig
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def render_visualization_interface(self):
        """Render Streamlit visualization interface"""
        st.markdown("### 🎨 Quantum Visualization")
        
        # Visualization type selection
        vis_type = st.selectbox(
            "Visualization Type",
            [t.value for t in VisualizationType]
        )
        
        # State input for state visualizations
        if vis_type in [t.value for t in [
            VisualizationType.BLOCH_SPHERE,
            VisualizationType.DENSITY_MATRIX,
            VisualizationType.STATE_VECTOR
        ]]:
            st.markdown("#### Quantum State")
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
                
            if st.button("Visualize State"):
                result = self.visualize_state(
                    state,
                    VisualizationType(vis_type)
                )
                
                if result["success"]:
                    st.plotly_chart(result["figure"], use_container_width=True)
                else:
                    st.error(f"Visualization failed: {result.get('error')}")
                    
        # Process visualization
        else:
            st.markdown("#### Quantum Process")
            if st.button("Visualize Process"):
                result = self.visualize_process(
                    {"type": "sample"},
                    VisualizationType(vis_type)
                )
                
                if result["success"]:
                    st.plotly_chart(result["figure"], use_container_width=True)
                else:
                    st.error(f"Visualization failed: {result.get('error')}")
                    
    def _create_bloch_sphere(self, state: np.ndarray, params: Optional[Dict] = None) -> go.Figure:
        """Create Bloch sphere visualization"""
        if len(state) != 2:
            raise ValueError("Bloch sphere visualization requires a single qubit state")
            
        # Calculate Bloch sphere coordinates
        theta = 2 * np.arccos(np.abs(state[0]))
        phi = np.angle(state[1]) - np.angle(state[0])
        
        # Create sphere
        u = np.linspace(0, 2*np.pi, self.config.resolution)
        v = np.linspace(0, np.pi, self.config.resolution)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        
        fig = go.Figure()
        
        # Add sphere surface
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            opacity=0.3,
            showscale=False
        ))
        
        # Add state vector
        x_state = np.sin(theta) * np.cos(phi)
        y_state = np.sin(theta) * np.sin(phi)
        z_state = np.cos(theta)
        
        fig.add_trace(go.Scatter3d(
            x=[0, x_state],
            y=[0, y_state],
            z=[0, z_state],
            mode='lines+markers',
            line=dict(width=4),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title="Bloch Sphere Representation",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            )
        )
        
        return fig
        
    def _create_density_matrix(self, state: np.ndarray, params: Optional[Dict] = None) -> go.Figure:
        """Create density matrix visualization"""
        # Calculate density matrix
        density = np.outer(state, np.conj(state))
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=np.abs(density),
            text=[[f"{density[i,j]:.2f}" for j in range(len(state))]
                  for i in range(len(state))],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale=self.config.colormap
        ))
        
        fig.update_layout(
            title="Density Matrix",
            xaxis_title="Column Index",
            yaxis_title="Row Index"
        )
        
        return fig
        
    def _create_state_vector(self, state: np.ndarray, params: Optional[Dict] = None) -> go.Figure:
        """Create state vector visualization"""
        fig = go.Figure()
        
        # Add amplitude bars
        fig.add_trace(go.Bar(
            name="Amplitude",
            x=[f"|{i}⟩" for i in range(len(state))],
            y=np.abs(state),
            marker_color=np.angle(state),
            marker_colorscale="HSL",
            marker_showscale=True,
            marker_colorbar_title="Phase"
        ))
        
        fig.update_layout(
            title="Quantum State Vector",
            xaxis_title="Basis State",
            yaxis_title="Amplitude",
            yaxis_range=[0, 1]
        )
        
        return fig
        
    def _create_circuit_diagram(self, process_data: Dict, params: Optional[Dict] = None) -> go.Figure:
        """Create quantum circuit diagram"""
        # Get circuit from engine
        if not self.circuit.circuit:
            self.circuit.initialize_circuit()
            
        # Convert to diagram data
        circuit_data = str(self.circuit.circuit).split('\n')
        
        fig = go.Figure()
        
        # Add circuit elements
        for i, line in enumerate(circuit_data):
            fig.add_trace(go.Scatter(
                x=list(range(len(line))),
                y=[i] * len(line),
                mode='text',
                text=list(line),
                textfont=dict(size=14)
            ))
            
        fig.update_layout(
            title="Quantum Circuit Diagram",
            xaxis_title="Time Steps",
            yaxis_title="Qubits",
            yaxis_autorange="reversed"
        )
        
        return fig
        
    def _create_process_matrix(self, process_data: Dict, params: Optional[Dict] = None) -> go.Figure:
        """Create process matrix visualization"""
        # Generate sample process matrix
        dim = 4  # 2-qubit process
        process_matrix = np.random.normal(0, 1, (dim, dim)) + 1j * np.random.normal(0, 1, (dim, dim))
        process_matrix = process_matrix @ process_matrix.conj().T  # Make Hermitian
        
        fig = go.Figure(data=go.Heatmap(
            z=np.abs(process_matrix),
            text=[[f"{process_matrix[i,j]:.2f}" for j in range(dim)]
                  for i in range(dim)],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale=self.config.colormap
        ))
        
        fig.update_layout(
            title="Quantum Process Matrix",
            xaxis_title="Output State",
            yaxis_title="Input State"
        )
        
        return fig 