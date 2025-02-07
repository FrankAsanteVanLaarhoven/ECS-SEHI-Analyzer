from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt
import numpy as np

class QuantumVisualizer:
    def visualize_state(self, state: np.ndarray) -> plt.Figure:
        fig = plot_bloch_multivector(state)
        fig.set_size_inches(8, 8)
        fig.tight_layout()
        return fig

    def create_entanglement_network(self, circuit):
        return plot_circuit(circuit, output='mpl')
