import torch
import plotly.express as px
import streamlit as st

class PyTorchVisualizer:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def plot_feature_maps(self, tensor_data):
        """Visualize PyTorch tensors with GPU-MPS optimization"""
        # Move tensor to CPU if it's on MPS
        if isinstance(tensor_data, torch.Tensor):
            if tensor_data.device.type == 'mps':
                tensor_data = tensor_data.cpu()
            tensor_data = tensor_data.numpy()
        
        fig = px.imshow(
            tensor_data[0,0,:,:], 
            color_continuous_scale='viridis',
            title="Feature Map Visualization"
        )
        st.plotly_chart(fig)

    def plot_defect_analysis(self, data, predictions):
        """Plot defect analysis results"""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        # Create visualization logic here
        fig = px.scatter(
            data_frame=data,
            x='x',
            y='y',
            color=predictions,
            title="Defect Analysis Results"
        )
        st.plotly_chart(fig)
