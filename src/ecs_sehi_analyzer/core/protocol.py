import streamlit as st
import numpy as np
from typing import Optional
from sklearn.decomposition import PCA
import plotly.graph_objects as go

class ProtocolManager:
    def __init__(self):
        self.protocol_version = "1.0"
        self.active_protocols = []

    def initialize_protocol(self):
        return True

    def render_feature_space(self, data: Optional[np.ndarray] = None):
        """Render feature space visualization"""
        if data is None:
            # Generate sample data
            n_samples = 100
            n_features = 10
            data = np.random.randn(n_samples, n_features)
        
        # Ensure data has enough samples and features
        if data.ndim > 2:
            # Reshape to 2D: (samples, features)
            data = data.reshape(-1, data.shape[-1])
        
        # Only perform PCA if we have enough data
        if data.shape[0] > 1 and data.shape[1] > 1:
            try:
                # Perform PCA
                pca = PCA(n_components=min(2, min(data.shape[0], data.shape[1])))
                features = pca.fit_transform(data)
                
                # Create scatter plot
                fig = go.Figure(data=go.Scatter(
                    x=features[:, 0],
                    y=features[:, 1] if features.shape[1] > 1 else np.zeros_like(features[:, 0]),
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=np.arange(len(features)),
                        colorscale='Viridis',
                        showscale=True
                    )
                ))
                
                fig.update_layout(
                    title="Feature Space Analysis",
                    xaxis_title="Principal Component 1",
                    yaxis_title="Principal Component 2",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not perform PCA: {str(e)}")
                st.info("Try with data containing more samples or features")
        else:
            st.warning("Not enough data for feature space visualization")
            st.info("Need at least 2 samples with 2 features each") 