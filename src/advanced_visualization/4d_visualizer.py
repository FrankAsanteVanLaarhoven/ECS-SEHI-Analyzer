import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Optional, Tuple, Dict

class FourDVisualizer:
    def __init__(self, data: np.ndarray):
        self.data = data
        self._validate_data()
        self.analysis_results = {}
        
    def _validate_data(self):
        if self.data.ndim != 4:
            raise ValueError("4D data required (x,y,z,t)")
    
    def analyze_temporal_patterns(self) -> Dict[str, np.ndarray]:
        """Analyze temporal patterns in the data"""
        temporal_mean = np.mean(self.data, axis=-1)
        temporal_std = np.std(self.data, axis=-1)
        temporal_peaks = np.max(self.data, axis=-1)
        
        self.analysis_results.update({
            'temporal_mean': temporal_mean,
            'temporal_std': temporal_std,
            'temporal_peaks': temporal_peaks
        })
        return self.analysis_results
    
    def perform_dimensionality_reduction(self, method: str = 'tsne') -> np.ndarray:
        """Perform dimensionality reduction using specified method"""
        flattened = self.data.reshape(-1, self.data.shape[3])
        
        if method.lower() == 'tsne':
            embedded = TSNE(n_components=3,
                          perplexity=30,
                          n_iter=1000).fit_transform(flattened)
        elif method.lower() == 'pca':
            embedded = PCA(n_components=3).fit_transform(flattened)
        else:
            raise ValueError(f"Unsupported method: {method}")
            
        return embedded
            
    def render_4d_manifold(self, method: str = 'tsne') -> go.Figure:
        """Temporal 4D visualization using dimensionality reduction"""
        embedded = self.perform_dimensionality_reduction(method)
        
        fig = go.Figure(data=go.Scatter3d(
            x=embedded[:,0], y=embedded[:,1], z=embedded[:,2],
            mode='markers',
            marker=dict(
                size=4,
                color=self.data[...,0].flatten(),
                colorscale='Viridis',
                opacity=0.8
            ),
            customdata=np.stack([self.data[...,i].flatten() 
                               for i in range(self.data.shape[3])], -1),
            hovertemplate="<b>Time:</b> %{customdata[3]}<br>" +
                         "Energy: %{customdata[0]}<br>" +
                         "Intensity: %{customdata[1]}<br>" +
                         "Width: %{customdata[2]}<extra></extra>"
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title=f'{method.upper()}1',
                yaxis_title=f'{method.upper()}2',
                zaxis_title=f'{method.upper()}3',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            template="plotly_dark",
            height=800
        )
        return fig
    
    def render_temporal_analysis(self) -> go.Figure:
        """Visualize temporal analysis results"""
        if not self.analysis_results:
            self.analyze_temporal_patterns()
            
        fig = go.Figure()
        
        # Add temporal mean
        fig.add_trace(go.Heatmap(
            z=self.analysis_results['temporal_mean'],
            colorscale='Viridis',
            name='Temporal Mean'
        ))
        
        # Add temporal std as contours
        fig.add_trace(go.Contour(
            z=self.analysis_results['temporal_std'],
            colorscale='RdBu',
            name='Temporal STD',
            showscale=True
        ))
        
        fig.update_layout(
            title="Temporal Analysis",
            template="plotly_dark",
            height=600
        )
        return fig

class DataVisualizer4D:
    def __init__(self):
        self.frame_duration = 100  # ms between frames
        self.current_frame = 0
        self.playing = False
    
    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """Preprocess 4D data for visualization"""
        # Handle different input shapes
        if data.ndim == 4:
            if data.shape[2] == 1:  # Shape (H, W, 1, T)
                data = data.squeeze(axis=2)  # Convert to (H, W, T)
            else:  # Shape (H, W, C, T)
                data = np.mean(data, axis=2)  # Average over channels
        return data
    
    def create_animation(self, data: np.ndarray) -> go.Figure:
        """Create animated visualization of 4D data"""
        data = self.preprocess_data(data)
        
        # Create frames for animation
        frames = []
        for t in range(data.shape[-1]):
            frames.append(
                go.Frame(
                    data=[go.Heatmap(
                        z=data[..., t],
                        colorscale='Viridis',
                        showscale=True,
                        zmin=data.min(),
                        zmax=data.max()
                    )],
                    name=f'frame_{t}'
                )
            )
        
        # Create base figure
        fig = go.Figure(
            data=[go.Heatmap(z=data[..., 0], colorscale='Viridis')],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': '▶️ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': self.frame_duration, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate',
                        }]
                    },
                    {
                        'label': '⏸️ Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'currentvalue': {'prefix': 'Frame: '},
                'steps': [
                    {
                        'label': f'{i}',
                        'method': 'animate',
                        'args': [[f'frame_{i}'], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                    for i in range(data.shape[-1])
                ]
            }]
        )
        
        return fig
    
    def render_controls(self):
        """Render visualization controls"""
        st.sidebar.subheader("🎮 Visualization Controls")
        
        # Animation speed
        self.frame_duration = st.sidebar.slider(
            "Frame Duration (ms)",
            min_value=50,
            max_value=1000,
            value=100,
            step=50,
            key="viz_frame_duration"
        )
        
        # Color settings
        colormap = st.sidebar.selectbox(
            "Colormap",
            ["Viridis", "Plasma", "Inferno", "Magma"],
            key="viz_colormap"
        )
        
        # View settings
        st.sidebar.checkbox(
            "Show Color Scale",
            value=True,
            key="viz_show_colorscale"
        )
        
        st.sidebar.checkbox(
            "Auto-scale Range",
            value=True,
            key="viz_auto_scale"
        )
    
    def render_visualization(self, data: Optional[np.ndarray] = None):
        """Render the complete visualization with controls"""
        if data is None:
            # Generate sample 4D data
            x = np.linspace(-5, 5, 50)
            y = np.linspace(-5, 5, 50)
            X, Y = np.meshgrid(x, y)
            t = np.linspace(0, 2*np.pi, 20)
            data = np.zeros((50, 50, 1, len(t)))
            
            for i, time in enumerate(t):
                data[..., 0, i] = np.sin(np.sqrt(X**2 + Y**2) - time)
        
        try:
            # Create and display visualization
            fig = self.create_animation(data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display data info
            st.info(f"Data shape: {data.shape}")
            st.info(f"Value range: [{data.min():.2f}, {data.max():.2f}]")
            
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
            st.info("Try adjusting the data shape or visualization settings")
