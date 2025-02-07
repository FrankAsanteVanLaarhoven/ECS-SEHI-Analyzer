import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple, Dict, List
import pandas as pd
from dataclasses import dataclass

@dataclass
class VisualizationConfig:
    """Configuration for 4D visualization"""
    resolution: Tuple[int, int] = (512, 512)
    time_steps: int = 50
    colorscale: str = "Viridis"
    opacity: float = 0.7
    surface_count: int = 15
    animation_frame_duration: int = 100

class DataVisualizer4D:
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.current_data = None
        self.figure = None
    
    def preprocess_data(self, data: Optional[np.ndarray] = None) -> np.ndarray:
        """Preprocess 4D data for visualization"""
        if data is None:
            # Generate sample 2D + time data
            x = np.linspace(-5, 5, 50)
            y = np.linspace(-5, 5, 50)
            X, Y = np.meshgrid(x, y)
            t = np.linspace(0, 2*np.pi, 20)
            
            # Create 3D array (height, width, time)
            data = np.zeros((50, 50, len(t)))
            for i, time in enumerate(t):
                data[:, :, i] = np.sin(np.sqrt(X**2 + Y**2) - time)
            return data
        
        # Handle different input shapes
        if data.ndim == 4:  # (H, W, C, T)
            if data.shape[2] == 1:
                data = data.squeeze(axis=2)  # Remove single channel dimension
            else:
                data = np.mean(data, axis=2)  # Average over channels
        elif data.ndim == 2:  # Single frame
            data = data[..., np.newaxis]  # Add time dimension
        
        return data

    def create_frame_data(self, data: np.ndarray, frame_idx: int) -> pd.DataFrame:
        """Create DataFrame for a single frame"""
        frame = data[:, :, frame_idx]
        df = pd.DataFrame()
        df['y'], df['x'] = np.mgrid[0:frame.shape[0], 0:frame.shape[1]].reshape(2, -1)
        df['value'] = frame.flatten()
        return df

    def create_animation(self, data: Optional[np.ndarray] = None) -> go.Figure:
        """Create animated visualization of time-series data"""
        data = self.preprocess_data(data)
        
        # Create the first frame
        df = self.create_frame_data(data, 0)
        
        # Create the heatmap using px.density_heatmap
        fig = px.density_heatmap(
            df,
            x='x',
            y='y',
            z='value',
            title="Time Evolution",
            labels={'value': 'Intensity'},
            color_continuous_scale='Viridis'
        )
        
        # Create frames for animation
        frames = []
        for t in range(data.shape[2]):
            frame_df = self.create_frame_data(data, t)
            frames.append(
                go.Frame(
                    data=[go.Heatmap(
                        x=frame_df['x'].unique(),
                        y=frame_df['y'].unique(),
                        z=frame_df['value'].values.reshape(data.shape[0], data.shape[1]),
                        colorscale='Viridis',
                        showscale=True
                    )],
                    name=f'frame_{t}'
                )
            )
        
        fig.frames = frames
        
        # Update layout with animation controls
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': '▶️ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': self.config.animation_frame_duration, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                    }]
                }, {
                    'label': '⏸️ Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }]
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
                    for i in range(data.shape[2])
                ]
            }]
        )
        
        # Update axes labels
        fig.update_xaxes(title="X Position")
        fig.update_yaxes(title="Y Position")
        
        return fig

    def render_controls(self):
        """Render visualization controls"""
        st.sidebar.subheader("🎮 Visualization Controls")
        
        # Animation speed
        self.config.animation_frame_duration = st.sidebar.slider(
            "Frame Duration (ms)",
            min_value=50,
            max_value=1000,
            value=self.config.animation_frame_duration,
            step=50,
            key="viz_frame_duration"
        )
        
        # Color settings
        colormap = st.sidebar.selectbox(
            "Colormap",
            ["Viridis", "Plasma", "Inferno", "Magma"],
            key="viz_colormap",
            index=["Viridis", "Plasma", "Inferno", "Magma"].index(self.config.colorscale)
        )
        
        # View settings
        show_colorscale = st.sidebar.checkbox(
            "Show Color Scale",
            value=True,
            key="viz_show_colorscale"
        )
        
        auto_scale = st.sidebar.checkbox(
            "Auto-scale Range",
            value=True,
            key="viz_auto_scale"
        )
        
        if not auto_scale:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                vmin = st.number_input("Min Value", value=0.0, key="viz_vmin")
            with col2:
                vmax = st.number_input("Max Value", value=1.0, key="viz_vmax")
    
    def render_visualization(self, data: Optional[np.ndarray] = None):
        """Render the complete visualization with controls"""
        try:
            if data is not None:
                self.current_data = data
            
            # Visualization type selector
            viz_type = st.selectbox(
                "Visualization Type",
                ["Surface Plot", "Defect Map", "Chemical Analysis"],
                key="viz_type"
            )
            
            # Add visualization controls
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Render selected visualization
                if viz_type == "Surface Plot":
                    fig = self.create_surface_plot()
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                elif viz_type == "Defect Map":
                    fig = self.create_defect_map()
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    self.render_chemical_analysis()
                
            with col2:
                st.markdown("### Controls")
                st.slider("Resolution", 1, 100, 50, key="viz_resolution")
                st.slider("Sensitivity", 0.0, 1.0, 0.5, key="viz_sensitivity")
                
                with st.expander("Advanced Settings"):
                    st.selectbox(
                        "Colormap",
                        ["Viridis", "Plasma", "Inferno", "Magma"],
                        key="viz_colormap"
                    )
                    st.checkbox("Auto-scale", value=True, key="viz_autoscale")
            
        except Exception as e:
            st.error("Visualization error. Please check data format and settings.")
            if st.checkbox("Show error details"):
                st.exception(e)
            
    def render_chemical_analysis(self):
        """Render chemical analysis visualization"""
        # Generate sample chemical data
        elements = ['Si', 'O', 'N', 'C', 'H']
        concentrations = np.random.uniform(0, 100, len(elements))
        
        fig = go.Figure(data=[
            go.Bar(
                x=elements,
                y=concentrations,
                text=np.round(concentrations, 1),
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Chemical Composition Analysis",
            xaxis_title="Element",
            yaxis_title="Concentration (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional chemical metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Purity", "98.5%", "+0.5%")
        with col2:
            st.metric("Contamination", "1.5%", "-0.5%")
        with col3:
            st.metric("Layer Thickness", "245 nm", "+5 nm")

    def create_surface_plot(self) -> go.Figure:
        """Create 3D surface plot"""
        try:
            # Generate sample data if none provided
            if self.current_data is None:
                x = np.linspace(-5, 5, 100)
                y = np.linspace(-5, 5, 100)
                X, Y = np.meshgrid(x, y)
                Z = np.sin(np.sqrt(X**2 + Y**2))
                
            # Create basic surface plot
            fig = go.Figure()
            
            fig.add_trace(go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale='Viridis'
            ))
            
            # Update layout with basic 3D configuration
            fig.update_layout(
                title='Surface Analysis',
                scene={
                    'xaxis': {'title': 'X (μm)'},
                    'yaxis': {'title': 'Y (μm)'},
                    'zaxis': {'title': 'Z (nm)'},
                    'camera': {
                        'up': {'x': 0, 'y': 0, 'z': 1},
                        'center': {'x': 0, 'y': 0, 'z': 0},
                        'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}
                    }
                },
                width=800,
                height=600,
                margin=dict(l=65, r=50, b=65, t=90)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating surface plot: {str(e)}")
            return None
        
    def create_defect_map(self) -> go.Figure:
        """Create defect detection heatmap"""
        try:
            # Generate sample defect data
            if self.current_data is None:
                x = np.linspace(-5, 5, 100)
                y = np.linspace(-5, 5, 100)
                X, Y = np.meshgrid(x, y)
                # Simulate defects as Gaussian peaks
                defects = np.zeros_like(X)
                for _ in range(5):
                    x0, y0 = np.random.uniform(-4, 4, 2)
                    amplitude = np.random.uniform(0.5, 1.0)
                    sigma = np.random.uniform(0.2, 0.5)
                    defects += amplitude * np.exp(-((X-x0)**2 + (Y-y0)**2)/(2*sigma**2))
            
            # Create basic heatmap
            fig = go.Figure()
            
            fig.add_trace(go.Heatmap(
                z=defects,
                colorscale='Viridis'
            ))
            
            # Update layout with basic configuration
            fig.update_layout(
                title='Defect Detection Map',
                xaxis_title='X Position (μm)',
                yaxis_title='Y Position (μm)',
                width=800,
                height=600,
                margin=dict(l=65, r=50, b=65, t=90)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating defect map: {str(e)}")
            return None

    def create_4d_visualization(self, data: np.ndarray, timestamps: List[float]) -> go.Figure:
        """Create 4D visualization of temporal-spatial data"""
        if data.ndim != 4:  # (time, x, y, features)
            raise ValueError("Data must be 4-dimensional (time, x, y, features)")
            
        self.current_data = data
        fig = go.Figure()
        
        # Create frames for temporal evolution
        frames = []
        for t in range(min(len(timestamps), data.shape[0])):
            frame = self._create_time_slice(data[t], timestamps[t])
            frames.append(frame)
            
        # Add base plot
        fig.add_trace(frames[0].data[0])
        
        # Configure animation
        fig.frames = frames
        fig.update_layout(
            scene=dict(
                xaxis_title="X Position (nm)",
                yaxis_title="Y Position (nm)",
                zaxis_title="Intensity",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            updatemenus=[self._create_animation_menu()],
            sliders=[self._create_time_slider(timestamps)]
        )
        
        self.figure = fig
        return fig
    
    def render_interactive_view(self, title: str = "4D Material Analysis"):
        """Render interactive visualization in Streamlit"""
        if self.figure is None:
            st.warning("No visualization data available")
            return
            
        st.markdown(f"### {title}")
        
        # Control panel
        with st.expander("Visualization Controls", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                colorscale = st.selectbox(
                    "Color Scale",
                    ["Viridis", "Plasma", "Inferno", "Magma", "Cividis"]
                )
                opacity = st.slider("Opacity", 0.0, 1.0, self.config.opacity)
                
            with col2:
                surface_count = st.slider(
                    "Surface Count",
                    5, 30, self.config.surface_count
                )
                frame_duration = st.slider(
                    "Animation Speed (ms)",
                    50, 500, self.config.animation_frame_duration
                )
                
        # Update visualization
        self.config.colorscale = colorscale
        self.config.opacity = opacity
        self.config.surface_count = surface_count
        self.config.animation_frame_duration = frame_duration
        
        if self.current_data is not None:
            self.figure.update_traces(
                colorscale=colorscale,
                opacity=opacity
            )
            
        # Display plot
        st.plotly_chart(self.figure, use_container_width=True)
        
    def _create_time_slice(self, data_slice: np.ndarray, timestamp: float) -> go.Frame:
        """Create a single time slice for visualization"""
        surface = go.Surface(
            x=np.linspace(0, 1, data_slice.shape[0]),
            y=np.linspace(0, 1, data_slice.shape[1]),
            z=data_slice,
            colorscale=self.config.colorscale,
            opacity=self.config.opacity,
            showscale=True,
            name=f"t={timestamp:.2f}s"
        )
        
        return go.Frame(
            data=[surface],
            name=str(timestamp)
        )
    
    def _create_animation_menu(self) -> Dict:
        """Create animation control menu"""
        return dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, {"frame": {"duration": self.config.animation_frame_duration}}]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0}}])
            ]
        )
    
    def _create_time_slider(self, timestamps: List[float]) -> Dict:
        """Create time slider for animation control"""
        return dict(
            active=0,
            currentvalue={"prefix": "Time: ", "suffix": " s"},
            steps=[dict(
                method="animate",
                args=[[str(t)], {"frame": {"duration": self.config.animation_frame_duration}}],
                label=f"{t:.2f}"
            ) for t in timestamps]
        ) 