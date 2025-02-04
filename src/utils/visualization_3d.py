import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional

class LidarVisualizer:
    """Visualization tools for LiDAR point cloud data."""
    
    def visualize_point_cloud(self, 
                            points: np.ndarray,
                            color_by: str = 'intensity',
                            colormap: str = 'viridis') -> go.Figure:
        """Create interactive 3D visualization of point cloud."""
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=points[:, 3] if color_by == 'intensity' else None,
                colorscale=colormap,
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            title="LiDAR Point Cloud",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            ),
            template="plotly_dark"
        )
        
        return fig

class SEHIVisualizer:
    """Visualization tools for SEHI data."""
    
    def visualize_hyperspectral_cube(self,
                                   data: np.ndarray,
                                   energy_axis: np.ndarray,
                                   colormap: str = 'viridis') -> go.Figure:
        """Create interactive visualization of hyperspectral data cube."""
        x, y, z = np.meshgrid(
            np.arange(data.shape[0]),
            np.arange(data.shape[1]),
            energy_axis
        )
        
        fig = go.Figure(data=[go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=data.flatten(),
            opacity=0.1,
            surface_count=20,
            colorscale=colormap
        )])
        
        fig.update_layout(
            title="SEHI Data Cube",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Energy (eV)"
            ),
            template="plotly_dark"
        )
        
        return fig 

def create_3d_animation(results, speed=1.0):
    """Create animated 3D visualization of reconstruction process."""
    frames = []
    n_frames = 60
    
    # Calculate camera path
    cameras = results['camera_poses']
    path = interpolate_camera_path(cameras, n_frames)
    
    # Create frames
    for i in range(n_frames):
        frame = go.Frame(
            data=[
                # Point cloud
                go.Scatter3d(
                    x=results['dense_cloud'][:, 0],
                    y=results['dense_cloud'][:, 1],
                    z=results['dense_cloud'][:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=results['dense_cloud'][:, 3],
                        colorscale='Viridis',
                        opacity=0.8
                    )
                ),
                # Camera position
                go.Scatter3d(
                    x=[path[i][0]],
                    y=[path[i][1]],
                    z=[path[i][2]],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='blue',
                        symbol='square'
                    )
                )
            ],
            name=f"frame{i}"
        )
        frames.append(frame)
    
    # Create figure with animation
    fig = go.Figure(
        frames=frames,
        layout=go.Layout(
            scene=dict(
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='data'
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 50/speed, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate'
                    }]
                }]
            }],
            template="plotly_dark"
        )
    )
    
    return fig

def interpolate_camera_path(cameras, n_frames):
    """Create smooth camera path for animation."""
    # Extract camera positions
    positions = cameras[:, :3, 3]
    
    # Create interpolated path
    t = np.linspace(0, 1, n_frames)
    path = []
    
    for i in range(n_frames):
        # Circular path around the model
        angle = 2 * np.pi * t[i]
        radius = np.mean(np.abs(positions)) * 1.5
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = np.mean(positions[:, 2])
        path.append([x, y, z])
    
    return np.array(path) 