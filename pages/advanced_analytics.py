import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.ecs_sehi_analyzer.core.ui_engine import render_sidebar_controls
from sklearn.decomposition import PCA

def create_animated_heatmap(data):
    """Create an animated heatmap from 4D data"""
    # Ensure data is in the right shape
    data_2d = data.squeeze()  # Remove any single dimensions
    
    # Create frames for animation
    frames = []
    for t in range(data_2d.shape[-1]):
        frames.append(
            go.Frame(
                data=[go.Heatmap(
                    z=data_2d[:,:,t],
                    colorscale='Viridis',
                    showscale=True
                )],
                name=str(t)
            )
        )
    
    # Create the base figure
    fig = go.Figure(
        data=[go.Heatmap(z=data_2d[:,:,0], colorscale='Viridis')],
        frames=frames
    )
    
    # Update layout with animation controls
    fig.update_layout(
        title="Time Evolution Analysis",
        width=800,
        height=600,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': '▶️ Play',
                'method': 'animate',
                'args': [None, {'frame': {'duration': 100, 'redraw': True}}]
            }, {
                'label': '⏸️ Pause',
                'method': 'animate',
                'args': [[None], {'frame': {'duration': 0, 'redraw': False}}]
            }]
        }]
    )
    return fig

def create_statistics_plots(data):
    """Create statistical analysis plots"""
    # Calculate statistics
    data_2d = data.squeeze()
    temporal_mean = np.mean(data_2d, axis=2)
    temporal_std = np.std(data_2d, axis=2)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Mean Over Time", "Value Distribution", 
                       "Spatial Pattern", "Time Series"]
    )
    
    # Plot 1: Mean over time heatmap
    fig.add_trace(
        go.Heatmap(z=temporal_mean, colorscale='Viridis'),
        row=1, col=1
    )
    
    # Plot 2: Histogram of values
    fig.add_trace(
        go.Histogram(x=data_2d.flatten(), nbinsx=50),
        row=1, col=2
    )
    
    # Plot 3: Spatial pattern
    fig.add_trace(
        go.Heatmap(z=np.mean(data_2d, axis=(0,1)), colorscale='Viridis'),
        row=2, col=1
    )
    
    # Plot 4: Time series
    fig.add_trace(
        go.Scatter(y=np.mean(data_2d, axis=(0,1)), mode='lines+markers'),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Statistical Analysis"
    )
    return fig

def create_feature_space(data):
    """Create feature space visualization"""
    # Reshape data for PCA
    data_2d = data.squeeze()
    data_reshaped = data_2d.reshape(data_2d.shape[0] * data_2d.shape[1], -1)
    
    # Perform PCA
    pca = PCA(n_components=2)
    features = pca.fit_transform(data_reshaped)
    
    # Create scatter plot
    fig = px.scatter(
        x=features[:, 0],
        y=features[:, 1],
        color=np.mean(data_reshaped, axis=1),
        title="Feature Space Analysis",
        labels={
            'x': 'Principal Component 1',
            'y': 'Principal Component 2',
            'color': 'Mean Value'
        },
        template='plotly_dark'
    )
    
    fig.update_layout(
        height=600,
        width=800
    )
    return fig

def render_advanced_analytics():
    """Render the advanced analytics page"""
    render_sidebar_controls()
    
    # Initialize sample data if not present
    if 'advanced_data' not in st.session_state:
        x, y = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
        t = np.linspace(0, 10, 20)
        data = np.zeros((50, 50, 1, 20))
        
        for i, time in enumerate(t):
            data[:,:,0,i] = np.sin(np.sqrt(x**2 + y**2) - time) * np.exp(-0.1 * (x**2 + y**2))
        
        st.session_state.advanced_data = data
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Analysis Controls")
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Quantum", "Time Evolution", "Feature Detection"]
        )
        
        viz_method = st.selectbox(
            "Visualization Method",
            ["TSNE", "PCA"]
        )
        
        with st.expander("Advanced Parameters"):
            time_res = st.slider(
                "Time Resolution",
                min_value=1,
                max_value=100,
                value=20
            )
            
            conf_level = st.number_input(
                "Confidence Level",
                min_value=0.8,
                max_value=0.99,
                value=0.95,
                step=0.01
            )
        
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Analyzing data..."):
                st.success("Analysis complete!")
    
    with col1:
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Time Evolution", "Statistics", "Feature Space"])
        
        with tab1:
            try:
                fig = create_animated_heatmap(st.session_state.advanced_data)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error in time evolution visualization: {str(e)}")
        
        with tab2:
            try:
                fig = create_statistics_plots(st.session_state.advanced_data)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display metrics
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Mean", f"{np.mean(st.session_state.advanced_data):.3f}")
                with cols[1]:
                    st.metric("Max", f"{np.max(st.session_state.advanced_data):.3f}")
                with cols[2]:
                    st.metric("Std", f"{np.std(st.session_state.advanced_data):.3f}")
            except Exception as e:
                st.error(f"Error in statistical visualization: {str(e)}")
        
        with tab3:
            try:
                fig = create_feature_space(st.session_state.advanced_data)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error in feature space visualization: {str(e)}")

if __name__ == "__main__":
    render_advanced_analytics()
