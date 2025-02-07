import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.ecs_sehi_analyzer.core.ui_engine import (
    render_sidebar_controls,
    show_loading_spinner
)

def create_chemical_plots(data):
    """Create chemical analysis visualizations"""
    # Create distribution map
    dist_fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale='Viridis',
        colorbar=dict(
            title=dict(
                text="Intensity",
                side="right"
            )
        )
    ))
    
    dist_fig.update_layout(
        title=None,  # Remove duplicate title
        xaxis_title="Position (nm)",
        yaxis_title="Position (nm)",
        template="plotly_dark",
        margin=dict(t=30)  # Reduce top margin since we removed title
    )
    
    # Create histogram
    hist_data = data.flatten()
    hist_fig = go.Figure(data=go.Histogram(
        x=hist_data,
        nbinsx=50,
        name="Distribution",
        marker_color='purple'
    ))
    
    hist_fig.update_layout(
        title=None,  # Remove duplicate title
        xaxis_title="Chemical Composition",
        yaxis_title="Count",
        template="plotly_dark",
        margin=dict(t=30)  # Reduce top margin
    )
    
    return dist_fig, hist_fig

def render_chemical_analysis():
    """Render the chemical analysis page"""
    render_sidebar_controls()
    
    # Initialize sample data if needed
    if 'chemical_data' not in st.session_state:
        st.session_state.chemical_data = np.random.randn(500, 500)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Analysis Controls")
        
        analysis_method = st.selectbox(
            "Analysis Method",
            ["Basic", "Advanced", "Machine Learning"],
            key="analysis_method"
        )
        
        color_scale = st.selectbox(
            "Color Scale",
            ["Viridis", "Plasma", "Inferno", "Magma"],
            key="color_scale"
        )
        
        with st.expander("Advanced Options"):
            st.slider(
                "Resolution",
                min_value=100,
                max_value=1000,
                value=500,
                step=100,
                key="resolution"
            )
            
            st.checkbox(
                "Show Markers",
                value=True,
                key="show_markers"
            )
            
            marker_types = st.multiselect(
                "Marker Types",
                ["Distribution", "Peaks", "Clusters"],
                default=["Distribution"],
                key="marker_types"
            )
        
        if st.button("Run Analysis", type="primary"):
            with show_loading_spinner():
                st.success("Analysis complete!")
    
    with col1:
        # Add section headers instead of plot titles
        st.subheader("📊 Chemical Distribution Map")
        
        # Create and display plots
        dist_fig, hist_fig = create_chemical_plots(st.session_state.chemical_data)
        
        # Update color scales based on selection
        dist_fig.update_traces(colorscale=color_scale.lower())
        
        # Display plots
        st.plotly_chart(dist_fig, use_container_width=True)
        
        st.subheader("📈 Composition Analysis")
        st.plotly_chart(hist_fig, use_container_width=True)

if __name__ == "__main__":
    render_chemical_analysis()
