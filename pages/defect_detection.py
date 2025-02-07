import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.ecs_sehi_analyzer.core.ui_engine import (
    render_sidebar_controls,
    show_loading_spinner,
    render_plot,
    show_success
)

def generate_sample_defects(size=512, num_defects=5):
    """Generate sample data with defects"""
    data = np.random.rand(size, size)
    
    # Add artificial defects
    for _ in range(num_defects):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        radius = np.random.randint(10, 30)
        
        Y, X = np.ogrid[-y:size-y, -x:size-x]
        dist = np.sqrt(X*X + Y*Y)
        mask = dist <= radius
        data[mask] = np.random.uniform(0.8, 1.0)
    
    return data

def create_defect_plots(data, threshold):
    """Create visualization for defect detection"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Original Image",
            "Defect Probability Map",
            "Detected Defects",
            "Size Distribution"
        )
    )
    
    # Original image
    fig.add_trace(
        go.Heatmap(z=data, colorscale='viridis', name="Original"),
        row=1, col=1
    )
    
    # Probability map
    prob_map = (data > threshold).astype(float)
    fig.add_trace(
        go.Heatmap(z=prob_map, colorscale='RdBu', name="Probability"),
        row=1, col=2
    )
    
    # Detected defects
    defects = np.zeros_like(data)
    defects[data > threshold] = 1
    fig.add_trace(
        go.Heatmap(z=defects, colorscale='Reds', name="Defects"),
        row=2, col=1
    )
    
    # Size distribution
    fig.add_trace(
        go.Histogram(x=data[data > threshold].flatten(), name="Size Dist"),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        template="plotly_dark"
    )
    
    return fig

def render_defect_detection():
    """Render the defect detection page"""
    render_sidebar_controls()
    
    st.title("Defect Detection")
    
    # Initialize session state
    if 'defect_data' not in st.session_state:
        st.session_state.defect_data = np.random.rand(512, 512)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Detection Settings")
        
        threshold = st.slider(
            "Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
            key="defect_threshold"
        )
        
        with st.expander("Advanced Settings"):
            st.selectbox(
                "Detection Method",
                ["Threshold", "ML-based", "Pattern Matching"],
                key="detection_method"
            )
            
            st.number_input(
                "Min Defect Size",
                min_value=1,
                max_value=100,
                value=10,
                key="min_defect_size"
            )
            
            st.checkbox("Filter Noise", value=True, key="filter_noise")
        
        if st.button("Detect Defects", type="primary"):
            with show_loading_spinner():
                defects = st.session_state.defect_data > threshold
                num_defects = np.sum(defects)
                show_success(f"Found {int(num_defects)} potential defects")
    
    with col1:
        tabs = st.tabs(["Original", "Defects", "Analysis"])
        
        with tabs[0]:
            fig = px.imshow(
                st.session_state.defect_data,
                color_continuous_scale='viridis',
                title="Original Image"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            defects = st.session_state.defect_data > threshold
            fig = px.imshow(
                defects.astype(float),
                color_continuous_scale='RdBu',
                title="Detected Defects"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            # Display metrics
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                st.metric("Total Area", f"{100:.1f} μm²")
            with metrics_cols[1]:
                defect_ratio = np.mean(defects)
                st.metric("Defect Ratio", f"{defect_ratio*100:.2f}%")
            with metrics_cols[2]:
                st.metric("Avg. Defect Size", f"{20:.1f} nm")

if __name__ == "__main__":
    render_defect_detection()
