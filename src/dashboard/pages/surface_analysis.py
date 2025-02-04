import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.surface_analysis import SurfaceAnalyzer

def render_surface_analysis():
    """Render the 3D surface analysis page."""
    st.markdown('<h1 class="main-header">SEHI Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = SurfaceAnalyzer()
    
    # Create main layout
    left_panel, main_view = st.columns([1, 3])
    
    # Control Panel
    with left_panel:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.subheader("3D Surface Controls")
        
        # Processing settings
        st.markdown('<p class="control-label">Processing Settings</p>', unsafe_allow_html=True)
        resolution = st.slider(
            "Resolution",
            min_value=128,
            max_value=1024,
            value=512,
            step=128,
            key="surface_resolution",
            help="Higher resolution provides more detailed surface mapping"
        )
        
        noise_reduction = st.slider(
            "Noise Reduction",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            format="%.2f",
            key="surface_noise_reduction",
            help="Higher values reduce noise but may smooth out fine details"
        )
        
        # Analysis settings
        st.markdown('<p class="control-label">Analysis Settings</p>', unsafe_allow_html=True)
        analysis_method = st.selectbox(
            "Analysis Method",
            ["Basic", "Advanced", "Machine Learning"],
            key="surface_analysis_method"
        )
        
        # Visualization options
        view_mode = st.selectbox(
            "View Mode",
            ["Height Map", "Roughness Map", "Gradient Map"],
            key="surface_view_mode"
        )
        
        # Analysis button
        if st.button(
            "Generate Surface",
            type="primary",
            key="surface_generate_btn"
        ):
            with main_view:
                st.markdown('<div class="visualization-area">', unsafe_allow_html=True)
                with st.spinner("Generating surface analysis..."):
                    # Run analysis
                    results = analyzer.analyze_surface(
                        resolution=resolution,
                        noise_reduction=noise_reduction,
                        method=analysis_method,
                        view_mode=view_mode
                    )
                    
                    # Display statistics
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Mean Height", f"{results['stats']['mean_height']:.2f} nm")
                    with cols[1]:
                        st.metric("RMS Roughness", f"{results['stats']['rms_roughness']:.2f} nm")
                    with cols[2]:
                        st.metric("Peak Height", f"{results['stats']['peak_height']:.2f} nm")
                    with cols[3]:
                        st.metric("Surface Area", f"{results['stats']['surface_area']:.2f} μm²")
                    
                    # Create and display 3D plot
                    fig = analyzer.create_surface_plot(results, f"3D Surface Analysis - {view_mode}")
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main Visualization Area
    with main_view:
        if "surface_generate_btn" not in st.session_state:
            st.markdown('<div class="visualization-area">', unsafe_allow_html=True)
            st.markdown("""
                <div style="text-align: center; padding: 40px;">
                    <h3 style="color: #94A3B8;">3D Surface Analysis</h3>
                    <p style="color: #64748B;">
                        Configure analysis parameters and click 'Generate Surface' 
                        to begin. The 3D visualization will appear here.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True) 