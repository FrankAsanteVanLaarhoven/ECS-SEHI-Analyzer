import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.multimodal_analyzer import MultiModalAnalyzer

def render_multimodal_analysis():
    """Render the multi-modal analysis page."""
    st.markdown('<h1 class="main-header">Multi-Modal Analysis</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = MultiModalAnalyzer()
    
    # Create layout
    left_col, main_col = st.columns([1, 3])
    
    with left_col:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.subheader("Multi-Modal Controls")
        
        # Analysis modes
        analysis_modes = st.multiselect(
            "Analysis Modes",
            ["Chemical-Surface", "Surface-Defect", "Chemical-Defect"],
            default=["Chemical-Surface"],
            help="Select analysis modes to combine"
        )
        
        # Resolution control
        resolution = st.slider(
            "Resolution",
            min_value=128,
            max_value=1024,
            value=512,
            step=128,
            help="Analysis resolution"
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            fusion_method = st.selectbox(
                "Fusion Method",
                ["Early Fusion", "Late Fusion", "Hybrid"]
            )
            
            correlation_threshold = st.slider(
                "Correlation Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                format="%.2f"
            )
        
        if st.button("Run Multi-Modal Analysis", type="primary"):
            if not analysis_modes:
                st.error("Please select at least one analysis mode.")
                return
                
            with main_col:
                with st.spinner("Running multi-modal analysis..."):
                    # Generate sample data
                    data = np.random.normal(0.5, 0.1, (resolution, resolution))
                    
                    # Run analysis
                    results = analyzer.analyze_multimodal(
                        data=data,
                        analysis_modes=analysis_modes,
                        fusion_method=fusion_method,
                        correlation_threshold=correlation_threshold
                    )
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    # Show correlation matrix
                    if len(analysis_modes) > 1:
                        corr_fig = analyzer.create_multimodal_plot(results)
                        st.plotly_chart(corr_fig, use_container_width=True)
                    
                    # Show mode-specific results
                    tabs = st.tabs(analysis_modes)
                    for mode, tab in zip(analysis_modes, tabs):
                        with tab:
                            mode_fig = analyzer.create_mode_plot(results, mode)
                            st.plotly_chart(mode_fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main visualization area
    with main_col:
        if "multimodal_analysis_btn" not in st.session_state:
            st.markdown('<div class="visualization-area">', unsafe_allow_html=True)
            st.markdown("""
                <div style="text-align: center; padding: 40px;">
                    <h3 style="color: #94A3B8;">Multi-Modal Analysis</h3>
                    <p style="color: #64748B;">
                        Select analysis modes and parameters, then click 
                        'Run Multi-Modal Analysis' to begin. Results will appear here.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True) 