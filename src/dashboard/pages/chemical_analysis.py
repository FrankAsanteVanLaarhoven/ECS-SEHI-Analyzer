import streamlit as st
import numpy as np
from utils.chemical.chemical_analyzer import ChemicalAnalyzer

def render_chemical_analysis():
    """Render the chemical analysis page."""
    st.markdown('<h1 class="main-header">Chemical Analysis</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = ChemicalAnalyzer()
    
    # Create layout
    left_col, main_col = st.columns([1, 3])
    
    with left_col:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.subheader("Chemical Analysis Controls")
        
        # Element selection
        elements = ["Carbon", "Silicon", "Oxygen", "Nitrogen", "Hydrogen"]
        selected_elements = st.multiselect(
            "Select Elements",
            elements,
            default=["Carbon", "Silicon"],
            help="Choose elements to analyze"
        )
        
        # Analysis parameters
        resolution = st.slider(
            "Resolution",
            min_value=128,
            max_value=1024,
            value=512,
            step=128,
            help="Analysis resolution"
        )
        
        sensitivity = st.slider(
            "Sensitivity",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            format="%.2f",
            help="Analysis sensitivity"
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            analysis_mode = st.selectbox(
                "Analysis Mode",
                ["Standard", "High Precision", "Fast Scan"]
            )
            
            background_correction = st.checkbox(
                "Background Correction",
                value=True
            )
        
        # Analysis button
        if st.button("Run Analysis", type="primary"):
            if not selected_elements:
                st.error("Please select at least one element.")
                return
                
            with main_col:
                with st.spinner("Running chemical analysis..."):
                    # Generate sample data
                    data = np.random.normal(0.5, 0.1, (resolution, resolution))
                    
                    # Run analysis
                    results = analyzer.analyze_composition(data, selected_elements)
                    
                    # Display results
                    tabs = st.tabs(selected_elements)
                    for element, tab in zip(selected_elements, tabs):
                        with tab:
                            # Show element stats
                            stats = results['stats'][element]
                            cols = st.columns(4)
                            with cols[0]:
                                st.metric("Mean", f"{stats['mean']:.3f}")
                            with cols[1]:
                                st.metric("Std Dev", f"{stats['std']:.3f}")
                            with cols[2]:
                                st.metric("Max", f"{stats['max']:.3f}")
                            with cols[3]:
                                st.metric("Min", f"{stats['min']:.3f}")
                            
                            # Show element distribution
                            fig = analyzer.create_composition_plot(results, element)
                            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main visualization area
    with main_col:
        if "chemical_analysis_btn" not in st.session_state:
            st.markdown('<div class="visualization-area">', unsafe_allow_html=True)
            st.markdown("""
                <div style="text-align: center; padding: 40px;">
                    <h3 style="color: #94A3B8;">Chemical Analysis</h3>
                    <p style="color: #64748B;">
                        Select elements and analysis parameters, then click 
                        'Run Analysis' to begin. Results will appear here.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True) 