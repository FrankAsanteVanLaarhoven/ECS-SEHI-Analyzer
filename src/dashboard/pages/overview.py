import streamlit as st
import plotly.graph_objects as go
from utils.visualization import DataVisualizer

def render_overview(dashboard):
    """Render overview page with expandable analysis cards and system status."""
    
    # Create two columns: main content (3) and system status (1)
    main_col, status_col = st.columns([3, 1])
    
    with main_col:
        # Surface Analysis Card
        with st.expander("Surface Analysis - Sample A (2024-01-15 14:30:22)", expanded=True):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write("**Analysis Type:** Defect Analysis")
                st.write("**Status:** Complete")
                st.write("**Processing Time:** 3.2 minutes")
            
            with col2:
                st.write("**Key Findings:**")
                st.write("• Defect Density: 2.3 defects/μm²")
                st.write("• Average Defect Size: 45.2 nm")
                st.write("• Surface Roughness: 1.2 nm RMS")
                st.write("• Confidence Score: 0.94")
        
        # Composition Mapping Card
        with st.expander("Composition Mapping - Battery Interface (2024-01-15 11:15:03)", expanded=True):
            st.write("**Analysis Type:** Chemical Distribution")
            st.write("**Status:** Complete")
            st.write("**Key Findings:**")
            st.write("• Dominant Elements: Li, Ni, Mn, Co")
            
        # Particle Analysis Card
        with st.expander("Particle Analysis - Catalyst (2024-01-14 16:45:11)", expanded=True):
            st.write("**Analysis Type:** Size Distribution")
            st.write("**Status:** Complete")
            st.write("**Key Findings:**")
            st.write("• Mean Size: 25.3 nm")
            st.write("• Size Distribution: Log-normal")
    
    with status_col:
        # System Status Section
        st.markdown("### Active Models")
        st.markdown("#### 4/5 Online")
        
        st.markdown("### GPU Utilization")
        st.markdown("#### 76%")
        
        st.markdown("### Processing Queue")
        st.markdown("#### 2 Tasks")
        
        st.markdown("### Average Processing Time")
        st.markdown("#### 4.3 minutes")
        
        # Activity Log Section
        st.markdown("### Activity Log")
        
        # Activity log entries with different background colors
        st.markdown("""
            <div style='background-color: #1B4332; padding: 10px; margin: 5px 0; border-radius: 5px;'>
                14:30:22 - Surface analysis completed
            </div>
            <div style='background-color: #1A374D; padding: 10px; margin: 5px 0; border-radius: 5px;'>
                14:15:18 - New data uploaded
            </div>
            <div style='background-color: #3C2A21; padding: 10px; margin: 5px 0; border-radius: 5px;'>
                13:45:03 - Model calibration
            </div>
            <div style='background-color: #1A374D; padding: 10px; margin: 5px 0; border-radius: 5px;'>
                13:30:55 - System backup
            </div>
        """, unsafe_allow_html=True)
        
        # Quick Actions Section
        st.markdown("### Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            st.button("New Analysis", key="new_analysis_btn", use_container_width=True)
            st.button("View Reports", key="view_reports_btn", use_container_width=True)
        with col2:
            st.button("System Check", key="system_check_btn", use_container_width=True)
            st.button("Clear Cache", key="clear_cache_btn", type="secondary", use_container_width=True) 