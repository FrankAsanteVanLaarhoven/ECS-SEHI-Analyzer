import streamlit as st
import plotly.graph_objects as go

def configure_page():
    """Configure the Streamlit page settings"""
    st.set_page_config(
        page_title="SEHI Analysis Dashboard",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def render_sidebar_controls():
    """Render the sidebar controls"""
    with st.sidebar:
        with st.expander("Data Settings", expanded=True):
            # Only set the selectbox value if it doesn't exist in session state
            if "data_source" not in st.session_state:
                st.session_state.data_source = "Sample Data"
            
            st.selectbox(
                "Data Source",
                ["Sample Data", "Upload File"],
                key="data_source"
            )
            
            if st.session_state.data_source == "Upload File":
                st.file_uploader(
                    "Upload Data File",
                    type=["csv", "npy"],
                    key="uploaded_file"
                )
        
        with st.expander("Analysis Settings", expanded=True):
            # Initialize resolution in session state if not present
            if "resolution" not in st.session_state:
                st.session_state.resolution = 512
                
            # Initialize noise_reduction in session state if not present
            if "noise_reduction" not in st.session_state:
                st.session_state.noise_reduction = 0.5
            
            resolution = st.slider(
                "Resolution",
                min_value=128,
                max_value=1024,
                value=st.session_state.resolution,
                step=128,
                key="resolution_slider"  # Changed key to avoid conflict
            )
            # Update session state after slider interaction
            st.session_state.resolution = resolution
            
            noise_reduction = st.slider(
                "Noise Reduction",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.noise_reduction,
                step=0.01,
                key="noise_reduction_slider"  # Changed key to avoid conflict
            )
            # Update session state after slider interaction
            st.session_state.noise_reduction = noise_reduction

def render_analysis_controls():
    """Render common analysis controls"""
    if "analysis_method" not in st.session_state:
        st.session_state.analysis_method = "Basic"
        
    if "viz_type" not in st.session_state:
        st.session_state.viz_type = "2D Map"
    
    analysis_method = st.selectbox(
        "Analysis Method",
        ["Basic", "Advanced", "Expert"],
        key="analysis_method_select"  # Changed key to avoid conflict
    )
    st.session_state.analysis_method = analysis_method
    
    viz_type = st.selectbox(
        "Visualization Type",
        ["2D Map", "3D Surface", "Contour Plot"],
        key="viz_type_select"  # Changed key to avoid conflict
    )
    st.session_state.viz_type = viz_type

def show_loading_spinner():
    """Show a loading spinner with message"""
    return st.spinner("Processing... Please wait...")

def render_plot(fig, title=None):
    """Render a plotly figure with consistent styling"""
    if title:
        fig.update_layout(
            title=title,
            title_x=0.5,
            margin=dict(t=50, l=0, r=0, b=0),
            template="plotly_dark",
            height=600
        )
    st.plotly_chart(fig, use_container_width=True)

def show_error(message):
    """Display error message with consistent styling"""
    st.error(f"🚨 Error: {message}")

def show_success(message):
    """Display success message with consistent styling"""
    st.success(f"✅ {message}")

def show_info(message):
    """Display info message with consistent styling"""
    st.info(f"ℹ️ {message}")
