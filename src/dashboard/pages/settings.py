import streamlit as st

def render_settings():
    """Render the settings page."""
    st.markdown("# Settings")
    
    # Audio Settings
    st.subheader("Audio Settings")
    with st.expander("Audio Configuration"):
        st.slider(
            "Default Volume",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="default_volume"
        )
        
        st.slider(
            "Default Duration (minutes)",
            min_value=1,
            max_value=60,
            value=5,
            step=1,
            key="default_duration"
        )
        
        st.selectbox(
            "Default Background Sound",
            ["None", "Ocean Waves", "Rain Forest", "White Noise"],
            key="default_background"
        )
    
    # Visualization Settings
    st.subheader("Visualization Settings")
    with st.expander("Display Configuration"):
        st.checkbox(
            "Show Wave Patterns",
            value=True,
            key="show_waves"
        )
        
        st.checkbox(
            "Show Real-time Analysis",
            value=True,
            key="show_analysis"
        )
        
        st.selectbox(
            "Color Theme",
            ["Light", "Dark", "Auto"],
            key="color_theme"
        )
    
    # Export Settings
    st.subheader("Export Settings")
    with st.expander("Export Configuration"):
        st.selectbox(
            "Default Export Format",
            ["WAV", "MP3", "OGG"],
            key="export_format"
        )
        
        st.number_input(
            "Sample Rate (Hz)",
            min_value=8000,
            max_value=96000,
            value=44100,
            step=100,
            key="sample_rate"
        )
        
        st.checkbox(
            "Include Metadata",
            value=True,
            key="include_metadata"
        )
    
    # Save Settings
    if st.button("Save Settings", type="primary"):
        st.success("Settings saved successfully!") 