import streamlit as st

def render_navigation():
    """Render the main navigation bar."""
    st.markdown("""
        <div class="tool-navigation">
            <a href="#" class="nav-item" id="surface">🔲 3D Surface</a>
            <a href="#" class="nav-item" id="chemical">🧪 Chemical Analysis</a>
            <a href="#" class="nav-item" id="defect">🎯 Defect Detection</a>
            <a href="#" class="nav-item" id="results">📊 Analysis Results</a>
            <a href="#" class="nav-item" id="sustainability">🌱 Sustainability</a>
            <a href="#" class="nav-item" id="multimodal">🔄 Multi-Modal</a>
        </div>
    """, unsafe_allow_html=True) 