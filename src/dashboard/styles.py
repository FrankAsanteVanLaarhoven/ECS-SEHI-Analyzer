"""Shared styles for the SEHI Analysis Dashboard."""

GLOBAL_STYLES = """
<style>
/* Base Theme */
:root {
    --primary: #FF4B4B;
    --primary-hover: #FF6B6B;
    --secondary: #1E293B;
    --accent: #3B82F6;
    --success: #10B981;
    --warning: #F59E0B;
    --error: #EF4444;
    --background: #0F172A;
    --surface: rgba(17, 23, 39, 0.7);
    --border: rgba(255, 255, 255, 0.1);
    --text-primary: #FFFFFF;
    --text-secondary: #94A3B8;
    --text-tertiary: #64748B;
}

/* Common Components */
.main-header {
    font-size: 2.5em;
    font-weight: 600;
    margin-bottom: 1em;
    color: var(--text-primary);
}

.tool-navigation {
    display: flex;
    gap: 1px;
    background: var(--secondary);
    padding: 4px;
    border-radius: 8px;
    margin-bottom: 20px;
}

.nav-item {
    flex: 1;
    padding: 12px 20px;
    text-align: center;
    color: var(--text-secondary);
    text-decoration: none;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 6px;
}

.nav-item:hover {
    background: rgba(255, 255, 255, 0.05);
    color: var(--text-primary);
}

.nav-item.active {
    background: var(--surface);
    color: var(--text-primary);
}

.control-panel {
    background: var(--surface);
    border-radius: 10px;
    padding: 20px;
    border: 1px solid var(--border);
}

.visualization-area {
    background: var(--surface);
    border-radius: 10px;
    padding: 20px;
    border: 1px solid var(--border);
    min-height: 500px;
}

/* Controls */
.stSlider, .stSelectbox {
    margin: 1rem 0;
}

/* Plotly Customization */
.js-plotly-plot .plotly .modebar {
    background: rgba(17, 23, 39, 0.7) !important;
}

.js-plotly-plot .plotly .modebar-btn path {
    fill: var(--text-secondary) !important;
}

/* Responsive Design */
@media (max-width: 768px) {
    .tool-navigation {
        flex-direction: column;
    }
    
    .nav-item {
        width: 100%;
        text-align: left;
    }
}
</style>
"""

def inject_styles():
    """Inject global styles into the Streamlit app."""
    import streamlit as st
    st.markdown(GLOBAL_STYLES, unsafe_allow_html=True) 