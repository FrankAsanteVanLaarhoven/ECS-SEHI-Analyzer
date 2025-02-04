import streamlit as st
from pathlib import Path
import sys

# Set page config first, before any other Streamlit commands
st.set_page_config(
    page_title="SEHI Analysis Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root and src to Python path
project_root = Path(__file__).parent.absolute()
sys.path.extend([str(project_root), str(project_root / 'src')])

# Configure modern purple glassmorphic theme
st.markdown("""
<style>
    /* Modern gradient background with subtle animation */
    .main {
        background: linear-gradient(125deg, #1E1E2E 0%, #2D2B55 50%, #2B2F40 100%);
        animation: gradientShift 15s ease infinite;
        position: relative;
        overflow: hidden;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Floating glass effect */
    .main::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 50% -20%, rgba(130, 87, 229, 0.15), transparent 70%);
        z-index: 0;
    }
    
    /* Enhanced glassmorphic cards */
    .element-container {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 24px !important;
        padding: 24px !important;
        margin: 12px 0 !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2) !important;
        transform: translateZ(0);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .element-container:hover {
        transform: translateY(-2px) translateZ(0);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.3) !important;
        border: 1px solid rgba(130, 87, 229, 0.2) !important;
    }
    
    /* Modern button styling */
    .stButton>button {
        background: linear-gradient(135deg, rgba(130, 87, 229, 0.1) 0%, rgba(130, 87, 229, 0.05) 100%) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(130, 87, 229, 0.2) !important;
        border-radius: 12px !important;
        color: #E2E8F0 !important;
        padding: 12px 24px !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, rgba(130, 87, 229, 0.2) 0%, rgba(130, 87, 229, 0.1) 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(130, 87, 229, 0.2) !important;
    }
    
    /* Metrics with depth */
    .stMetric {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(16px) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        border: 1px solid rgba(130, 87, 229, 0.1) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Interactive sliders */
    .stSlider {
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #8257E5 0%, #9466FF 100%) !important;
    }
    
    /* Modern select boxes */
    .stSelectbox > div {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(130, 87, 229, 0.1) !important;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div:hover {
        border-color: rgba(130, 87, 229, 0.3) !important;
    }
    
    /* Typography enhancements */
    h1, h2, h3 {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar refinements */
    .css-1n76uvr {
        background: rgba(30, 30, 46, 0.9) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(130, 87, 229, 0.1) !important;
    }
    
    /* Navigation enhancement */
    .nav-button {
        background: rgba(130, 87, 229, 0.05) !important;
        backdrop-filter: blur(12px) !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        margin: 6px !important;
        color: #E2E8F0 !important;
        border: 1px solid rgba(130, 87, 229, 0.1) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-weight: 500 !important;
    }
    
    .nav-button:hover {
        background: rgba(130, 87, 229, 0.1) !important;
        transform: translateY(-2px);
        border-color: rgba(130, 87, 229, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Application entry point."""
    try:
        from src.dashboard import Dashboard, inject_styles
        
        # Apply styles
        inject_styles()
        
        # Initialize and run dashboard
        dashboard = Dashboard()
        dashboard.main()
        
    except Exception as e:
        st.error(f"Error running dashboard: {e}")
        st.exception(e)

if __name__ == "__main__":
    main() 