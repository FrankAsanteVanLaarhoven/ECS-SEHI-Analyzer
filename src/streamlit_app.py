import streamlit as st
from pathlib import Path
import sys

# Add src directory to Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Set page configuration
st.set_page_config(
    page_title="SEHI Analysis Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Application entry point."""
    try:
        from dashboard.dashboard import SEHIDashboard
        from dashboard.styles import inject_styles
        
        # Initialize dashboard
        dashboard = SEHIDashboard()
        
        # Apply styles
        inject_styles()
        
        # Run dashboard
        dashboard.main()
        
    except Exception as e:
        st.error("Critical application error occurred!")
        st.write(f"\nTraceback:")
        st.exception(e)

if __name__ == "__main__":
    main() 