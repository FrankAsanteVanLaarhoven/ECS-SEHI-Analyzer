import streamlit as st
import sys
from pathlib import Path

# Add the project root to Python path
root_path = Path(__file__).parent
sys.path.insert(0, str(root_path))

from src.ecs_sehi_analyzer.core.config import setup_page

# Must be first Streamlit command
setup_page()

def main():
    st.sidebar.title("ECS SEHI Analyzer")
    st.sidebar.markdown("---")
    
    # Main page content
    st.title("Welcome to ECS SEHI Analyzer")
    st.markdown("""
    Select a module from the sidebar to begin:
    - 📊 Collaboration & Analysis
    - 🧪 Chemical Analysis
    - 🎯 Defect Detection
    - 🌈 3D Surface Visualization
    """)

if __name__ == "__main__":
    main() 