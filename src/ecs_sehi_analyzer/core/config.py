import streamlit as st

def setup_page():
    st.set_page_config(
        page_title="ECS SEHI Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    ) 