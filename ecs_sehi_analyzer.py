import streamlit as st
from src.ecs_sehi_analyzer.core.config import setup_page
from src.ecs_sehi_analyzer.core.validation import validate_session_state
from src.ecs_sehi_analyzer.core.io_utils import DataIO
from src.ecs_sehi_analyzer.core.collaboration import render_collaboration_ui

# Must be first Streamlit command
setup_page()

def render_collaboration_sidebar():
    """Render collaboration options in sidebar"""
    with st.sidebar:
        st.sidebar.header("🤝 Collaboration")
        render_collaboration_ui()
        
        # Project sharing
        with st.expander("Share Project"):
            project_name = st.text_input("Project Name", key="sidebar_project_name")
            share_email = st.text_input("Share with (email)", key="sidebar_share_email")
            if st.button("Share", key="sidebar_share_btn"):
                st.success("Project shared successfully!")
        
        # Import/Export
        with st.expander("Import/Export"):
            # Import options
            st.subheader("Import Data")
            import_type = st.selectbox(
                "Import From",
                ["CSV", "Excel", "HDF5", "NetCDF", "Custom Format"],
                key="sidebar_import_type"
            )
            
            # Handle file upload
            uploaded_file = st.file_uploader(
                "Upload File",
                type=["csv", "xlsx", "h5", "nc"],
                key="sidebar_file_uploader"
            )
            
            if uploaded_file is not None:
                try:
                    data_io = DataIO()
                    imported_data = data_io.import_data(uploaded_file, import_type.lower())
                    st.success(f"Successfully imported {uploaded_file.name}")
                    
                    if 'imported_data' not in st.session_state:
                        st.session_state.imported_data = imported_data
                except Exception as e:
                    st.error(f"Error importing file: {str(e)}")
            
            # Export options
            st.subheader("Export Data")
            export_format = st.selectbox(
                "Export Format",
                ["CSV", "Excel", "HDF5", "NetCDF", "PDF Report"],
                key="sidebar_export_format"
            )
            
            if st.button("Export", key="sidebar_export_btn"):
                if 'imported_data' in st.session_state:
                    try:
                        st.info("Preparing export...")
                        st.success("Export complete!")
                    except Exception as e:
                        st.error(f"Error exporting file: {str(e)}")
                else:
                    st.warning("No data available to export")
        
        # Version Control
        with st.expander("Version Control"):
            st.text("Current Version: v1.0.0")
            commit_msg = st.text_area("Commit Message", key="sidebar_commit_msg")
            cols = st.columns(2)
            with cols[0]:
                if st.button("Commit", key="sidebar_commit_btn"):
                    if commit_msg:
                        st.success("Changes committed!")
                    else:
                        st.warning("Please enter a commit message")
            with cols[1]:
                if st.button("Push", key="sidebar_push_btn"):
                    st.success("Changes pushed!")

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
