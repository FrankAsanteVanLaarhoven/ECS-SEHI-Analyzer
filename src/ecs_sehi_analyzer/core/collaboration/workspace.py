import streamlit as st
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
import plotly.graph_objects as go

@dataclass
class WorkspaceTab:
    id: str
    name: str
    type: str
    content: Dict
    collaborators: List[str]
    
class WorkspaceManager:
    def __init__(self):
        if 'workspace_tabs' not in st.session_state:
            st.session_state.workspace_tabs = []
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = None
            
    def create_tab(self, name: str, tab_type: str) -> WorkspaceTab:
        """Create a new workspace tab"""
        tab_id = f"tab_{len(st.session_state.workspace_tabs)}"
        tab = WorkspaceTab(
            id=tab_id,
            name=name,
            type=tab_type,
            content={},
            collaborators=[]
        )
        st.session_state.workspace_tabs.append(tab)
        return tab
    
    def render_workspace(self):
        """Render the collaborative workspace"""
        st.subheader("🔬 Research Workspace")
        
        # Workspace controls
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            tab_name = st.text_input("New Tab Name", key="new_tab_name")
        with col2:
            tab_type = st.selectbox(
                "Tab Type",
                ["Code Editor", "3D Visualization", "Protocol", "Dataset", "Discussion"],
                key="new_tab_type"
            )
        with col3:
            if st.button("Create Tab", key="create_tab_btn"):
                if tab_name:
                    self.create_tab(tab_name, tab_type)
                    st.success(f"Created {tab_type} tab: {tab_name}")
        
        # Render tabs
        if st.session_state.workspace_tabs:
            tabs = st.tabs([tab.name for tab in st.session_state.workspace_tabs])
            
            for i, (tab, tab_content) in enumerate(zip(tabs, st.session_state.workspace_tabs)):
                with tab:
                    self._render_tab_content(tab_content)
        else:
            st.info("Create a new tab to start collaborating!")
    
    def _render_tab_content(self, tab: WorkspaceTab):
        """Render specific tab content"""
        if tab.type == "Code Editor":
            self._render_code_editor(tab)
        elif tab.type == "3D Visualization":
            self._render_3d_tab(tab)
        elif tab.type == "Protocol":
            self._render_protocol_tab(tab)
        elif tab.type == "Dataset":
            self._render_dataset_tab(tab)
        elif tab.type == "Discussion":
            self._render_discussion_tab(tab)
    
    def _render_code_editor(self, tab: WorkspaceTab):
        """Render code editor"""
        st.markdown("### 📝 Code Editor")
        
        # Language selection
        language = st.selectbox(
            "Language",
            ["Python", "R", "Julia", "MATLAB"],
            key=f"lang_{tab.id}"
        )
        
        # Code editor
        code = st.text_area(
            "Code Input",
            value=tab.content.get('code', ''),
            height=300,
            key=f"code_{tab.id}"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Run ▶️", key=f"run_{tab.id}"):
                st.info("Code execution is currently in development")
        with col2:
            st.download_button(
                "Download Script",
                code,
                file_name=f"script.{language.lower()}",
                mime="text/plain",
                key=f"download_{tab.id}"
            )
    
    def _render_3d_tab(self, tab: WorkspaceTab):
        """Render 3D visualization workspace"""
        st.markdown("### 🎮 3D Visualization")
        
        # 3D controls
        col1, col2 = st.columns(2)
        with col1:
            visualization_type = st.selectbox(
                "Visualization Type",
                ["Surface", "Volume", "Point Cloud"],
                key=f"viz_type_{tab.id}"
            )
        with col2:
            color_map = st.selectbox(
                "Color Map",
                ["viridis", "plasma", "inferno"],
                key=f"color_map_{tab.id}"
            )
        
        # Create sample 3D data
        x, y, z = np.mgrid[-2:2:20j, -2:2:20j, -2:2:20j]
        values = np.sin(np.sqrt(x*x + y*y + z*z))
        
        # Create 3D visualization
        fig = go.Figure(data=go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=values.flatten(),
            opacity=0.1,
            surface_count=17,
            colorscale=color_map
        ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_protocol_tab(self, tab: WorkspaceTab):
        """Render protocol management"""
        st.markdown("### 📋 Protocol Manager")
        
        # Protocol steps
        steps = tab.content.get('steps', [])
        
        # Add step
        new_step = st.text_area("New Step", key=f"new_step_{tab.id}")
        if st.button("Add Step", key=f"add_step_{tab.id}"):
            steps.append(new_step)
            tab.content['steps'] = steps
        
        # Display steps
        for i, step in enumerate(steps):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(f"{i+1}. {step}")
            with col2:
                if st.button("✓", key=f"complete_step_{tab.id}_{i}"):
                    st.success(f"Step {i+1} completed!")
    
    def _render_dataset_tab(self, tab: WorkspaceTab):
        """Render dataset visualization and analysis"""
        st.markdown("### 📊 Dataset Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Dataset",
            type=["csv", "xlsx", "h5"],
            key=f"upload_{tab.id}"
        )
        
        if uploaded_file:
            st.success("Dataset uploaded successfully!")
            # Add data processing here
    
    def _render_discussion_tab(self, tab: WorkspaceTab):
        """Render discussion and comments"""
        st.markdown("### 💬 Discussion")
        
        # New comment
        new_comment = st.text_area("New Comment", key=f"comment_{tab.id}")
        if st.button("Post", key=f"post_{tab.id}"):
            if 'comments' not in tab.content:
                tab.content['comments'] = []
            tab.content['comments'].append({
                'text': new_comment,
                'author': 'Current User',
                'timestamp': 'Now'
            })
        
        # Display comments
        if 'comments' in tab.content:
            for comment in tab.content['comments']:
                with st.expander(f"{comment['author']} - {comment['timestamp']}"):
                    st.write(comment['text']) 