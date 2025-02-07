import streamlit as st

class WorkspaceManager:
    def __init__(self):
        self.current_project = None
        self.active_users = []
    
    def create_project(self, name: str):
        self.current_project = name
        
    def join_project(self, project_id: str):
        self.current_project = project_id
        
    def save_workspace(self):
        if self.current_project:
            st.success(f"Workspace {self.current_project} saved!")
        else:
            st.warning("No active project to save") 