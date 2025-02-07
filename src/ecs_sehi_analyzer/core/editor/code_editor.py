import streamlit as st
from typing import Optional
import sys
from io import StringIO
import contextlib

class CodeEditor:
    def __init__(self):
        self.history = []
        
    @contextlib.contextmanager
    def capture_output(self):
        """Capture stdout and stderr"""
        new_out, new_err = StringIO(), StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = new_out, new_err
            yield sys.stdout, sys.stderr
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            
    def execute_code(self, code: str) -> tuple[bool, str]:
        """Execute Python code and return result"""
        try:
            with self.capture_output() as (out, err):
                exec(code)
            output = out.getvalue()
            error = err.getvalue()
            return True, output if output else "Code executed successfully!"
        except Exception as e:
            return False, f"Error: {str(e)}"
            
    def render_editor(self):
        """Render code editor interface"""
        st.markdown("### 💻 Code Editor")
        
        # Editor settings
        with st.expander("Editor Settings"):
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox(
                    "Theme",
                    ["Dark", "Light"],
                    key="editor_theme"
                )
            with col2:
                st.selectbox(
                    "Language",
                    ["Python", "R", "Julia"],
                    key="editor_language"
                )
        
        # Code input
        code = st.text_area(
            "Code",
            height=300,
            key="code_input",
            help="Enter your Python code here"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Run", key="run_code_btn"):
                success, result = self.execute_code(code)
                if success:
                    st.success(result)
                else:
                    st.error(result)
                    
        with col2:
            st.download_button(
                "Download Code",
                code,
                file_name="code.py",
                mime="text/plain",
                key="download_code"
            ) 