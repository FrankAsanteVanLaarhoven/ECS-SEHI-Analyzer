import streamlit as st
from typing import Optional, Dict
import plotly.graph_objects as go
import json
import numpy as np

class SandboxInterface:
    def __init__(self):
        self.supported_languages = {
            "python": "Python",
            "javascript": "JavaScript",
            "r": "R",
            "julia": "Julia",
            "matlab": "MATLAB",
            "cpp": "C++",
            "java": "Java"
        }
        self.current_output = None
        self.visualization_data = None
        
    def render_sandbox(self):
        """Render the sandbox interface with three-screen layout"""
        st.markdown("### 🎮 Sandbox Playground")
        
        # Create two columns for top screens
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_code_editor()
            
        with col2:
            self._render_visualization_screen()
            
        # Horizontal screen below
        st.markdown("---")
        self._render_output_screen()
        
    def _render_code_editor(self):
        """Render the code editor screen"""
        st.markdown("#### 💻 Code Editor")
        
        # Language selector
        selected_language = st.selectbox(
            "Select Language",
            list(self.supported_languages.values()),
            key="sandbox_language"
        )
        
        # Get language key
        lang_key = [k for k, v in self.supported_languages.items() 
                   if v == selected_language][0]
        
        # Use text area instead of ace editor
        code = st.text_area(
            "Code Editor",
            value=self._get_template(lang_key),
            height=400,
            key="sandbox_editor"
        )
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("▶️ Run", type="primary"):
                self._execute_code(code, lang_key)
        with col2:
            if st.button("🔄 Reset"):
                st.session_state.sandbox_editor = self._get_template(lang_key)
        with col3:
            st.download_button(
                "💾 Save Code",
                code,
                file_name=f"sandbox_code.{lang_key}",
                mime="text/plain"
            )
            
    def _render_visualization_screen(self):
        """Render the visualization screen"""
        st.markdown("#### 📊 Visualization")
        
        # Visualization type selector
        viz_type = st.selectbox(
            "Visualization Type",
            ["Line Plot", "Scatter Plot", "Bar Chart", "3D Surface", "Custom"],
            key="sandbox_viz_type"
        )
        
        # Visualization area
        with st.container(height=400):
            if self.visualization_data:
                try:
                    self._create_visualization(viz_type, self.visualization_data)
                except Exception as e:
                    st.error(f"Visualization error: {str(e)}")
            else:
                st.info("Run your code to see visualization")
                
        # Visualization controls
        with st.expander("Visualization Settings"):
            st.selectbox("Color Theme", ["Default", "Dark", "Light"])
            st.slider("Plot Size", 50, 100, 75)
            st.checkbox("Show Legend")
            
    def _render_output_screen(self):
        """Render the horizontal output screen"""
        st.markdown("#### 📱 Output Console")
        
        # Create tabs for different output views
        output_tabs = st.tabs(["Console", "Data", "Debug", "Documentation"])
        
        with output_tabs[0]:
            if self.current_output:
                st.code(self.current_output)
            else:
                st.info("Code output will appear here")
                
        with output_tabs[1]:
            if self.visualization_data:
                st.json(self.visualization_data)
            else:
                st.info("Data view will appear here")
                
        with output_tabs[2]:
            with st.expander("Debug Information", expanded=True):
                st.markdown("""
                - Memory Usage: 124 MB
                - CPU Usage: 15%
                - Execution Time: 0.45s
                """)
                
        with output_tabs[3]:
            st.markdown(self._get_documentation())
            
    def _get_documentation(self):
        """Get sandbox documentation"""
        return """
        ### Sandbox Documentation
        
        #### Supported Languages
        - Python
        - JavaScript
        - R
        - Julia
        - MATLAB
        - C++
        - Java
        
        #### Data Format
        Output data should be in the format:
        ```python
        {
            "type": "plot_type",
            "data": {
                "x": [...],
                "y": [...],
                "z": [...],  # Optional
                "labels": [...],  # Optional
            }
        }
        ```
        """
        
    def _get_template(self, language: str) -> str:
        """Get code template for selected language"""
        templates = {
            "python": '''
import numpy as np
import json

# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create visualization data
data = {
    "type": "line",
    "data": {
        "x": x.tolist(),
        "y": y.tolist(),
        "labels": ["Time", "Amplitude"]
    }
}

# Print output
print(json.dumps(data))
''',
            "javascript": '''
// Generate sample data
const x = Array.from({length: 100}, (_, i) => i * 0.1);
const y = x.map(val => Math.sin(val));

// Create visualization data
const data = {
    type: "line",
    data: {
        x: x,
        y: y,
        labels: ["Time", "Amplitude"]
    }
};

console.log(JSON.stringify(data));
'''
        }
        return templates.get(language, "// Enter your code here")
        
    def _execute_code(self, code: str, language: str):
        """Execute code and update output"""
        try:
            # Here you would implement actual code execution
            # For now, we'll simulate output
            if language == "python":
                exec_globals = {}
                exec(code, exec_globals)
                self.current_output = "Code executed successfully"
                # Parse output for visualization
                output = exec_globals.get('data', None)
                if output:
                    self.visualization_data = output
            else:
                st.warning(f"Execution for {language} not implemented yet")
        except Exception as e:
            st.error(f"Execution error: {str(e)}")
            
    def _create_visualization(self, viz_type: str, data: Dict):
        """Create visualization based on data"""
        fig = go.Figure()
        
        if viz_type == "Line Plot":
            fig.add_trace(go.Scatter(
                x=data["data"]["x"],
                y=data["data"]["y"],
                mode='lines',
                name=data["data"].get("labels", ["X", "Y"])[1]
            ))
            
        elif viz_type == "Scatter Plot":
            fig.add_trace(go.Scatter(
                x=data["data"]["x"],
                y=data["data"]["y"],
                mode='markers',
                name=data["data"].get("labels", ["X", "Y"])[1]
            ))
            
        elif viz_type == "Bar Chart":
            fig.add_trace(go.Bar(
                x=data["data"]["x"],
                y=data["data"]["y"],
                name=data["data"].get("labels", ["X", "Y"])[1]
            ))
            
        elif viz_type == "3D Surface":
            fig.add_trace(go.Surface(
                z=data["data"]["z"] if "z" in data["data"] else np.random.randn(20, 20),
                x=data["data"]["x"],
                y=data["data"]["y"],
                colorscale="Viridis"
            ))
            
        fig.update_layout(
            title="Output Visualization",
            xaxis_title=data["data"].get("labels", ["X", "Y"])[0],
            yaxis_title=data["data"].get("labels", ["X", "Y"])[1],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True) 