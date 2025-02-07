import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(root_path))

try:
    # Import core components
    from src.ecs_sehi_analyzer.core.editor.code_editor import CodeEditor
    from src.ecs_sehi_analyzer.core.hologram.hologram_engine import HologramEngine
    from src.ecs_sehi_analyzer.core.studio.screen_studio import ScreenStudio
    from src.ecs_sehi_analyzer.core.accessibility.voice_control import VoiceControl
    from src.ecs_sehi_analyzer.core.visualization.visualizer_4d import DataVisualizer4D
    from src.ecs_sehi_analyzer.core.sustainability import SustainabilityEngine, SustainabilityMetrics
    from src.ecs_sehi_analyzer.core.quantum import QuantumEngine
    from src.ecs_sehi_analyzer.core.ai import ProtocolAssistant
    from src.ecs_sehi_analyzer.core.sandbox import SandboxInterface
except ImportError as e:
    st.error(f"Import Error: {str(e)}")
    st.info("Please ensure all required packages are installed")
    raise

# Initialize components
code_editor = CodeEditor()
hologram_engine = HologramEngine()
screen_studio = ScreenStudio()
voice_control = VoiceControl()
visualizer_4d = DataVisualizer4D()
quantum_engine = QuantumEngine()
protocol_assistant = ProtocolAssistant()
sustainability_metrics = SustainabilityMetrics()
sandbox = SandboxInterface()

# Sidebar configuration
st.sidebar.title("ECS SEHI Analyzer")
st.sidebar.markdown("---")

# Navigation menu in sidebar
menu_options = {
    "run": "🏃‍♂️ Run",
    "collaboration": "👥 Collaboration",
    "sandbox_playground": "🎮 Sandbox Playground",
    "advanced_analytics": "📊 Advanced Analytics",
    "chemical_analysis": "🧪 Chemical Analysis",
    "defect_detection": "🎯 Defect Detection",
    "surface_3d": "🌈 3D Surface"
}

selected_page = st.sidebar.radio("Navigation", list(menu_options.values()))

# Main content area
st.title("Welcome to ECS SEHI Analyzer")

if selected_page == menu_options["run"]:
    st.markdown("""
    Select a module from the sidebar to begin:
    - 📊 Collaboration & Analysis
    - 🧪 Chemical Analysis
    - 🎯 Defect Detection
    - 🌈 3D Surface Visualization
    """)

# Main content tabs
tabs = st.tabs([
    "📝 Research Notes",
    "💻 Code Notebook",
    "📊 Analysis",
    "🌐 Hologram",
    "👥 Collaboration",
    "🎥 Studio",
    "🌱 Sustainability",
    "🔮 Quantum",
    "🤖 AI Assistant",
    "⚙️ Settings"
])

with tabs[0]:  # Research Notes
    st.markdown("### 📝 Research Notes")
    
    # Notes organization
    note_categories = ["General", "Experiments", "Results", "Literature", "Ideas"]
    selected_category = st.selectbox("Note Category", note_categories)
    
    # Initialize session state for notes
    if 'notes' not in st.session_state:
        st.session_state.notes = {cat: "" for cat in note_categories}
    
    # Notes editor
    notes = st.text_area(
        f"{selected_category} Notes",
        value=st.session_state.notes.get(selected_category, ""),
        height=400,
        key=f"notes_{selected_category}",
        help="Enter your research notes here"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Save Notes", key=f"save_{selected_category}"):
            st.session_state.notes[selected_category] = notes
            st.success("Notes saved successfully!")
            
    with col2:
        if st.button("Clear", key=f"clear_{selected_category}"):
            st.session_state.notes[selected_category] = ""
            st.info(f"Cleared {selected_category} notes")
            
    with col3:
        st.download_button(
            "Download Notes",
            notes,
            file_name=f"research_notes_{selected_category.lower()}.txt",
            mime="text/plain",
            key=f"download_{selected_category}"
        )

with tabs[1]:  # Code Notebook
    st.markdown("### 💻 Code Notebook")
    
    # Code notebook tabs
    code_tabs = st.tabs(["Editor", "History", "Documentation"])
    
    with code_tabs[0]:
        code_editor.render_editor()
        
    with code_tabs[1]:
        if 'code_history' not in st.session_state:
            st.session_state.code_history = []
            
        for i, entry in enumerate(reversed(st.session_state.code_history)):
            with st.expander(f"Entry {len(st.session_state.code_history)-i}"):
                st.code(entry, language="python")
                
    with code_tabs[2]:
        st.markdown("""
### Code Documentation
Write your code documentation here using Markdown.

#### Example:
```python
def analyze_data(data):
    \"\"\"
    Analyze experimental data
    
    Parameters:
    - data: numpy array of measurements
    
    Returns:
    - dict of analysis results
    \"\"\"
    pass
```
        """)

with tabs[2]:  # Analysis
    st.markdown("### 📊 Analysis Dashboard")
    
    # Analysis type selector
    analysis_type = st.selectbox(
        "Analysis Type",
        ["Chemical Analysis", "Surface Analysis", "Defect Detection", "Custom Analysis"]
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main visualization area
        if analysis_type == "Chemical Analysis":
            visualizer_4d.render_visualization(None)
        elif analysis_type == "Surface Analysis":
            st.plotly_chart(visualizer_4d.create_surface_plot())
        elif analysis_type == "Defect Detection":
            st.plotly_chart(visualizer_4d.create_defect_map())
        else:
            st.info("Select analysis type to begin")
            
    with col2:
        # Analysis controls
        st.markdown("#### Controls")
        
        with st.expander("Data Selection", expanded=True):
            st.file_uploader("Upload Data", key="analysis_data")
            st.selectbox("Data Source", ["Local", "Remote", "Database"])
            
        with st.expander("Analysis Parameters"):
            st.slider("Resolution", 1, 100, 50)
            st.slider("Sensitivity", 0.0, 1.0, 0.5)
            st.multiselect("Features", ["Height", "Composition", "Defects"])
            
        with st.expander("Export Options"):
            st.download_button(
                "Export Results",
                "analysis_results",
                file_name="analysis_results.csv",
                mime="text/csv"
            )

with tabs[3]:  # Hologram
    col1, col2 = st.columns([7, 3])
    
    with col1:
        st.markdown("### Holographic View")
        hologram_engine.render_hologram(None)
        
    with col2:
        st.markdown("### Detection Settings")
        
        # Detection threshold
        st.markdown("#### Detection Threshold")
        threshold = st.slider("", 0.0, 1.0, 0.8, key="detection_threshold")
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            st.slider("Sensitivity", 0.0, 1.0, 0.7, key="hologram_sensitivity")
            st.slider("Resolution", 1, 100, 50, key="hologram_resolution")
            st.slider("Noise Reduction", 0.0, 1.0, 0.5, key="hologram_noise")
        
        # Detect button
        st.button("Detect Defects", type="primary", key="detect_defects")
        
        # Results
        st.success("✅ Found 52256 potential defects")
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Area", "100.0 μm²")
            st.metric("Defect Ratio", "19.93%")
        with col2:
            st.metric("Avg. Defect Size", "20.0 nm")
        
        # Hologram controls
        st.markdown("### 🎮 Controls")
        st.slider("Quality", 1, 100, 51, key="hologram_quality")
        st.slider("Frame Rate", 1, 100, 30, key="hologram_framerate")

with tabs[4]:  # Collaboration
    st.markdown("### 🤝 Team Collaboration")
    
    # Video Conference Section
    st.markdown("#### 📹 Video Conferencing")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Microsoft Teams")
        st.image("https://upload.wikimedia.org/wikipedia/commons/c/c9/Microsoft_Office_Teams_%282018%E2%80%93present%29.svg", width=50)
        if st.button("Join Teams Meeting"):
            st.link_button("Join Meeting", "https://teams.microsoft.com/")
        st.caption("Current users: 3 online")
        
    with col2:
        st.markdown("##### Zoom")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Zoom_Communications_Logo.svg/200px-Zoom_Communications_Logo.svg.png", width=50)
        meeting_id = st.text_input("Meeting ID", placeholder="Enter Zoom ID")
        if st.button("Join Zoom"):
            st.link_button("Join Meeting", f"https://zoom.us/j/{meeting_id}")
            
    with col3:
        st.markdown("##### Loom")
        st.image("https://upload.wikimedia.org/wikipedia/commons/3/3c/Loom_logo.svg", width=50)
        if st.button("Record Loom"):
            st.link_button("Start Recording", "https://www.loom.com/record")
        st.caption("Quick screen recordings")

    # Document Collaboration
    st.markdown("---")
    st.markdown("#### 📄 Document Collaboration")
    
    doc_col1, doc_col2 = st.columns(2)
    
    with doc_col1:
        st.markdown("##### Notion Workspace")
        with st.expander("Project Documents"):
            st.markdown("""
            - 📑 Project Overview
            - 📊 Analysis Results
            - 📈 Research Data
            - 📝 Team Notes
            """)
        if st.button("Open Notion"):
            st.link_button("Go to Notion", "https://notion.so")
            
    with doc_col2:
        st.markdown("##### Shared Resources")
        uploaded_file = st.file_uploader("Share Document", 
            type=['pdf', 'docx', 'xlsx', 'csv'])
        if uploaded_file is not None:
            st.success(f"Uploaded: {uploaded_file.name}")
            st.download_button(
                "Download File",
                uploaded_file,
                file_name=uploaded_file.name
            )

    # Team Chat
    st.markdown("---")
    st.markdown("#### 💬 Team Chat")
    
    chat_col1, chat_col2 = st.columns([2, 1])
    
    with chat_col1:
        st.text_area("Message", height=100, placeholder="Type your message here...")
        col1, col2 = st.columns([1, 4])
        with col1:
            st.button("Send", type="primary")
        with col2:
            st.button("📎 Attach")
            
    with chat_col2:
        st.markdown("##### Active Team Members")
        st.markdown("""
        - 🟢 John Doe (Online)
        - 🟢 Jane Smith (Online)
        - 🟡 Mike Johnson (Away)
        - ⚪ Sarah Wilson (Offline)
        """)

with tabs[5]:  # Studio tab
    st.markdown("### 🎬 SEHI Studio")
    
    studio_tabs = st.tabs([
        "🎥 Screen Recording",
        "🎙️ Voice Commands",
        "📹 Video Call"
    ])
    
    with studio_tabs[0]:
        screen_studio.render_studio_controls()
        
    with studio_tabs[1]:
        voice_control.render_voice_controls()
        
    with studio_tabs[2]:
        st.markdown("### 📹 Video Conference")
        
        call_col1, call_col2 = st.columns([3, 1])
        
        with call_col1:
            st.camera_input("Camera Feed", key="studio_camera")
            
        with call_col2:
            st.markdown("#### Call Controls")
            st.button("📞 Start Call", key="studio_start_call")
            st.button("🔇 Mute", key="studio_mute")
            st.button("🎥 Toggle Camera", key="studio_toggle_camera")
            
            with st.expander("Call Settings"):
                st.slider("Video Quality", 0, 100, 80, key="studio_video_quality")
                st.selectbox("Audio Input", ["Default Microphone", "Headset"], key="studio_audio_input")
                st.selectbox("Video Input", ["Webcam", "Screen Share"], key="studio_video_input")

with tabs[6]:  # Sustainability tab
    try:
        sustainability = SustainabilityEngine()
        sustainability.render_sustainability_dashboard()
    except Exception as e:
        st.error(f"Error loading sustainability module: {str(e)}")
        st.info("Please ensure all required packages are installed")

with tabs[7]:  # Quantum tab
    quantum_engine.render_quantum_interface()
    
with tabs[8]:  # AI Assistant tab
    protocol_assistant.render_assistant_interface()

with tabs[9]:  # Settings
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Theme", ["Dark", "Light", "System"])
        st.checkbox("Enable Notifications")
    with col2:
        st.selectbox("Language", ["English", "Dutch", "German"])
        st.checkbox("Auto-save")

if selected_page == menu_options["sandbox_playground"]:
    sandbox.render_sandbox() 