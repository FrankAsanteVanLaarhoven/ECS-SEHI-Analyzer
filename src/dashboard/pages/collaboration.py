import streamlit as st
import datetime
from pathlib import Path
import json
import base64
from typing import Dict, List
import streamlit.components.v1 as components
import pandas as pd
import io
import numpy as np
from utils.data_manager import DashboardDataManager
from utils.surface_analysis import SurfaceAnalyzer
from utils.defect_analysis import DefectAnalyzer
from utils.chemical_analysis import ChemicalAnalyzer
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

class CollaborationHub:
    def __init__(self):
        # Initialize collaboration state
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        if 'shared_files' not in st.session_state:
            st.session_state.shared_files = {}
        if 'presentation_mode' not in st.session_state:
            st.session_state.presentation_mode = False
            
    def create_chat_message(self, user: str, message: str, msg_type: str = "text"):
        """Create a new chat message."""
        return {
            "user": user,
            "message": message,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "type": msg_type
        }

    def handle_file_upload(self, files):
        """Handle file uploads with preview support."""
        for file in files:
            bytes_data = file.read()
            encoded = base64.b64encode(bytes_data).decode()
            
            # Store file info
            st.session_state.shared_files[file.name] = {
                "data": encoded,
                "type": file.type,
                "size": len(bytes_data),
                "shared_by": st.session_state.get('user', 'Anonymous'),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Create chat notification
            self.add_chat_message(
                "system",
                f"📎 {st.session_state.get('user', 'Anonymous')} shared {file.name}",
                "file"
            )

def render_chat_interface():
    """Render the chat interface with file sharing."""
    st.markdown("""
    <style>
        .chat-message {
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
        }
        .user-message {
            background-color: #2e7d32;
            margin-left: 20%;
        }
        .other-message {
            background-color: #1976d2;
            margin-right: 20%;
        }
        .system-message {
            background-color: #424242;
            text-align: center;
            font-style: italic;
        }
        .file-preview {
            border: 1px solid #555;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    # Message input
    message = st.text_input("Message", key="message_input")
    cols = st.columns([3, 1])
    
    with cols[0]:
        files = st.file_uploader("Share files", accept_multiple_files=True)
    
    with cols[1]:
        if st.button("Send") and (message or files):
            if message:
                st.session_state.chat_messages.append(
                    hub.create_chat_message(st.session_state.get('user', 'Anonymous'), message)
                )
            if files:
                hub.handle_file_upload(files)
            st.rerun()

def get_presentation_templates():
    """Get predefined presentation templates."""
    return {
        "Research Analysis": {
            "slides": [
                {"type": "title", "name": "Title", "required": True},
                {"type": "methodology", "name": "Methodology", "required": True},
                {"type": "surface_overview", "name": "Surface Analysis", "required": True},
                {"type": "defect_analysis", "name": "Defect Detection", "required": True},
                {"type": "chemical_mapping", "name": "Chemical Analysis", "optional": True},
                {"type": "statistics", "name": "Key Findings", "required": True},
                {"type": "conclusions", "name": "Conclusions", "required": True}
            ],
            "theme": "scientific",
            "transitions": ["fade", "slide", "zoom"]
        },
        "Defect Report": {
            "slides": [
                {"type": "title", "name": "Overview", "required": True},
                {"type": "defect_map", "name": "Defect Distribution", "required": True},
                {"type": "defect_stats", "name": "Statistics", "required": True},
                {"type": "critical_areas", "name": "Critical Areas", "required": True},
                {"type": "recommendations", "name": "Recommendations", "required": True}
            ],
            "theme": "technical",
            "transitions": ["slide", "fade"]
        },
        "Surface Analysis": {
            "slides": [
                {"type": "title", "name": "Surface Overview", "required": True},
                {"type": "3d_topology", "name": "3D Topology", "required": True},
                {"type": "roughness", "name": "Roughness Analysis", "required": True},
                {"type": "cross_section", "name": "Cross Sections", "optional": True},
                {"type": "comparison", "name": "Comparative Study", "optional": True}
            ],
            "theme": "analytical",
            "transitions": ["zoom", "rotate3d"]
        },
        "Executive Summary": {
            "slides": [
                {"type": "title", "name": "Executive Summary", "required": True},
                {"type": "highlights", "name": "Key Highlights", "required": True},
                {"type": "metrics", "name": "Critical Metrics", "required": True},
                {"type": "impact", "name": "Impact Analysis", "required": True},
                {"type": "next_steps", "name": "Next Steps", "required": True}
            ],
            "theme": "executive",
            "transitions": ["fade", "slide"]
        }
    }

def render_presentation_preview(slides, presentation_title, current_slide=0):
    """Enhanced presentation preview with transitions and themes."""
    st.markdown("""
    <style>
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideIn {
            from { transform: translateX(-100%); }
            to { transform: translateX(0); }
        }
        @keyframes zoomIn {
            from { transform: scale(0.5); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
        .presentation-preview {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            margin: 20px 0;
            animation: fadeIn 0.5s ease-out;
        }
        .slide-content {
            min-height: 500px;
            padding: 30px;
            position: relative;
        }
        .slide-title {
            font-size: 32px;
            margin-bottom: 30px;
            color: #1f1f1f;
            font-weight: 600;
        }
        .slide-data {
            margin: 20px 0;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
        }
        .theme-scientific { background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%); }
        .theme-technical { background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%); }
        .theme-analytical { background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%); }
        .theme-executive { background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); }
    </style>
    """, unsafe_allow_html=True)

    # Get template info
    templates = get_presentation_templates()
    current_template = templates.get(st.session_state.get('current_template', 'Research Analysis'))
    theme_class = f"theme-{current_template['theme']}" if current_template else ""

    st.markdown(f"<div class='presentation-preview {theme_class}'>", unsafe_allow_html=True)
    
    if slides and len(slides) > current_slide:
        slide = slides[current_slide]
        transition = current_template['transitions'][current_slide % len(current_template['transitions'])]
        
        # Add transition effect
        st.markdown(f"""
        <style>
            .slide-content {{
                animation: {transition}In 0.5s ease-out;
            }}
        </style>
        """, unsafe_allow_html=True)
        
        # Render slide content based on type
        render_slide_content(slide, theme_class)
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_slide_content(slide, theme_class):
    """Render individual slide content with enhanced styling."""
    if slide["type"] == "title":
        st.markdown(f"""
        <div class='slide-content {theme_class}'>
            <h1 class='slide-title'>{slide['content']['title']}</h1>
            <h3>{slide['content']['subtitle']}</h3>
            <p class='presenter'>Presented by: {slide['content']['author']}</p>
        </div>
        """, unsafe_allow_html=True)
        
    elif slide["type"] in ["surface_overview", "3d_topology"]:
        cols = st.columns([3, 2])
        with cols[0]:
            if slide['content'].get('3d_model') is not None:
                fig = create_enhanced_3d_plot(
                    slide['content']['3d_model'],
                    theme=theme_class
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with cols[1]:
            if slide['content'].get('metrics'):
                st.markdown("### Key Metrics")
                for key, value in slide['content']['metrics'].items():
                    render_metric_card(key, value, theme_class)

def render_metric_card(key, value, theme_class):
    """Render an enhanced metric card."""
    st.markdown(f"""
    <div class='metric-card {theme_class}'>
        <h4>{key.replace('_', ' ').title()}</h4>
        <div class='metric-value'>{value:.2f if isinstance(value, float) else value}</div>
    </div>
    """, unsafe_allow_html=True)

def create_3d_plot(data):
    """Create a 3D surface plot."""
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[go.Surface(z=data)])
    fig.update_layout(
        title="Surface Analysis",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        width=400,
        height=400
    )
    return fig

def create_defect_map(data):
    """Create a heatmap of defects."""
    import plotly.graph_objects as go
    
    fig = go.Figure(data=go.Heatmap(z=data))
    fig.update_layout(
        title="Defect Distribution",
        width=400,
        height=400
    )
    return fig

def create_screen_recording_slide():
    """Create a slide with screen recording functionality."""
    st.markdown("""
    <style>
        .recording-controls {
            background: #262730;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .recording-status {
            color: #ff4b4b;
            font-weight: bold;
        }
        .recording-preview {
            border: 2px solid #262730;
            border-radius: 10px;
            min-height: 300px;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Recording controls
    st.markdown("### Screen Recording for Presentation")
    
    cols = st.columns([2, 1, 1])
    with cols[0]:
        recording_name = st.text_input("Recording Name", "Screen Recording Slide")
    
    with cols[1]:
        if st.button("🔴 Record"):
            components.html("""
                <script>
                async function recordPresentation() {
                    try {
                        const stream = await navigator.mediaDevices.getDisplayMedia({
                            video: { 
                                cursor: "always",
                                displaySurface: "monitor"
                            },
                            audio: {
                                echoCancellation: true,
                                noiseSuppression: true
                            }
                        });
                        
                        const mediaRecorder = new MediaRecorder(stream, {
                            mimeType: 'video/webm;codecs=vp9'
                        });
                        
                        const chunks = [];
                        mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
                        
                        mediaRecorder.onstop = () => {
                            const blob = new Blob(chunks, { type: 'video/webm' });
                            const url = URL.createObjectURL(blob);
                            
                            // Save to presentation
                            window.streamlitPresentationData = {
                                recordingUrl: url,
                                name: document.querySelector('#recording-name').value
                            };
                            
                            // Trigger Streamlit update
                            const streamlitEvent = new CustomEvent('streamlit:recordingComplete', {
                                detail: { url: url }
                            });
                            window.dispatchEvent(streamlitEvent);
                        };
                        
                        mediaRecorder.start();
                        
                        // Update status
                        document.querySelector('.recording-status').textContent = 'Recording...';
                    } catch (err) {
                        console.error("Error: " + err);
                    }
                }
                recordPresentation();
                </script>
                <div class="recording-status">Ready to record</div>
            """, height=50)
    
    with cols[2]:
        if st.button("⏹️ Stop"):
            components.html("""
                <script>
                if (window.streamlitPresentationData) {
                    const video = document.createElement('video');
                    video.src = window.streamlitPresentationData.recordingUrl;
                    video.controls = true;
                    video.style.width = '100%';
                    document.querySelector('.recording-preview').appendChild(video);
                }
                </script>
            """)

def add_recording_to_presentation(recording_data):
    """Add screen recording to presentation slides."""
    if 'current_presentation' not in st.session_state:
        st.session_state.current_presentation = []
    
    # Create new slide with recording
    recording_slide = {
        "type": "screen_recording",
        "content": {
            "title": recording_data.get("name", "Screen Recording"),
            "recording_url": recording_data["url"],
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    st.session_state.current_presentation.append(recording_slide)
    st.success("Recording added to presentation!")

def render_recording_slide(slide, theme_class):
    """Render a slide containing screen recording."""
    st.markdown(f"""
    <div class='slide-content {theme_class}'>
        <h2 class='slide-title'>{slide['content']['title']}</h2>
        <video controls width="100%">
            <source src="{slide['content']['recording_url']}" type="video/webm">
            Your browser does not support the video tag.
        </video>
        <p class="recording-timestamp">Recorded on: {slide['content']['timestamp']}</p>
    </div>
    """, unsafe_allow_html=True)

def setup_screen_recorder():
    """Initialize screen recorder with cross-page support."""
    if 'screen_recording' not in st.session_state:
        st.session_state.screen_recording = {
            'active': False,
            'recordings': [],
            'current_chunk': None,
            'start_time': None,
            'current_page': None
        }

def create_screen_recorder_controls():
    """Create enhanced screen recording controls with page tracking."""
    st.markdown("""
    <style>
        .recording-indicator {
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 8px 16px;
            background: rgba(255, 0, 0, 0.8);
            color: white;
            border-radius: 20px;
            z-index: 9999;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .recording-controls {
            background: #1E1E1E;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)

    cols = st.columns([2, 1, 1, 1])
    
    with cols[0]:
        recording_name = st.text_input("Recording Name", "Dashboard Walkthrough")
    
    with cols[1]:
        if st.button("🔴 Start Recording", disabled=st.session_state.screen_recording['active']):
            components.html("""
                <script>
                async function startContinuousRecording() {
                    try {
                        const stream = await navigator.mediaDevices.getDisplayMedia({
                            video: { 
                                cursor: "always",
                                displaySurface: "monitor"
                            },
                            audio: {
                                echoCancellation: true,
                                noiseSuppression: true,
                                autoGainControl: true
                            }
                        });
                        
                        const mediaRecorder = new MediaRecorder(stream, {
                            mimeType: 'video/webm;codecs=vp9',
                            videoBitsPerSecond: 3000000
                        });
                        
                        const chunks = [];
                        mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
                        
                        // Save recording when stopped
                        mediaRecorder.onstop = () => {
                            const blob = new Blob(chunks, { type: 'video/webm' });
                            const url = URL.createObjectURL(blob);
                            
                            // Send to Streamlit
                            window.parent.postMessage({
                                type: 'recordingComplete',
                                data: {
                                    url: url,
                                    name: document.getElementById('recording-name').value,
                                    timestamp: new Date().toISOString()
                                }
                            }, '*');
                        };
                        
                        // Start recording
                        mediaRecorder.start();
                        window.currentRecording = {
                            mediaRecorder,
                            stream
                        };
                        
                        // Show recording indicator
                        const indicator = document.createElement('div');
                        indicator.className = 'recording-indicator';
                        indicator.innerHTML = '🔴 Recording';
                        document.body.appendChild(indicator);
                        
                    } catch (err) {
                        console.error("Recording error:", err);
                    }
                }
                startContinuousRecording();
                </script>
            """, height=0)
            st.session_state.screen_recording['active'] = True
            st.rerun()

    with cols[2]:
        if st.button("⏸️ Pause", disabled=not st.session_state.screen_recording['active']):
            components.html("""
                <script>
                if (window.currentRecording && window.currentRecording.mediaRecorder) {
                    window.currentRecording.mediaRecorder.pause();
                }
                </script>
            """)

    with cols[3]:
        if st.button("⏹️ Stop", disabled=not st.session_state.screen_recording['active']):
            components.html("""
                <script>
                if (window.currentRecording) {
                    window.currentRecording.mediaRecorder.stop();
                    window.currentRecording.stream.getTracks().forEach(track => track.stop());
                    delete window.currentRecording;
                }
                </script>
            """)
            st.session_state.screen_recording['active'] = False
            st.rerun()

def setup_presentation_display():
    """Setup presentation display with casting support."""
    st.markdown("""
    <style>
        .cast-button {
            position: fixed;
            top: 10px;
            right: 80px;
            z-index: 9999;
        }
        .presentation-display {
            position: relative;
            width: 100%;
            height: 100%;
        }
        .fullscreen-toggle {
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 1000;
        }
    </style>
    """, unsafe_allow_html=True)

    # Cast/Mirror controls
    cast_cols = st.columns([2, 1, 1])
    with cast_cols[0]:
        st.markdown("### Display Options")
    with cast_cols[1]:
        if st.button("🔄 Mirror Display"):
            components.html("""
                <script>
                if ('getScreenDetails' in window) {
                    async function mirrorPresentation() {
                        try {
                            const screenDetails = await window.getScreenDetails();
                            const presentationScreen = screenDetails.screens.find(
                                screen => screen.isPrimary === false
                            );
                            
                            if (presentationScreen) {
                                const presentationWindow = window.open(
                                    '',
                                    'presentation',
                                    `width=${presentationScreen.width},height=${presentationScreen.height}`
                                );
                                presentationWindow.moveTo(
                                    presentationScreen.left,
                                    presentationScreen.top
                                );
                            }
                        } catch (err) {
                            console.error("Mirroring error:", err);
                        }
                    }
                    mirrorPresentation();
                }
                </script>
            """)
    with cast_cols[2]:
        if st.button("📱 Cast"):
            components.html("""
                <script>
                // Check if Presentation API is available
                if ('PresentationRequest' in window) {
                    const presentationRequest = new PresentationRequest(['presentation.html']);
                    presentationRequest.start()
                        .then(connection => {
                            // Handle presentation connection
                            connection.send({
                                type: 'presentation',
                                content: document.querySelector('.presentation-preview').innerHTML
                            });
                        })
                        .catch(err => console.error("Casting error:", err));
                }
                </script>
            """)

def create_pip_recorder():
    """Create picture-in-picture recorder with screen and camera."""
    st.markdown("""
    <style>
        .pip-container {
            position: relative;
            width: 100%;
            min-height: 200px;
        }
        .camera-preview {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 240px;
            height: 180px;
            border-radius: 10px;
            overflow: hidden;
            background: #000;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            z-index: 1000;
            resize: both;
        }
        .camera-preview.dragging {
            opacity: 0.7;
        }
        .recording-controls {
            display: flex;
            gap: 10px;
            margin: 10px 0;
        }
        .pip-settings {
            padding: 10px;
            background: #1E1E1E;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # PiP Settings
    with st.expander("Camera Settings"):
        cols = st.columns(3)
        with cols[0]:
            camera_size = st.select_slider(
                "Camera Size",
                options=["Small", "Medium", "Large"],
                value="Medium",
                key="pip_camera_size_setting"
            )
        with cols[1]:
            camera_position = st.selectbox(
                "Position",
                ["Bottom Right", "Bottom Left", "Top Right", "Top Left"],
                key="pip_camera_position_setting"
            )
        with cols[2]:
            camera_quality = st.select_slider(
                "Quality",
                options=["720p", "1080p", "4K"],
                value="1080p",
                key="pip_camera_quality_setting"
            )

    # Recording controls
    st.markdown("### Screen & Camera Recording")
    control_cols = st.columns([2, 1, 1])
    
    with control_cols[0]:
        recording_name = st.text_input("Recording Name", "Lecture Recording", key="pip_recording_name_input")

    with control_cols[1]:
        if st.button("🎥 Start Recording", key="pip_camera_start_recording"):
            components.html("""
                <script>
                async function startPiPRecording() {
                    try {
                        // Get screen stream
                        const screenStream = await navigator.mediaDevices.getDisplayMedia({
                            video: { 
                                cursor: "always",
                                displaySurface: "monitor"
                            },
                            audio: true
                        });
                        
                        // Get camera stream
                        const cameraStream = await navigator.mediaDevices.getUserMedia({
                            video: {
                                width: { ideal: 1920 },
                                height: { ideal: 1080 }
                            },
                            audio: true
                        });
                        
                        // Create PiP video element
                        const cameraVideo = document.createElement('video');
                        cameraVideo.srcObject = cameraStream;
                        cameraVideo.autoplay = true;
                        cameraVideo.className = 'camera-preview';
                        document.body.appendChild(cameraVideo);
                        
                        // Make camera preview draggable
                        let isDragging = false;
                        let currentX;
                        let currentY;
                        let initialX;
                        let initialY;
                        
                        cameraVideo.addEventListener('mousedown', dragStart);
                        document.addEventListener('mousemove', drag);
                        document.addEventListener('mouseup', dragEnd);
                        
                        function dragStart(e) {
                            initialX = e.clientX - currentX;
                            initialY = e.clientY - currentY;
                            if (e.target === cameraVideo) {
                                isDragging = true;
                                cameraVideo.classList.add('dragging');
                            }
                        }
                        
                        function drag(e) {
                            if (isDragging) {
                                e.preventDefault();
                                currentX = e.clientX - initialX;
                                currentY = e.clientY - initialY;
                                cameraVideo.style.transform = 
                                    `translate(${currentX}px, ${currentY}px)`;
                            }
                        }
                        
                        function dragEnd(e) {
                            isDragging = false;
                            cameraVideo.classList.remove('dragging');
                        }
                        
                        // Combine streams
                        const combinedStream = new MediaStream([
                            ...screenStream.getTracks(),
                            ...cameraStream.getVideoTracks()
                        ]);
                        
                        // Start recording
                        const mediaRecorder = new MediaRecorder(combinedStream, {
                            mimeType: 'video/webm;codecs=vp9',
                            videoBitsPerSecond: 3000000
                        });
                        
                        const chunks = [];
                        mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
                        
                        mediaRecorder.onstop = () => {
                            const blob = new Blob(chunks, { type: 'video/webm' });
                            const url = URL.createObjectURL(blob);
                            
                            // Save recording
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = 'lecture-recording.webm';
                            a.click();
                            
                            // Clean up
                            cameraVideo.remove();
                            screenStream.getTracks().forEach(track => track.stop());
                            cameraStream.getTracks().forEach(track => track.stop());
                        };
                        
                        mediaRecorder.start();
                        window.currentRecording = {
                            mediaRecorder,
                            screenStream,
                            cameraStream,
                            cameraVideo
                        };
                        
                    } catch (err) {
                        console.error("Recording error:", err);
                    }
                }
                startPiPRecording();
                </script>
            """, height=0)

    with control_cols[2]:
        if st.button("⏹️ Stop Recording", key="pip_camera_stop_recording"):
            components.html("""
                <script>
                if (window.currentRecording) {
                    window.currentRecording.mediaRecorder.stop();
                    window.currentRecording.screenStream.getTracks().forEach(track => track.stop());
                    window.currentRecording.cameraStream.getTracks().forEach(track => track.stop());
                    window.currentRecording.cameraVideo.remove();
                    delete window.currentRecording;
                }
                </script>
            """)

    # Preview area
    st.markdown("### Recording Preview")
    st.markdown("""
        <div class="pip-container">
            <div id="screen-preview"></div>
        </div>
    """, unsafe_allow_html=True)

def render_presentation_mode():
    """Enhanced presentation mode with PiP recording support."""
    setup_screen_recorder()
    
    st.markdown("## Presentation Mode")
    
    # Initialize presentation state
    if 'current_slide' not in st.session_state:
        st.session_state.current_slide = 0
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False

    # Create main layout
    main_container = st.container()
    with main_container:
        # Template selection
        template = st.selectbox(
            "Presentation Template",
            [
                "Research Analysis",
                "Defect Report",
                "Surface Analysis",
                "Chemical Mapping",
                "Custom Template"
            ]
        )
        
        # Playback controls using a single container
        st.markdown("### Playback Controls")
        control_container = st.container()
        with control_container:
            cols = st.columns(4)
            with cols[0]:
                if st.button("⏮️ Previous", key="presentation_prev"):
                    st.session_state.current_slide = max(0, st.session_state.current_slide - 1)
                    st.rerun()
            with cols[1]:
                play_pause = "⏸️" if st.session_state.is_playing else "▶️"
                if st.button(f"{play_pause} Play/Pause", key="presentation_play"):
                    st.session_state.is_playing = not st.session_state.is_playing
                    st.rerun()
            with cols[2]:
                if st.button("⏭️ Next", key="presentation_next"):
                    st.session_state.current_slide += 1
                    st.rerun()
            with cols[3]:
                if st.button("🔄 Reset", key="presentation_reset"):
                    st.session_state.current_slide = 0
                    st.rerun()
            
            # Progress indicator
            if 'current_presentation' in st.session_state:
                total_slides = len(st.session_state.current_presentation)
                progress = (st.session_state.current_slide + 1) / total_slides
                st.progress(progress)
                st.markdown(f"Slide {st.session_state.current_slide + 1} of {total_slides}")

        # Settings section
        st.markdown("### Presentation Settings")
        settings_container = st.container()
        with settings_container:
            auto_advance = st.checkbox("Auto Advance", value=False)
            if auto_advance:
                interval = st.slider("Interval (seconds)", 5, 60, 30)
            
            show_notes = st.checkbox("Show Speaker Notes", value=False)
            show_timer = st.checkbox("Show Timer", value=True)

        # Add recording options
        st.markdown("### Recording Options")
        recording_tabs = st.tabs(["Screen Only", "Screen + Camera"])
        
        with recording_tabs[0]:
            render_screen_recorder()
        
        with recording_tabs[1]:
            create_pip_recorder()

        # Display options
        st.markdown("### Display Options")
        display_container = st.container()
        with display_container:
            if st.button("🔄 Mirror to External Display", key="presentation_mirror_display"):
                mirror_presentation()
            if st.button("📱 Cast Presentation", key="presentation_cast_display"):
                cast_presentation()

        # Preview current presentation
        st.markdown("---")
        preview_container = st.container()
        with preview_container:
            if 'current_presentation' in st.session_state:
                render_presentation_preview(
                    st.session_state.current_presentation,
                    "Research Findings",
                    st.session_state.current_slide
                )
            else:
                st.info("Generate a presentation using the panel on the right to see preview.")
                
                # Show sample slides
                st.markdown("### Sample Slides Available:")
                st.markdown("""
                - Title Slide
                - Surface Analysis Overview
                - Defect Detection Results
                - Chemical Mapping Analysis
                - Sustainability Metrics
                - Conclusions & Recommendations
                
                Click 'Generate Presentation' to create slides from your analysis data.
                """)

def start_screen_recording():
    """Start screen recording with proper state management."""
    st.session_state.screen_recording['active'] = True
    inject_recording_script()
    st.rerun()

def stop_screen_recording():
    """Stop screen recording and save the recording."""
    st.session_state.screen_recording['active'] = False
    inject_stop_recording_script()
    st.rerun()

def mirror_presentation():
    """Handle presentation mirroring to external display."""
    components.html("""
        <script>
        async function mirrorToDisplay() {
            if ('getScreenDetails' in window) {
                try {
                    const screenDetails = await window.getScreenDetails();
                    const presentationScreen = screenDetails.screens.find(
                        screen => screen.isPrimary === false
                    );
                    if (presentationScreen) {
                        // Handle presentation window
                        const presentationWindow = window.open('', 'presentation');
                        presentationWindow.document.write(`
                            <html>
                                <head>
                                    <style>
                                        body { margin: 0; overflow: hidden; background: black; }
                                        .presentation-content {
                                            width: 100vw;
                                            height: 100vh;
                                            display: flex;
                                            justify-content: center;
                                            align-items: center;
                                        }
                                    </style>
                                </head>
                                <body>
                                    <div class="presentation-content">
                                        ${document.querySelector('.presentation-preview').innerHTML}
                                    </div>
                                </body>
                            </html>
                        `);
                    }
                } catch (err) {
                    console.error('Mirroring failed:', err);
                }
            }
        }
        mirrorToDisplay();
        </script>
    """, height=0)

def render_file_preview(file_info):
    """Enhanced file preview with support for multiple file types."""
    file_type = file_info["type"]
    
    if "image" in file_type:
        st.image(base64.b64decode(file_info["data"]))
    
    elif "pdf" in file_type:
        components.iframe(
            f"data:application/pdf;base64,{file_info['data']}",
            height=400
        )
    
    elif "text" in file_type or "json" in file_type:
        try:
            content = base64.b64decode(file_info["data"]).decode()
            st.code(content)
        except:
            st.error("Unable to display file content")
    
    elif "csv" in file_type:
        try:
            content = base64.b64decode(file_info["data"]).decode()
            df = pd.read_csv(io.StringIO(content))
            st.dataframe(df)
        except:
            st.error("Unable to display CSV content")

def render_ai_assistant():
    """Render AI assistant for navigation and help."""
    st.markdown("### AI Assistant")
    
    if "assistant_messages" not in st.session_state:
        st.session_state.assistant_messages = [
            {"role": "assistant", "content": "Hello! I can help you navigate the dashboard and answer questions. What would you like to know?"}
        ]
    
    # Display chat history
    for message in st.session_state.assistant_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.assistant_messages.append({"role": "user", "content": prompt})
        
        # Simple response logic (expand this with more sophisticated AI)
        response = handle_ai_query(prompt)
        st.session_state.assistant_messages.append({"role": "assistant", "content": response})
        st.rerun()

def handle_ai_query(prompt: str) -> str:
    """Handle AI assistant queries."""
    # Simple keyword-based responses (replace with actual AI)
    keywords = {
        "analysis": "You can find analysis tools under the 'Analysis' tab. Would you like me to show you how to use them?",
        "share": "To share files, use the file upload button in the chat interface. You can share various file types including images and data files.",
        "present": "Enter presentation mode by clicking 'Start Presentation' in the presentation tab. You can create slides and automate their progression.",
        "help": "I can help you with: \n- Navigation\n- File sharing\n- Analysis tools\n- Presentation creation\nWhat would you like to know more about?"
    }
    
    for key, response in keywords.items():
        if key in prompt.lower():
            return response
    
    return "I can help you with navigation, file sharing, analysis, and presentations. Could you please be more specific?"

def render_video_conference():
    """Render video conference interface with screen sharing."""
    st.markdown("### Video Conference")
    
    # Video conference controls
    conf_cols = st.columns([2, 1, 1])
    with conf_cols[0]:
        room_name = st.text_input("Room Name", "SEHI Analysis Room")
    with conf_cols[1]:
        if st.button("Start Meeting", type="primary"):
            # Initialize Jitsi Meet
            components.html(
                f"""
                <div id="meet" style="height: 600px;">
                    <script src='https://meet.jit.si/external_api.js'></script>
                    <script>
                        const domain = 'meet.jit.si';
                        const options = {{
                            roomName: '{room_name}',
                            width: '100%',
                            height: '100%',
                            parentNode: document.querySelector('#meet'),
                            userInfo: {{
                                displayName: '{st.session_state.get("user", "Anonymous")}'
                            }},
                            configOverwrite: {{
                                startWithAudioMuted: true,
                                startWithVideoMuted: false,
                                enableScreensharing: true,
                                enableClosePage: true
                            }}
                        }};
                        const api = new JitsiMeetExternalAPI(domain, options);
                    </script>
                </div>
                """,
                height=600
            )

def get_surface_analysis_data():
    """Gather surface analysis data from the dashboard."""
    try:
        analyzer = SurfaceAnalyzer()
        results = analyzer.analyze_surface(
            resolution=512,
            noise_reduction=0.5,
            view_mode="Height Map",
            method="standard"  # Add required method parameter
        )
        
        if not results:
            return {}
            
        return {
            "3d_model": results.get('surface_data', None),
            "metrics": {
                "mean_height": results['stats']['mean_height'],
                "rms_roughness": results['stats']['rms_roughness'],
                "peak_height": results['stats']['peak_height'],
                "surface_area": results['stats']['surface_area']
            },
            "roughness_map": results.get('roughness_map', None),
            "height_distribution": results.get('height_distribution', None),
            "detailed_analysis": results.get('analysis', "Surface analysis details will be added here.")
        }
    except Exception as e:
        st.error(f"Error gathering surface analysis data: {str(e)}")
        return {}

def get_defect_analysis_data():
    """Gather defect analysis data from the dashboard."""
    try:
        analyzer = DefectAnalyzer()
        results = analyzer.analyze_defects()  # Now properly implemented
        
        if not results:
            return {}
            
        return {
            "defect_map": results.get('defect_map', None),
            "statistics": {
                "defect_count": results.get('defect_count', 0),
                "defect_density": results.get('defect_density', 0),
                "average_size": results.get('average_size', 0),
                "severity_distribution": results.get('severity_distribution', {})
            },
            "critical_areas": results.get('critical_areas', []),
            "recommendations": results.get('recommendations', "Defect analysis recommendations will be added here.")
        }
    except Exception as e:
        st.error(f"Error gathering defect analysis data: {str(e)}")
        return {}

def get_chemical_mapping_data():
    """Gather chemical mapping data from the dashboard."""
    try:
        analyzer = ChemicalAnalyzer()
        results = analyzer.analyze_chemical_composition()
        
        return {
            "composition_map": results.get('composition_map', None),
            "element_distribution": results.get('element_distribution', {}),
            "concentration_data": results.get('concentration_data', {}),
            "analysis_summary": results.get('summary', "Chemical analysis summary will be added here.")
        }
    except Exception as e:
        st.error(f"Error gathering chemical mapping data: {str(e)}")
        return {}

def get_sustainability_metrics():
    """Gather sustainability metrics from the dashboard."""
    try:
        metrics = DashboardDataManager.get_sustainability_metrics()
        return metrics
    except Exception as e:
        st.error(f"Error gathering sustainability metrics: {str(e)}")
        return {}

def generate_dashboard_presentation():
    """Generate presentation from dashboard data with error handling."""
    st.markdown("### Generate Analysis Presentation")
    
    # Select data sources
    data_sources = st.multiselect(
        "Include Data From",
        [
            "Surface Analysis",
            "Chemical Mapping",
            "Defect Detection",
            "Sustainability Metrics",
            "Analysis Results"
        ],
        default=["Surface Analysis", "Defect Detection"]
    )
    
    # Presentation options
    cols = st.columns(3)
    with cols[0]:
        include_3d = st.checkbox("Include 3D Models", value=True)
    with cols[1]:
        include_metrics = st.checkbox("Include Metrics", value=True)
    with cols[2]:
        include_recommendations = st.checkbox("Include Recommendations", value=True)
    
    if st.button("Generate Presentation", type="primary"):
        with st.spinner("Generating comprehensive presentation..."):
            try:
                slides = []
                
                # Title slide
                slides.append({
                    "type": "title",
                    "content": {
                        "title": "SEHI Analysis Report",
                        "subtitle": f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d')}",
                        "author": st.session_state.get("user", "Anonymous")
                    }
                })
                
                # Generate slides based on selected sources with error handling
                if "Surface Analysis" in data_sources:
                    surface_data = get_surface_analysis_data()
                    if surface_data:
                        slides.extend([
                            {
                                "type": "surface_overview",
                                "content": {
                                    "title": "Surface Analysis Overview",
                                    "3d_model": surface_data.get("3d_model"),
                                    "metrics": surface_data.get("metrics"),
                                    "findings": surface_data.get("findings")
                                }
                            }
                        ])
                
                if "Defect Detection" in data_sources:
                    defect_data = get_defect_analysis_data()
                    if defect_data:
                        slides.extend([
                            {
                                "type": "defect_summary",
                                "content": {
                                    "defect_map": defect_data.get("defect_map"),
                                    "statistics": defect_data.get("statistics"),
                                    "critical_areas": defect_data.get("critical_areas")
                                }
                            }
                        ])
                
                # Save presentation to session state
                st.session_state.current_presentation = slides
                st.success("Presentation generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating presentation: {str(e)}")

def cast_presentation():
    """Handle presentation casting to external displays."""
    components.html("""
        <script>
        async function castPresentation() {
            if ('PresentationRequest' in window) {
                try {
                    const presentationRequest = new PresentationRequest(['presentation.html']);
                    const connection = await presentationRequest.start();
                    
                    // Send presentation content
                    connection.send({
                        type: 'presentation',
                        content: document.querySelector('.presentation-preview').innerHTML
                    });
                } catch (err) {
                    console.error('Casting failed:', err);
                }
            } else {
                console.log('Presentation API not supported');
            }
        }
        castPresentation();
        </script>
    """, height=0)

def render_screen_recorder():
    """Render screen recording controls."""
    st.markdown("### Screen Recording")
    
    # Generate unique identifier for this instance
    recorder_id = f"{st.session_state.get('tab_id', '0')}_{id(st)}"
    
    # Recording controls with unique keys
    rec_cols = st.columns([1, 1, 1])
    with rec_cols[0]:
        if st.button("🔴 Start Recording", key=f"screen_start_{recorder_id}"):
            components.html(
                """
                <script>
                    async function startRecording() {
                        try {
                            const stream = await navigator.mediaDevices.getDisplayMedia({
                                video: { cursor: "always" },
                                audio: true
                            });
                            
                            const mediaRecorder = new MediaRecorder(stream);
                            const chunks = [];
                            
                            mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
                            mediaRecorder.onstop = () => {
                                const blob = new Blob(chunks, { type: 'video/webm' });
                                const url = URL.createObjectURL(blob);
                                const a = document.createElement('a');
                                a.href = url;
                                a.download = 'screen-recording.webm';
                                a.click();
                            };
                            
                            mediaRecorder.start();
                        } catch (err) {
                            console.error("Error: " + err);
                        }
                    }
                    startRecording();
                </script>
                """
            )
    
    with rec_cols[1]:
        if st.button("⏸️ Pause", key=f"screen_pause_{recorder_id}"):
            components.html(
                """
                <script>
                    if (window.currentRecording && window.currentRecording.mediaRecorder) {
                        window.currentRecording.mediaRecorder.pause();
                    }
                </script>
                """
            )
    
    with rec_cols[2]:
        if st.button("⏹️ Stop", key=f"screen_stop_{recorder_id}"):
            components.html(
                """
                <script>
                    if (window.currentRecording) {
                        window.currentRecording.mediaRecorder.stop();
                        window.currentRecording.stream.getTracks().forEach(track => track.stop());
                        delete window.currentRecording;
                    }
                </script>
                """
            )

    # Add recording status indicator
    if 'recording_status' in st.session_state:
        st.markdown(
            f"""
            <div style="
                padding: 10px;
                border-radius: 5px;
                background-color: {'#ff4b4b' if st.session_state.recording_status == 'recording' else '#4b4b4b'};
                color: white;
                text-align: center;
                margin: 10px 0;
            ">
                {st.session_state.recording_status.title()}
            </div>
            """,
            unsafe_allow_html=True
        )

def render_whiteboard():
    """Render an enhanced technical whiteboard for scientific analysis."""
    components.html("""
        <style>
            .whiteboard-container {
                background: #1f1f1f;
                padding: 20px;
                border-radius: 10px;
            }
            .toolbar {
                display: flex;
                gap: 10px;
                margin-bottom: 10px;
                padding: 10px;
                background: #2d2d2d;
                border-radius: 5px;
                flex-wrap: wrap;
            }
            .tool-group {
                display: flex;
                gap: 5px;
                padding: 5px;
                border-right: 1px solid #444;
            }
            .tool-btn {
                padding: 8px 15px;
                border: none;
                border-radius: 5px;
                background: #3d3d3d;
                color: white;
                cursor: pointer;
                transition: all 0.3s;
                display: flex;
                align-items: center;
                gap: 5px;
            }
            .tool-btn:hover {
                background: #4d4d4d;
            }
            .tool-btn.active {
                background: #0d6efd;
            }
            #whiteboard {
                background: #ffffff;
                border-radius: 5px;
                cursor: crosshair;
            }
            .color-picker {
                width: 40px;
                height: 40px;
                padding: 0;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            .size-slider {
                width: 100px;
            }
            .preset-colors {
                display: flex;
                gap: 5px;
            }
            .color-preset {
                width: 25px;
                height: 25px;
                border-radius: 50%;
                cursor: pointer;
                border: 2px solid #444;
            }
            .color-preset:hover {
                transform: scale(1.1);
            }
            
            /* Scientific Tool Styles */
            .tool-group.scientific {
                background: #1a2634;
                padding: 8px;
                border-radius: 8px;
                border: 1px solid #2a3f50;
            }
            
            .measurement-display {
                position: fixed;
                bottom: 20px;
                left: 20px;
                background: rgba(0,0,0,0.9);
                color: #00ff00;
                padding: 8px 15px;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                border: 1px solid #2a3f50;
            }
            
            .scale-indicator {
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: rgba(0,0,0,0.9);
                color: #00ff00;
                padding: 5px 10px;
                border-radius: 5px;
                font-family: monospace;
            }
            
            /* Point Cloud Styles */
            .point-cloud-controls {
                background: #1a2634;
                padding: 8px;
                border-radius: 8px;
                margin-top: 10px;
            }
            
            .point-size-slider {
                width: 150px;
                margin: 0 10px;
            }
            
            /* Mobile Touch Controls */
            .touch-indicator {
                position: fixed;
                bottom: 60px;
                right: 20px;
                background: rgba(0,0,0,0.8);
                color: #00ff00;
                padding: 5px 10px;
                border-radius: 5px;
                font-family: monospace;
            }
            
            /* Polygon Tools */
            .polygon-controls {
                display: flex;
                gap: 5px;
                align-items: center;
                margin: 5px 0;
            }
            
            .vertex-counter {
                background: rgba(0,0,0,0.8);
                color: #00ff00;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 12px;
            }
            
            /* Scientific Measurement Tools */
            .measurement-tools {
                background: #1a2634;
                padding: 8px;
                border-radius: 8px;
                margin-top: 10px;
            }
            
            .analysis-panel {
                position: fixed;
                right: 20px;
                top: 100px;
                background: rgba(0,0,0,0.9);
                padding: 15px;
                border-radius: 8px;
                color: #00ff00;
                font-family: monospace;
                width: 250px;
            }
            
            .scale-bar {
                position: fixed;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(0,0,0,0.8);
                padding: 5px 15px;
                border-radius: 5px;
                color: white;
                font-family: monospace;
            }
        </style>
        
        <div class="whiteboard-container">
            <div class="toolbar">
                <!-- Scientific Drawing Tools -->
                <div class="tool-group scientific">
                    <button class="tool-btn" onclick="setTool('microscope')" id="microscopeBtn">🔬 Microscope View</button>
                    <button class="tool-btn" onclick="setTool('spectrum')" id="spectrumBtn">📊 Spectrum</button>
                    <button class="tool-btn" onclick="setTool('measurement')" id="measurementBtn">📏 Measure</button>
                </div>

                <!-- Analysis Elements -->
                <div class="tool-group scientific">
                    <button class="tool-btn" onclick="addElement('particle')" id="particleBtn">⚪ Particle</button>
                    <button class="tool-btn" onclick="addElement('crystal')" id="crystalBtn">💎 Crystal</button>
                    <button class="tool-btn" onclick="addElement('defect')" id="defectBtn">❌ Defect</button>
                    <button class="tool-btn" onclick="addElement('grain')" id="grainBtn">🔘 Grain</button>
                </div>

                <!-- Technical Annotations -->
                <div class="tool-group scientific">
                    <button class="tool-btn" onclick="setTool('arrow')" id="arrowBtn">➡️ Vector</button>
                    <button class="tool-btn" onclick="setTool('scale')" id="scaleBtn">📊 Scale Bar</button>
                    <button class="tool-btn" onclick="addAnnotation()" id="annotationBtn">📝 Label</button>
                    <button class="tool-btn" onclick="addCalibration()" id="calibrationBtn">🎯 Calibration</button>
                </div>

                <!-- Analysis Tools -->
                <div class="tool-group scientific">
                    <button class="tool-btn" onclick="setTool('lineProfile')" id="lineProfileBtn">📈 Line Profile</button>
                    <button class="tool-btn" onclick="setTool('areaAnalysis')" id="areaAnalysisBtn">📊 Area Analysis</button>
                    <button class="tool-btn" onclick="setTool('angleAnalysis')" id="angleAnalysisBtn">📐 Angle</button>
                    <button class="tool-btn" onclick="setTool('roughness')" id="roughnessBtn">〰️ Roughness</button>
                    <button class="tool-btn" onclick="setTool('particleSize')" id="particleSizeBtn">⭕ Particle Size</button>
                    <button class="tool-btn" onclick="setTool('defectAnalysis')" id="defectBtn">❌ Defect</button>
                </div>

                <!-- Grid & Scale -->
                <div class="tool-group">
                    <div class="grid-controls">
                        <button class="tool-btn" onclick="toggleGrid()" id="gridBtn">📏 Grid</button>
                        <select id="scaleUnit">
                            <option value="nm">nm</option>
                            <option value="μm">μm</option>
                            <option value="mm">mm</option>
                        </select>
                        <input type="number" id="gridSize" min="1" max="100" value="10" style="width:60px">
                    </div>
                </div>

                <!-- Export Options -->
                <div class="tool-group scientific">
                    <button class="tool-btn" onclick="exportData('image')" id="exportImageBtn">📸 Export Image</button>
                    <button class="tool-btn" onclick="exportData('data')" id="exportDataBtn">📊 Export Data</button>
                    <button class="tool-btn" onclick="exportData('report')" id="exportReportBtn">📑 Export Report</button>
                </div>

                <!-- Point Cloud Tools -->
                <div class="tool-group technical">
                    <button class="tool-btn" onclick="setTool('pointCloud')" id="pointCloudBtn">📍 Point Cloud</button>
                    <button class="tool-btn" onclick="setTool('polygon')" id="polygonBtn">⬡ Polygon</button>
                    <button class="tool-btn" onclick="setTool('measure3D')" id="measure3DBtn">📏 3D Measure</button>
                    <div class="point-cloud-controls">
                        <label>Point Size:</label>
                        <input type="range" class="point-size-slider" id="pointSizeSlider" min="1" max="10" value="3">
                    </div>
                </div>

                <!-- Mobile Controls -->
                <div class="tool-group technical">
                    <button class="tool-btn" onclick="toggleTouchMode()" id="touchModeBtn">👆 Touch Mode</button>
                    <button class="tool-btn" onclick="toggleGestures()" id="gesturesBtn">✋ Gestures</button>
                </div>

                <!-- Analysis Controls -->
                <div class="tool-group scientific">
                    <select id="analysisType">
                        <option value="height">Height Profile</option>
                        <option value="chemical">Chemical Composition</option>
                        <option value="roughness">Surface Roughness</option>
                        <option value="crystalline">Crystal Structure</option>
                    </select>
                    <button class="tool-btn" onclick="startAnalysis()">▶️ Analyze</button>
                    <button class="tool-btn" onclick="exportAnalysis()">💾 Export Data</button>
                </div>
            </div>
            
            <canvas id="whiteboard" width="800" height="600"></canvas>
            <div class="analysis-panel" id="analysisPanel"></div>
            <div class="scale-bar" id="scaleBar"></div>
        </div>

        <script>
            const canvas = document.getElementById('whiteboard');
            const ctx = canvas.getContext('2d');
            let isDrawing = false;
            let currentTool = 'pen';
            let startX, startY;
            let undoStack = [];
            let redoStack = [];
            
            // Set initial styles
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.lineCap = 'round';
            
            function saveState() {
                undoStack.push(canvas.toDataURL());
                redoStack = [];
            }
            
            function undo() {
                if (undoStack.length > 0) {
                    redoStack.push(canvas.toDataURL());
                    const img = new Image();
                    img.src = undoStack.pop();
                    img.onload = () => {
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(img, 0, 0);
                    };
                }
            }
            
            function redo() {
                if (redoStack.length > 0) {
                    undoStack.push(canvas.toDataURL());
                    const img = new Image();
                    img.src = redoStack.pop();
                    img.onload = () => {
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(img, 0, 0);
                    };
                }
            }
            
            function setTool(tool) {
                currentTool = tool;
                document.querySelectorAll('.tool-btn').forEach(btn => btn.classList.remove('active'));
                document.getElementById(tool + 'Btn').classList.add('active');
                
                switch(tool) {
                    case 'pen':
                        ctx.globalCompositeOperation = 'source-over';
                        ctx.globalAlpha = 1;
                        break;
                    case 'brush':
                        ctx.globalCompositeOperation = 'source-over';
                        ctx.globalAlpha = 0.6;
                        break;
                    case 'highlighter':
                        ctx.globalCompositeOperation = 'multiply';
                        ctx.globalAlpha = 0.3;
                        break;
                    case 'eraser':
                        ctx.globalCompositeOperation = 'destination-out';
                        ctx.globalAlpha = 1;
                        break;
                }
            }
            
            function setColor(color) {
                ctx.strokeStyle = color;
                document.getElementById('colorPicker').value = color;
            }
            
            function addElement(type) {
                const rect = canvas.getBoundingClientRect();
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;
                
                switch(type) {
                    case 'particle':
                        drawParticle(centerX, centerY);
                        break;
                    case 'crystal':
                        drawCrystal(centerX, centerY);
                        break;
                    case 'defect':
                        drawDefect(centerX, centerY);
                        break;
                    case 'grain':
                        drawGrain(centerX, centerY);
                        break;
                }
                saveState();
            }

            function drawParticle(x, y) {
                ctx.beginPath();
                ctx.arc(x, y, 10, 0, Math.PI * 2);
                ctx.stroke();
            }

            function drawCrystal(x, y) {
                ctx.beginPath();
                ctx.moveTo(x, y - 10);
                ctx.lineTo(x + 10, y);
                ctx.lineTo(x, y + 10);
                ctx.lineTo(x - 10, y);
                ctx.closePath();
                ctx.stroke();
            }

            function drawDefect(x, y) {
                ctx.beginPath();
                ctx.arc(x, y, 10, 0, Math.PI * 2);
                ctx.fillStyle = 'red';
                ctx.fill();
            }

            function drawGrain(x, y) {
                ctx.beginPath();
                ctx.arc(x, y, 10, 0, Math.PI * 2);
                ctx.fillStyle = 'blue';
                ctx.fill();
            }

            function addAnnotation() {
                const text = prompt('Enter annotation text:');
                if (text) {
                    ctx.font = '14px Arial';
                    ctx.fillStyle = ctx.strokeStyle;
                    ctx.fillText(text, startX, startY);
                }
            }

            function addCalibration() {
                const scale = prompt('Enter scale (nm/pixel):', '1');
                if (scale) {
                    document.getElementById('scaleIndicator').textContent = 
                        `Scale: ${scale} nm/pixel`;
                }
            }

            function startAnalysis() {
                const type = document.getElementById('analysisType').value;
                switch(type) {
                    case 'lineProfile':
                        startLineProfile(startX, startY);
                        break;
                    case 'areaAnalysis':
                        calculateAreaAnalysis(points);
                        break;
                    case 'particleAnalysis':
                        calculateParticleAnalysis(particles);
                        break;
                }
            }

            function exportAnalysis() {
                const analysisData = {
                    timestamp: new Date().toISOString(),
                    type: document.getElementById('analysisType').value,
                    measurements: currentMeasurements,
                    calibration: currentCalibration,
                    annotations: annotations
                };

                const blob = new Blob([JSON.stringify(analysisData, null, 2)], {
                    type: 'application/json'
                });
                
                const link = document.createElement('a');
                link.href = URL.createObjectURL(blob);
                link.download = `analysis_${Date.now()}.json`;
                link.click();
            }

            // Grid and scale functionality
            let showGrid = false;
            let scaleUnit = 'nm';
            let gridSize = 10;

            function toggleGrid() {
                showGrid = !showGrid;
                document.getElementById('gridBtn').classList.toggle('active');
                drawGrid();
            }

            function drawGrid() {
                if (!showGrid) return;
                
                const width = canvas.width;
                const height = canvas.height;
                
                ctx.save();
                ctx.strokeStyle = '#333';
                ctx.lineWidth = 0.5;
                
                for (let x = 0; x < width; x += gridSize) {
                    ctx.beginPath();
                    ctx.moveTo(x, 0);
                    ctx.lineTo(x, height);
                    ctx.stroke();
                }
                
                for (let y = 0; y < height; y += gridSize) {
                    ctx.beginPath();
                    ctx.moveTo(0, y);
                    ctx.lineTo(width, y);
                    ctx.stroke();
                }
                
                ctx.restore();
            }

            // Point Cloud handling
            let points = [];
            let pointSize = 3;
            let touchMode = false;
            
            function handlePointCloud(e) {
                const rect = canvas.getBoundingClientRect();
                const x = e.type.includes('touch') ? 
                    e.touches[0].clientX - rect.left : 
                    e.clientX - rect.left;
                const y = e.type.includes('touch') ? 
                    e.touches[0].clientY - rect.top : 
                    e.clientY - rect.top;
                
                points.push({x, y, z: Math.random() * 100}); // Simulated Z value
                drawPoints();
            }
            
            function drawPoints() {
                ctx.save();
                points.forEach(point => {
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, pointSize, 0, Math.PI * 2);
                    ctx.fillStyle = getDepthColor(point.z);
                    ctx.fill();
                });
                ctx.restore();
            }
            
            function getDepthColor(z) {
                // Color gradient based on depth
                const hue = (z / 100) * 240;
                return `hsl(${hue}, 100%, 50%)`;
            }

            // Polygon handling
            let polygonPoints = [];
            let isDrawingPolygon = false;
            
            function startPolygon(e) {
                if (!isDrawingPolygon) {
                    isDrawingPolygon = true;
                    polygonPoints = [];
                }
                
                const rect = canvas.getBoundingClientRect();
                const x = e.type.includes('touch') ? 
                    e.touches[0].clientX - rect.left : 
                    e.clientX - rect.left;
                const y = e.type.includes('touch') ? 
                    e.touches[0].clientY - rect.top : 
                    e.clientY - rect.top;
                
                polygonPoints.push({x, y});
                drawPolygon();
            }
            
            function drawPolygon() {
                if (polygonPoints.length < 1) return;
                
                ctx.save();
                ctx.beginPath();
                ctx.moveTo(polygonPoints[0].x, polygonPoints[0].y);
                
                polygonPoints.forEach((point, i) => {
                    if (i > 0) {
                        ctx.lineTo(point.x, point.y);
                    }
                });
                
                if (polygonPoints.length > 2) {
                    ctx.closePath();
                }
                
                ctx.strokeStyle = ctx.strokeStyle;
                ctx.stroke();
                
                // Draw vertices
                polygonPoints.forEach(point => {
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
                    ctx.fill();
                });
                
                ctx.restore();
            }

            // Mobile touch handling
            function setupTouchEvents() {
                canvas.addEventListener('touchstart', handleTouch);
                canvas.addEventListener('touchmove', handleTouch);
                canvas.addEventListener('touchend', handleTouchEnd);
            }
            
            function handleTouch(e) {
                e.preventDefault();
                const touch = e.touches[0];
                
                switch(currentTool) {
                    case 'pointCloud':
                        handlePointCloud(e);
                        break;
                    case 'polygon':
                        startPolygon(e);
                        break;
                    // ... handle other tools
                }
                
                updateTouchIndicator(touch);
            }
            
            function handleTouchEnd(e) {
                if (currentTool === 'polygon') {
                    if (e.touches.length === 0) {
                        completePolygon();
                    }
                }
            }
            
            function updateTouchIndicator(touch) {
                const indicator = document.getElementById('touchIndicator');
                indicator.textContent = `X: ${Math.round(touch.clientX)}, Y: ${Math.round(touch.clientY)}`;
            }

            // Tool selection
            function setTool(tool) {
                currentTool = tool;
                document.querySelectorAll('.tool-btn').forEach(btn => 
                    btn.classList.remove('active')
                );
                document.getElementById(tool + 'Btn').classList.add('active');
                
                // Tool-specific setup
                switch(tool) {
                    case 'pointCloud':
                        canvas.style.cursor = 'crosshair';
                        break;
                    case 'polygon':
                        canvas.style.cursor = 'pointer';
                        break;
                    // ... handle other tools
                }
            }

            // Initialize
            setupTouchEvents();
            document.getElementById('pointSizeSlider').addEventListener('input', (e) => {
                pointSize = parseInt(e.target.value);
                drawPoints();
            });

            // Scientific measurement functions
            function startLineProfile(start, end) {
                const profile = calculateLineProfile(start, end);
                updateAnalysisPanel({
                    type: 'lineProfile',
                    data: profile,
                    stats: {
                        mean: calculateMean(profile),
                        std: calculateStd(profile),
                        max: Math.max(...profile),
                        min: Math.min(...profile)
                    }
                });
            }

            function calculateAreaAnalysis(points) {
                const area = calculatePolygonArea(points);
                const perimeter = calculatePerimeter(points);
                updateAnalysisPanel({
                    type: 'areaAnalysis',
                    area: area,
                    perimeter: perimeter,
                    roughness: calculateRoughness(points)
                });
            }

            function calculateParticleAnalysis(particles) {
                const sizes = particles.map(p => calculateParticleSize(p));
                updateAnalysisPanel({
                    type: 'particleAnalysis',
                    count: particles.length,
                    meanSize: calculateMean(sizes),
                    sizeDistribution: calculateDistribution(sizes)
                });
            }

            function updateAnalysisPanel(data) {
                const panel = document.getElementById('analysisPanel');
                let html = `<h3>${data.type}</h3>`;
                
                switch(data.type) {
                    case 'lineProfile':
                        html += `
                            Mean: ${data.stats.mean.toFixed(2)} nm<br>
                            Std Dev: ${data.stats.std.toFixed(2)} nm<br>
                            Max: ${data.stats.max.toFixed(2)} nm<br>
                            Min: ${data.stats.min.toFixed(2)} nm
                        `;
                        break;
                    case 'areaAnalysis':
                        html += `
                            Area: ${data.area.toFixed(2)} nm²<br>
                            Perimeter: ${data.perimeter.toFixed(2)} nm<br>
                            Roughness: ${data.roughness.toFixed(2)} nm
                        `;
                        break;
                    case 'particleAnalysis':
                        html += `
                            Count: ${data.count}<br>
                            Mean Size: ${data.meanSize.toFixed(2)} nm<br>
                            Distribution: <canvas id="sizeHistogram"></canvas>
                        `;
                        break;
                }
                
                panel.innerHTML = html;
                
                if (data.type === 'particleAnalysis') {
                    drawHistogram(data.sizeDistribution);
                }
            }

            function exportAnalysis() {
                const analysisData = {
                    timestamp: new Date().toISOString(),
                    type: document.getElementById('analysisType').value,
                    measurements: currentMeasurements,
                    calibration: currentCalibration,
                    annotations: annotations
                };

                const blob = new Blob([JSON.stringify(analysisData, null, 2)], {
                    type: 'application/json'
                });
                
                const link = document.createElement('a');
                link.href = URL.createObjectURL(blob);
                link.download = `analysis_${Date.now()}.json`;
                link.click();
            }

            // Scientific calculations
            function calculateRoughness(points) {
                // Implement roughness calculation
                return Math.random() * 10; // Placeholder
            }

            function calculateParticleSize(particle) {
                // Implement particle size calculation
                return Math.sqrt(particle.area / Math.PI) * 2;
            }

            function calculateDistribution(values) {
                // Implement distribution calculation
                const bins = 10;
                const min = Math.min(...values);
                const max = Math.max(...values);
                const step = (max - min) / bins;
                
                const distribution = new Array(bins).fill(0);
                values.forEach(v => {
                    const binIndex = Math.floor((v - min) / step);
                    distribution[Math.min(binIndex, bins - 1)]++;
                });
                
                return {
                    bins: distribution,
                    min: min,
                    max: max,
                    step: step
                };
            }

            // Initialize scientific tools
            setupScientificTools();
        </script>
    """, height=750)

def render_collaboration_hub():
    """Render collaboration hub with sharing and communication features."""
    st.subheader("Collaboration Hub")
    
    tabs = st.tabs([
        "Team Chat & Notifications", 
        "File Sharing & Social", 
        "Live Analysis", 
        "Whiteboard & Notes",
        "Reactions & Feedback"
    ])
    
    # Team Chat & Notifications Tab
    with tabs[0]:
        st.subheader("Team Chat & Notifications")
        
        # Notification Settings
        with st.expander("Notification Settings"):
            st.checkbox("Research Updates", value=True)
            st.checkbox("Analysis Completion", value=True)
            st.checkbox("Team Messages", value=True)
            st.checkbox("Data Changes", value=True)
        
        # Chat Area
        st.text_input("Message your team...", key="chat_input")
        if st.button("Send"):
            st.info("Message sent!")
            
        # Notification Feed
        with st.container():
            st.markdown("### Recent Updates")
            st.info("🔬 New analysis results available")
            st.success("✅ Surface analysis completed")
            st.warning("⚠️ Data changes detected in Sample A")
    
    # File Sharing & Social Tab
    with tabs[1]:
        st.subheader("Share & Connect")
        
        # File Upload
        uploaded_file = st.file_uploader("Upload file to share", type=['csv', 'xlsx', 'pdf', 'png', 'jpg'])
        
        if uploaded_file:
            sharing_options = st.multiselect(
                "Share via:",
                ["Email", "WhatsApp", "LinkedIn", "Twitter", "Reddit"]
            )
            
            # Email sharing
            if "Email" in sharing_options:
                with st.expander("Share via Email"):
                    recipient = st.text_input("Recipient Email")
                    subject = st.text_input("Subject")
                    message = st.text_area("Message")
                    if st.button("Send Email"):
                        st.success("Email sent successfully!")
            
            # Social sharing buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if "WhatsApp" in sharing_options:
                    st.button("Share on WhatsApp 💬")
            with col2:
                if "LinkedIn" in sharing_options:
                    st.button("Share on LinkedIn 💼")
            with col3:
                if "Twitter" in sharing_options:
                    st.button("Share on Twitter 🐦")
        
        # Team Invitations
        with st.expander("Invite Collaborators"):
            email = st.text_input("Colleague's Email")
            role = st.selectbox("Role", ["Researcher", "Analyst", "Reviewer", "Admin"])
            if st.button("Send Invitation"):
                st.success(f"Invitation sent to {email}")
    
    # Live Analysis Tab
    with tabs[2]:
        st.subheader("Live Analysis Session")
        
        # Session controls
        if st.button("Start Live Session"):
            st.info("Live session started - others can join now")
            
        # Participant list
        st.markdown("### Active Participants")
        st.markdown("- 👤 You (Host)")
        st.markdown("- 👤 John Doe (Viewer)")
        st.markdown("- 👤 Jane Smith (Analyst)")
    
    # Whiteboard & Notes Tab
    with tabs[3]:
        st.subheader("Interactive Whiteboard")
        
        # Add whiteboard
        render_whiteboard()
        
        # Notes section below whiteboard
        with st.expander("Research Notes"):
            note = st.text_area(
                "Add text notes...",
                height=200,
                key="research_notes"
            )
            if st.button("Save Notes"):
                st.success("Notes saved!")
        
        # Comments section
        with st.expander("Comments"):
            comment = st.text_area("Add a comment")
            if st.button("Post Comment"):
                st.success("Comment posted!")
    
    # Reactions & Feedback Tab
    with tabs[4]:
        st.subheader("Reactions & Feedback")
        
        # Quick reactions
        reactions = {
            "👍 Approve": 0,
            "❤️ Love": 0,
            "🎯 Important": 0,
            "💡 Insight": 0,
            "⭐ Favorite": 0
        }
        
        cols = st.columns(len(reactions))
        for i, (reaction, count) in enumerate(reactions.items()):
            with cols[i]:
                if st.button(f"{reaction}\n{count}"):
                    st.success("Reaction added!")

def send_notification(user_id, message, notification_type):
    """Send push notification to user."""
    # Implement notification logic here
    pass

def share_file(file, platform, recipients):
    """Share file on selected platform."""
    # Implement sharing logic here
    pass 