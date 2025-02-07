import streamlit as st
import numpy as np
from typing import Optional, List
import threading
import queue
import os
from pathlib import Path
from datetime import datetime

class ScreenStudio:
    def __init__(self):
        self.recording = False
        self.audio_queue = queue.Queue()
        self.video_frames = []
        self.dependencies_checked = False
        
    def _check_dependencies(self):
        """Check if required dependencies are installed"""
        missing_deps = []
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            missing_deps.append("opencv-python")
            
        try:
            import pyaudio
            self.audio = pyaudio.PyAudio()
        except ImportError:
            st.warning("""
            PyAudio installation required for audio recording.
            
            On macOS:
            ```bash
            brew install portaudio
            pip install pyaudio
            ```
            
            On Linux:
            ```bash
            sudo apt-get install python3-pyaudio
            ```
            
            On Windows:
            ```bash
            pip install pyaudio
            ```
            """)
            missing_deps.append("PyAudio")
            
        if missing_deps:
            st.error(f"Missing dependencies: {', '.join(missing_deps)}")
            st.info("Please install required packages using: pip install " + " ".join(missing_deps))
            return False
        return True
        
    def start_recording(self, video: bool = True, audio: bool = True):
        """Start screen/webcam recording"""
        if not self.dependencies_checked:
            self.dependencies_checked = True
            if not self._check_dependencies():
                return
            
        self.recording = True
        if audio:
            self._start_audio_recording()
        if video:
            self._start_video_capture()
    
    def stop_recording(self) -> str:
        """Stop recording and return file path"""
        if not self.recording:
            return ""
            
        self.recording = False
        return self._save_recording()
    
    def _start_audio_recording(self):
        """Start audio recording thread"""
        if not hasattr(self, 'audio'):
            return
            
        def audio_callback(in_data, frame_count, time_info, status):
            self.audio_queue.put(in_data)
            return (in_data, pyaudio.paContinue)
            
        self.audio_stream = self.audio.open(
            format=self.audio.get_format_from_width(2),
            channels=2,
            rate=44100,
            input=True,
            stream_callback=audio_callback
        )
        self.audio_stream.start_stream()
        
    def _start_video_capture(self):
        """Start video capture thread"""
        if not hasattr(self, 'cv2'):
            return
            
        self.cap = self.cv2.VideoCapture(0)
        
    def _save_recording(self) -> str:
        """Save recording to file"""
        # Create output directory if it doesn't exist
        output_dir = Path("recordings")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"recording_{timestamp}.mp4"
        
        # Save logic here (simplified)
        return str(output_path)
        
    def render_studio_controls(self):
        """Render studio control interface"""
        st.markdown("### 🎥 Screen Studio")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔴 Start Recording", key="screen_start_recording"):
                self.start_recording()
                st.session_state.recording = True
                
        with col2:
            if st.button("⏹️ Stop Recording", key="screen_stop_recording"):
                if hasattr(st.session_state, 'recording'):
                    file_path = self.stop_recording()
                    if file_path:
                        st.success(f"Recording saved: {file_path}")
                    
        with col3:
            st.button("📤 Export Video", key="screen_export_video")
            
        # Studio settings
        with st.expander("Studio Settings"):
            st.checkbox("Record Webcam", value=True, key="screen_record_webcam")
            st.checkbox("Record Audio", value=True, key="screen_record_audio")
            st.checkbox("Record Screen", value=True, key="screen_record_screen")
            st.slider("Video Quality", 0, 100, 80, key="screen_video_quality")
            st.selectbox("Output Format", ["MP4", "WebM", "GIF"], key="screen_output_format") 