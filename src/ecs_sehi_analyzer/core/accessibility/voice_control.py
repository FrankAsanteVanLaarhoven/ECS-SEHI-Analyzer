import streamlit as st
import speech_recognition as sr
from typing import Optional, Callable, Dict
import threading
import queue

class VoiceControl:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.commands = {}
        self.listening = False
        self.command_queue = queue.Queue()
        
    def register_command(self, phrase: str, action: Callable):
        """Register voice command and corresponding action"""
        self.commands[phrase.lower()] = action
        
    def start_listening(self):
        """Start voice command recognition"""
        self.listening = True
        threading.Thread(target=self._listen_loop, daemon=True).start()
        
    def _listen_loop(self):
        """Continuous listening loop"""
        with sr.Microphone() as source:
            while self.listening:
                try:
                    audio = self.recognizer.listen(source)
                    text = self.recognizer.recognize_google(audio).lower()
                    
                    for command, action in self.commands.items():
                        if command in text:
                            self.command_queue.put(action)
                except Exception as e:
                    pass
                    
    def render_voice_controls(self):
        """Render voice control interface"""
        st.markdown("### 🎤 Voice Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎤 Start Voice Control", key="voice_start"):
                self.start_listening()
                st.success("Voice control activated")
                
        with col2:
            if st.button("⏹️ Stop Voice Control", key="voice_stop"):
                self.listening = False
                st.info("Voice control deactivated")
                
        # Voice command settings
        with st.expander("Voice Settings"):
            st.selectbox("Language", ["English", "Dutch", "German"], key="voice_language")
            st.slider("Sensitivity", 0.0, 1.0, 0.5, key="voice_sensitivity")
            st.checkbox("Show Command Feedback", key="voice_feedback") 