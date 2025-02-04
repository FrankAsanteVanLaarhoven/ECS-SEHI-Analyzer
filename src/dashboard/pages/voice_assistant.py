import streamlit as st
import streamlit.components.v1 as components
import speech_recognition as sr
from gtts import gTTS
import openai
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import json
from pathlib import Path
import queue
import threading
import datetime

class SEHIAssistant:
    """SEHI's AI Voice Assistant with wake word detection."""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.wake_words = ["hey sehi", "okay sehi", "hi sehi"]
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Initialize OpenAI for chat
        try:
            self.openai = openai
            self.openai.api_key = st.secrets["openai_api_key"]
        except:
            st.warning("OpenAI integration limited. Please add API key to secrets.")
        
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    def start_listening(self):
        """Start listening for wake word."""
        self.is_listening = True
        threading.Thread(target=self._listen_for_wake_word, daemon=True).start()
    
    def stop_listening(self):
        """Stop listening."""
        self.is_listening = False
    
    def _listen_for_wake_word(self):
        """Continuously listen for wake word."""
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source)
                
                while self.is_listening:
                    try:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                        text = self.recognizer.recognize_google(audio).lower()
                        
                        if any(wake_word in text for wake_word in self.wake_words):
                            self._wake_word_detected()
                            
                    except sr.WaitTimeoutError:
                        continue
                    except Exception as e:
                        continue
                        
        except Exception as e:
            st.error(f"Error in wake word detection: {str(e)}")
    
    def _wake_word_detected(self):
        """Handle wake word detection."""
        self.speak("Yes, how can I help you?")
        self._listen_for_command()
    
    def _listen_for_command(self):
        """Listen for user command after wake word."""
        try:
            with sr.Microphone() as source:
                st.info("Listening for command...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                command = self.recognizer.recognize_google(audio).lower()
                
                # Process command
                response = self._process_command(command)
                self.speak(response)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": command,
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                })
                
        except Exception as e:
            self.speak("Sorry, I didn't catch that. Could you repeat?")
    
    def _process_command(self, command):
        """Process voice command with AI."""
        try:
            # Prepare context from chat history
            messages = [
                {"role": "system", "content": """You are SEHI, an intelligent scientific assistant specializing in 
                 surface analysis, point cloud processing, and material science. You can help with data analysis,
                 visualization, and provide detailed scientific explanations."""}
            ]
            
            # Add recent chat history
            for msg in st.session_state.chat_history[-5:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add current command
            messages.append({"role": "user", "content": command})
            
            # Get AI response
            response = self.openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message["content"]
            
        except Exception as e:
            return f"I encountered an error: {str(e)}"
    
    def speak(self, text):
        """Convert text to speech."""
        try:
            tts = gTTS(text=text, lang='en')
            tts.save("response.mp3")
            data, samplerate = sf.read("response.mp3")
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            st.error(f"Speech synthesis error: {str(e)}")

def render_voice_assistant():
    """Render voice assistant interface."""
    st.title("🎙️ SEHI Voice Assistant")
    
    # Initialize assistant if not exists
    if 'assistant' not in st.session_state:
        st.session_state.assistant = SEHIAssistant()
    
    # Voice activation section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
            ### Voice Commands
            Say "Hey SEHI" or "Okay SEHI" to activate the assistant.
            
            Example commands:
            - "Analyze this point cloud"
            - "Show me the surface statistics"
            - "Export the current model"
            - "What does this data mean?"
        """)
    
    with col2:
        if st.button("🎤 Start Listening", key="start_listening"):
            st.session_state.assistant.start_listening()
        if st.button("⏹️ Stop Listening", key="stop_listening"):
            st.session_state.assistant.stop_listening()
    
    # Chat history display
    st.markdown("### Conversation History")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(f"[{message['timestamp']}] {message['content']}")
    
    # Manual text input
    with st.expander("💬 Text Input"):
        text_input = st.text_input("Type your message:")
        if st.button("Send"):
            response = st.session_state.assistant._process_command(text_input)
            st.session_state.assistant.speak(response)