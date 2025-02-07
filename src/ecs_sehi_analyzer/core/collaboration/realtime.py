import streamlit as st
from dataclasses import dataclass
from typing import Optional, List, Dict
import json

@dataclass
class RealtimeMessage:
    type: str
    content: Dict
    sender: str
    timestamp: float

class RealtimeEngine:
    def __init__(self):
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'users' not in st.session_state:
            st.session_state.users = set()
    
    def send_message(self, msg_type: str, content: Dict, sender: str):
        """Send a realtime message"""
        message = RealtimeMessage(
            type=msg_type,
            content=content,
            sender=sender,
            timestamp=time.time()
        )
        st.session_state.messages.append(message)
    
    def render_chat(self):
        """Render realtime chat"""
        st.markdown("### 💬 Live Chat")
        
        # Message input
        message = st.text_input("Message", key="chat_message")
        if st.button("Send", key="send_message"):
            if message:
                self.send_message("chat", {"text": message}, "Current User")
        
        # Display messages
        for msg in reversed(st.session_state.messages):
            with st.expander(f"{msg.sender} - {msg.timestamp}"):
                st.write(msg.content['text']) 