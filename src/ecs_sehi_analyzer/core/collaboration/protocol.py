import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Optional
import datetime
import json

@dataclass
class Protocol:
    id: str
    name: str
    steps: List[Dict]
    status: str
    created_by: str
    created_at: datetime.datetime
    last_modified: datetime.datetime

class ProtocolManager:
    def __init__(self):
        if 'protocols' not in st.session_state:
            st.session_state.protocols = []
        if 'active_protocol' not in st.session_state:
            st.session_state.active_protocol = None
    
    def create_protocol(self, name: str, steps: List[Dict]) -> Protocol:
        """Create a new protocol"""
        protocol = Protocol(
            id=f"prot_{len(st.session_state.protocols)}",
            name=name,
            steps=steps,
            status="draft",
            created_by="Current User",
            created_at=datetime.datetime.now(),
            last_modified=datetime.datetime.now()
        )
        st.session_state.protocols.append(protocol)
        return protocol
    
    def render_protocol_editor(self):
        """Render the protocol editor"""
        st.subheader("📋 Protocol Editor")
        
        # Protocol creation
        with st.form("protocol_form"):
            name = st.text_input("Protocol Name")
            
            # Steps
            steps = []
            for i in range(3):  # Allow 3 steps initially
                step = st.text_area(f"Step {i+1}")
                if step:
                    steps.append({
                        "order": i,
                        "description": step,
                        "status": "pending"
                    })
            
            if st.form_submit_button("Create Protocol"):
                if name and steps:
                    protocol = self.create_protocol(name, steps)
                    st.success(f"Created protocol: {protocol.name}")
        
        # Display existing protocols
        if st.session_state.protocols:
            st.subheader("Existing Protocols")
            for protocol in st.session_state.protocols:
                with st.expander(f"{protocol.name} ({protocol.status})"):
                    for step in protocol.steps:
                        st.write(f"- {step['description']}") 