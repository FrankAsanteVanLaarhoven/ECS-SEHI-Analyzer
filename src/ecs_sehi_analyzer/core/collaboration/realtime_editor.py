import streamlit as st
from typing import Dict, List, Optional
import json
import time
from dataclasses import dataclass
from ..quantum.encryption import QuantumEncryptionEngine, SecureQuantumChannel

@dataclass
class CollaborationSession:
    """Session data for collaboration"""
    session_id: str
    owner: str
    participants: List[str]
    created_at: float
    document_version: int = 0
    is_active: bool = True

class ResearchDocument:
    def __init__(self):
        self.content: Dict = {}
        self.version: int = 0
        self.history: List[Dict] = []
        self.contributors: List[str] = []
        
    def update_content(self, changes: Dict, author: str) -> Dict:
        """Update document with version control"""
        self.version += 1
        timestamp = time.time()
        
        # Record change in history
        change_record = {
            "version": self.version,
            "author": author,
            "timestamp": timestamp,
            "changes": changes
        }
        self.history.append(change_record)
        
        # Update content
        self.content.update(changes)
        
        # Add contributor if new
        if author not in self.contributors:
            self.contributors.append(author)
            
        return {
            "version": self.version,
            "timestamp": timestamp,
            "status": "success"
        }
        
    def get_version(self, version: int) -> Optional[Dict]:
        """Retrieve specific version of document"""
        if version > self.version:
            return None
            
        content = {}
        for change in self.history[:version]:
            content.update(change["changes"])
        return content

class CollaborationHub:
    def __init__(self):
        self.encryption_engine = QuantumEncryptionEngine()
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self.documents: Dict[str, ResearchDocument] = {}
        
    def create_session(self, owner: str) -> str:
        """Create new collaboration session"""
        session_id = f"session_{int(time.time())}"
        self.active_sessions[session_id] = CollaborationSession(
            session_id=session_id,
            owner=owner,
            participants=[owner],
            created_at=time.time()
        )
        return session_id
        
    def join_session(self, session_id: str, participant: str) -> SecureQuantumChannel:
        """Join existing collaboration session"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
            
        session = self.active_sessions[session_id]
        if participant not in session.participants:
            session.participants.append(participant)
            
        # Create secure channel for participant
        return self.encryption_engine.secure_channel(participant)
        
    def update_document(self, session_id: str, changes: Dict, author: str) -> Dict:
        """Update document in collaboration session"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
            
        if session_id not in self.documents:
            self.documents[session_id] = ResearchDocument()
            
        document = self.documents[session_id]
        return document.update_content(changes, author)
    
    def render_collaboration_interface(self):
        """Render Streamlit interface for collaboration"""
        st.markdown("### 👥 Research Collaboration Hub")
        
        # Session management
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Create New Session"):
                session_id = self.create_session(st.session_state.get("user", "Anonymous"))
                st.success(f"Session created: {session_id}")
                
        with col2:
            existing_sessions = list(self.active_sessions.keys())
            if existing_sessions:
                session_id = st.selectbox("Join Existing Session", existing_sessions)
                if st.button("Join"):
                    channel = self.join_session(session_id, st.session_state.get("user", "Anonymous"))
                    st.success("Joined session successfully!")
                    
        # Document editor
        if "current_session" in st.session_state:
            session_id = st.session_state.current_session
            document = self.documents.get(session_id)
            
            if document:
                st.markdown("#### 📝 Document Editor")
                
                # Show current content
                content = st.text_area(
                    "Content",
                    value=json.dumps(document.content, indent=2),
                    height=200
                )
                
                # Update button
                if st.button("Update Document"):
                    try:
                        changes = json.loads(content)
                        result = self.update_document(
                            session_id,
                            changes,
                            st.session_state.get("user", "Anonymous")
                        )
                        st.success(f"Document updated to version {result['version']}")
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format")
                        
                # Show document history
                with st.expander("Document History"):
                    for change in document.history:
                        st.markdown(f"""
                        **Version {change['version']}**  
                        Author: {change['author']}  
                        Time: {time.ctime(change['timestamp'])}
                        """)
                        
                # Show active participants
                st.sidebar.markdown("#### 👥 Active Participants")
                session = self.active_sessions[session_id]
                for participant in session.participants:
                    st.sidebar.markdown(f"- {participant}") 