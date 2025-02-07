import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
from pathlib import Path
from typing import List, Dict, Optional
import datetime

class Collaboration:
    def __init__(self):
        # Initialize collaboration state
        if 'collaborators' not in st.session_state:
            st.session_state.collaborators = {}
        if 'shared_projects' not in st.session_state:
            st.session_state.shared_projects = {}
        if 'invitations' not in st.session_state:
            st.session_state.invitations = []
            
    def send_invitation(self, email: str, project_name: str, access_level: str = "view") -> bool:
        """Send collaboration invitation email"""
        # Create invitation
        invitation_id = f"inv_{len(st.session_state.invitations) + 1}"
        invitation = {
            "id": invitation_id,
            "project": project_name,
            "email": email,
            "access_level": access_level,
            "status": "pending",
            "date_sent": datetime.datetime.now().isoformat()
        }
        
        try:
            # Check if email is enabled
            if st.secrets.get("ENABLE_EMAIL", False):
                # Email configuration
                smtp_server = st.secrets.get("SMTP_SERVER", "smtp.gmail.com")
                smtp_port = st.secrets.get("SMTP_PORT", 587)
                sender_email = st.secrets.get("SENDER_EMAIL")
                sender_password = st.secrets.get("SENDER_PASSWORD")
                
                if not all([smtp_server, smtp_port, sender_email, sender_password]):
                    raise ValueError("Email configuration incomplete")
                
                # Create email content
                msg = MIMEMultipart()
                msg['From'] = sender_email
                msg['To'] = email
                msg['Subject'] = f"Invitation to collaborate on {project_name}"
                
                body = f"""
                You've been invited to collaborate on the project: {project_name}
                
                Access Level: {access_level}
                
                To accept this invitation, please click the following link:
                {st.secrets.get('APP_URL', 'http://localhost:8501')}?invite={invitation_id}
                
                This invitation will expire in 7 days.
                """
                
                msg.attach(MIMEText(body, 'plain'))
                
                # Send email
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()
                    server.login(sender_email, sender_password)
                    server.send_message(msg)
            else:
                # Development mode - just store the invitation without sending email
                st.info("Email sending is disabled. Running in development mode.")
            
            # Store invitation
            st.session_state.invitations.append(invitation)
            return True
            
        except Exception as e:
            if "No secrets found" in str(e):
                st.warning("Running in development mode - email sending disabled")
                st.session_state.invitations.append(invitation)
                return True
            else:
                st.error(f"Failed to send invitation: {str(e)}")
                return False
    
    def accept_invitation(self, invitation_id: str) -> bool:
        """Accept a collaboration invitation"""
        for inv in st.session_state.invitations:
            if inv['id'] == invitation_id and inv['status'] == 'pending':
                inv['status'] = 'accepted'
                project = inv['project']
                email = inv['email']
                
                if project not in st.session_state.collaborators:
                    st.session_state.collaborators[project] = []
                
                st.session_state.collaborators[project].append({
                    'email': email,
                    'access_level': inv['access_level'],
                    'joined': datetime.datetime.now().isoformat()
                })
                return True
        return False
    
    def get_project_collaborators(self, project_name: str) -> List[Dict]:
        """Get list of collaborators for a project"""
        return st.session_state.collaborators.get(project_name, [])
    
    def remove_collaborator(self, project_name: str, email: str) -> bool:
        """Remove a collaborator from a project"""
        if project_name in st.session_state.collaborators:
            st.session_state.collaborators[project_name] = [
                c for c in st.session_state.collaborators[project_name]
                if c['email'] != email
            ]
            return True
        return False

def render_collaboration_ui():
    """Render the collaboration UI"""
    collab = Collaboration()
    
    with st.expander("🤝 Project Sharing"):
        # Show development mode notice if email is not configured
        if not st.secrets.get("ENABLE_EMAIL", False):
            st.info("⚠️ Running in development mode - email notifications disabled")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            project_name = st.text_input("Project Name", key="collab_project_name")
            email = st.text_input("Collaborator Email", key="collab_email")
            access_level = st.selectbox(
                "Access Level",
                ["view", "edit", "admin"],
                key="collab_access_level",
                help="View: Can only view data\nEdit: Can modify data\nAdmin: Full access"
            )
        
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("Send Invitation", key="collab_send_invite", type="primary"):
                if project_name and email:
                    if collab.send_invitation(email, project_name, access_level):
                        st.success(f"Invitation created for {email}")
                else:
                    st.warning("Please enter project name and email")
    
    with st.expander("👥 Current Collaborators"):
        if project_name:
            collaborators = collab.get_project_collaborators(project_name)
            if collaborators:
                for i, collab_info in enumerate(collaborators):
                    cols = st.columns([3, 2, 1])
                    with cols[0]:
                        st.write(collab_info['email'])
                    with cols[1]:
                        st.write(collab_info['access_level'])
                    with cols[2]:
                        if st.button("Remove", key=f"collab_remove_{i}_{collab_info['email']}"):
                            if collab.remove_collaborator(project_name, collab_info['email']):
                                st.success(f"Removed {collab_info['email']}")
            else:
                st.info("No collaborators yet")
    
    with st.expander("📬 Pending Invitations"):
        pending = [inv for inv in st.session_state.invitations if inv['status'] == 'pending']
        if pending:
            for i, inv in enumerate(pending):
                cols = st.columns([2, 2, 1, 1])
                with cols[0]:
                    st.write(inv['email'])
                with cols[1]:
                    st.write(inv['project'])
                with cols[2]:
                    st.write(inv['access_level'])
                with cols[3]:
                    if st.button("Cancel", key=f"collab_cancel_{i}_{inv['id']}"):
                        inv['status'] = 'cancelled'
                        st.success("Invitation cancelled")
        else:
            st.info("No pending invitations")