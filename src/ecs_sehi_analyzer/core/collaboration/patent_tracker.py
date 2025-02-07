from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import streamlit as st
import json
import uuid
from enum import Enum

class PatentStatus(Enum):
    DRAFT = "draft"
    REVIEW = "review"
    SUBMITTED = "submitted"
    PENDING = "pending"
    GRANTED = "granted"
    REJECTED = "rejected"

@dataclass
class PatentClaim:
    claim_id: str
    text: str
    category: str
    priority: int
    dependencies: List[str] = field(default_factory=list)
    evidence: Dict[str, str] = field(default_factory=dict)

@dataclass
class Patent:
    patent_id: str
    title: str
    description: str
    inventors: List[str]
    claims: List[PatentClaim]
    status: PatentStatus
    created_at: datetime
    updated_at: datetime
    priority_date: datetime
    assignee: str
    metadata: Dict = field(default_factory=dict)

class PatentTracker:
    def __init__(self):
        self.patents: Dict[str, Patent] = {}
        self.claim_templates: Dict[str, str] = self._load_claim_templates()
        
    def create_patent(self, 
                     title: str,
                     description: str,
                     inventors: List[str],
                     assignee: str) -> str:
        """Create new patent application"""
        patent_id = f"PAT-{uuid.uuid4().hex[:8]}"
        now = datetime.now()
        
        patent = Patent(
            patent_id=patent_id,
            title=title,
            description=description,
            inventors=inventors,
            claims=[],
            status=PatentStatus.DRAFT,
            created_at=now,
            updated_at=now,
            priority_date=now,
            assignee=assignee
        )
        
        self.patents[patent_id] = patent
        return patent_id
        
    def add_claim(self, 
                 patent_id: str,
                 text: str,
                 category: str,
                 priority: int,
                 dependencies: List[str] = None) -> str:
        """Add claim to patent"""
        if patent_id not in self.patents:
            raise ValueError(f"Patent {patent_id} not found")
            
        claim = PatentClaim(
            claim_id=f"CLM-{uuid.uuid4().hex[:8]}",
            text=text,
            category=category,
            priority=priority,
            dependencies=dependencies or []
        )
        
        self.patents[patent_id].claims.append(claim)
        self.patents[patent_id].updated_at = datetime.now()
        
        return claim.claim_id
        
    def update_status(self, patent_id: str, status: PatentStatus):
        """Update patent status"""
        if patent_id not in self.patents:
            raise ValueError(f"Patent {patent_id} not found")
            
        self.patents[patent_id].status = status
        self.patents[patent_id].updated_at = datetime.now()
        
    def render_patent_interface(self):
        """Render Streamlit interface for patent management"""
        st.markdown("### 📜 Patent Management")
        
        # Patent creation
        with st.expander("Create New Patent", expanded=False):
            title = st.text_input("Patent Title")
            description = st.text_area("Description")
            inventors = st.text_input("Inventors (comma-separated)")
            assignee = st.text_input("Assignee")
            
            if st.button("Create Patent"):
                if title and description and inventors and assignee:
                    patent_id = self.create_patent(
                        title=title,
                        description=description,
                        inventors=[inv.strip() for inv in inventors.split(",")],
                        assignee=assignee
                    )
                    st.success(f"Patent created: {patent_id}")
                else:
                    st.error("Please fill all required fields")
                    
        # Patent list and management
        st.markdown("#### Existing Patents")
        for patent_id, patent in self.patents.items():
            with st.expander(f"{patent.title} ({patent_id})"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Status:** {patent.status.value}")
                    st.markdown(f"**Inventors:** {', '.join(patent.inventors)}")
                    st.markdown(f"**Priority Date:** {patent.priority_date.date()}")
                    
                with col2:
                    new_status = st.selectbox(
                        "Update Status",
                        [status.value for status in PatentStatus],
                        index=[status.value for status in PatentStatus].index(patent.status.value),
                        key=f"status_{patent_id}"
                    )
                    if new_status != patent.status.value:
                        self.update_status(patent_id, PatentStatus(new_status))
                        
                # Claims management
                st.markdown("##### Claims")
                if st.button("Add Claim", key=f"add_claim_{patent_id}"):
                    claim_text = st.text_area(
                        "Claim Text",
                        value=self.claim_templates.get("default", ""),
                        key=f"claim_text_{patent_id}"
                    )
                    category = st.selectbox(
                        "Category",
                        ["Method", "System", "Composition", "Device"],
                        key=f"claim_category_{patent_id}"
                    )
                    priority = st.number_input(
                        "Priority",
                        min_value=1,
                        max_value=100,
                        value=1,
                        key=f"claim_priority_{patent_id}"
                    )
                    
                    if st.button("Save Claim", key=f"save_claim_{patent_id}"):
                        self.add_claim(
                            patent_id=patent_id,
                            text=claim_text,
                            category=category,
                            priority=priority
                        )
                        
                # Display existing claims
                for claim in patent.claims:
                    st.markdown(f"""
                    **Claim {claim.claim_id}** (Priority: {claim.priority})  
                    Category: {claim.category}  
                    {claim.text}
                    """)
                    
    def _load_claim_templates(self) -> Dict[str, str]:
        """Load predefined claim templates"""
        return {
            "default": """
            A method comprising:
            - step 1...
            - step 2...
            - step 3...
            """,
            "method": """
            A method for analyzing material degradation, comprising:
            - receiving temporal-spatial data...
            - processing the data using quantum-safe encryption...
            - generating analysis results...
            """,
            "system": """
            A system comprising:
            - a quantum processor configured to...
            - a neural network configured to...
            - an output interface configured to...
            """
        } 