import pytest
from datetime import datetime
from src.ecs_sehi_analyzer.core.collaboration.patent_tracker import (
    PatentTracker,
    Patent,
    PatentClaim,
    PatentStatus
)

@pytest.fixture
def tracker():
    return PatentTracker()

@pytest.fixture
def sample_patent_data():
    return {
        "title": "Quantum Material Analysis Method",
        "description": "A novel method for analyzing material degradation",
        "inventors": ["John Doe", "Jane Smith"],
        "assignee": "Research Labs Inc."
    }

def test_patent_creation(tracker, sample_patent_data):
    patent_id = tracker.create_patent(**sample_patent_data)
    assert patent_id in tracker.patents
    patent = tracker.patents[patent_id]
    assert patent.title == sample_patent_data["title"]
    assert patent.status == PatentStatus.DRAFT

def test_claim_addition(tracker, sample_patent_data):
    patent_id = tracker.create_patent(**sample_patent_data)
    claim_id = tracker.add_claim(
        patent_id=patent_id,
        text="A method comprising...",
        category="Method",
        priority=1
    )
    
    patent = tracker.patents[patent_id]
    assert len(patent.claims) == 1
    assert patent.claims[0].claim_id == claim_id

def test_status_update(tracker, sample_patent_data):
    patent_id = tracker.create_patent(**sample_patent_data)
    tracker.update_status(patent_id, PatentStatus.REVIEW)
    assert tracker.patents[patent_id].status == PatentStatus.REVIEW

def test_invalid_patent_id(tracker):
    with pytest.raises(ValueError):
        tracker.add_claim(
            patent_id="invalid_id",
            text="Test claim",
            category="Method",
            priority=1
        ) 