import pytest
from datetime import datetime, timedelta
from src.ecs_sehi_analyzer.core.sustainability.metrics import (
    SustainabilityEngine,
    SustainabilityMetric,
    ImpactCategory
)

@pytest.fixture
def engine():
    return SustainabilityEngine()

@pytest.fixture
def sample_metrics():
    now = datetime.now()
    return [
        SustainabilityMetric(
            category=ImpactCategory.ENERGY,
            value=100.0,
            unit="kWh",
            timestamp=now,
            source="Lab Equipment",
            confidence=0.95
        ),
        SustainabilityMetric(
            category=ImpactCategory.WATER,
            value=500.0,
            unit="L",
            timestamp=now,
            source="Facility Meter",
            confidence=0.98
        )
    ]

def test_metric_recording(engine, sample_metrics):
    for metric in sample_metrics:
        engine.record_metric("PROJ-001", metric)
    
    assert "PROJ-001" in engine.metrics
    assert len(engine.metrics["PROJ-001"]) == 2

def test_impact_analysis(engine, sample_metrics):
    for metric in sample_metrics:
        engine.record_metric("PROJ-001", metric)
        
    analysis = engine.analyze_impact("PROJ-001")
    assert "total_impact" in analysis
    assert "trends" in analysis
    assert "recommendations" in analysis

def test_target_setting(engine):
    engine.set_target(ImpactCategory.ENERGY, 80.0)
    assert ImpactCategory.ENERGY.value in engine.targets
    assert engine.targets[ImpactCategory.ENERGY.value] == 80.0

def test_trend_analysis(engine):
    now = datetime.now()
    
    # Record increasing energy usage
    for i in range(5):
        metric = SustainabilityMetric(
            category=ImpactCategory.ENERGY,
            value=100.0 + i*10,
            unit="kWh",
            timestamp=now + timedelta(days=i),
            source="Lab Equipment"
        )
        engine.record_metric("PROJ-001", metric)
    
    analysis = engine.analyze_impact("PROJ-001")
    trends = analysis["trends"]
    
    assert ImpactCategory.ENERGY.value in trends
    assert trends[ImpactCategory.ENERGY.value]["direction"] == "increasing" 