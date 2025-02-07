import pytest
import numpy as np
import torch
from src.ecs_sehi_analyzer.core.research.sehi_processor import (
    SEHIProcessor,
    SEHIData,
    SEHIDegradationPredictor
)

@pytest.fixture
def processor():
    return SEHIProcessor()

@pytest.fixture
def sample_data():
    # Create synthetic SEHI data
    image = np.random.normal(0.5, 0.1, (512, 512))
    return SEHIData(
        image=image,
        metadata={"voltage": 20.0, "current": 1.5},
        timestamp=1234567890.0
    )

@pytest.fixture
def sample_series():
    # Create time series of SEHI data
    series = []
    base_time = 1234567890.0
    
    for i in range(5):
        image = np.random.normal(0.5 + i*0.1, 0.1, (512, 512))
        series.append(SEHIData(
            image=image,
            metadata={"voltage": 20.0, "current": 1.5},
            timestamp=base_time + i*3600
        ))
    return series

def test_model_initialization(processor):
    assert isinstance(processor.model, SEHIDegradationPredictor)
    assert processor.device in [torch.device("mps"), torch.device("cpu")]

def test_process_image(processor, sample_data):
    result = processor.process_image(sample_data)
    assert "degradation_score" in result
    assert "critical_areas" in result
    assert 0 <= result["degradation_score"] <= 1
    assert result["critical_areas"].shape == (512, 512)

def test_temporal_analysis(processor, sample_series):
    result = processor.analyze_temporal_changes(sample_series)
    assert "degradation_rate" in result
    assert "trend" in result
    assert len(result["scores"]) == len(sample_series)
    assert result["trend"] in ["accelerating", "stable"] 