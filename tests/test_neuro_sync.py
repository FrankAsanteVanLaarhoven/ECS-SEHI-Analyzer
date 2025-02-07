import pytest
import torch
import numpy as np
from datetime import datetime
from src.ecs_sehi_analyzer.core.quantum.neuro_sync import (
    NeuroSyncEngine,
    SyncConfig,
    MaterialAnalysisNetwork,
    NetworkConfig
)

@pytest.fixture
def sync_engine():
    return NeuroSyncEngine()

@pytest.fixture
def sample_models():
    config = NetworkConfig(
        input_dim=(64, 64),
        hidden_layers=[128, 64],
        epochs=1
    )
    
    models = {
        "model_a": MaterialAnalysisNetwork(config),
        "model_b": MaterialAnalysisNetwork(config)
    }
    
    return models

def test_model_registration(sync_engine, sample_models):
    for model_id, model in sample_models.items():
        sync_engine.register_model(model_id, model)
        
    assert len(sync_engine.active_models) == 2
    assert "model_a" in sync_engine.active_models
    assert "model_b" in sync_engine.active_models

def test_parameter_sync(sync_engine, sample_models):
    # Register models
    for model_id, model in sample_models.items():
        sync_engine.register_model(model_id, model)
        
    # Perform sync
    result = sync_engine.sync_parameters("model_a", "model_b")
    
    assert result["status"] == "success"
    assert "synced_parameters" in result
    assert "confidence" in result

def test_sync_quality_analysis(sync_engine):
    # Add some sync history
    sync_engine.sync_history.extend([
        {
            "source": "model_a",
            "target": "model_b",
            "timestamp": datetime.now(),
            "confidence": 0.95,
            "synced_params": 1000,
            "total_params": 1000
        },
        {
            "source": "model_a",
            "target": "model_b",
            "timestamp": datetime.now(),
            "confidence": 0.98,
            "synced_params": 1000,
            "total_params": 1000
        }
    ])
    
    analysis = sync_engine.analyze_sync_quality()
    assert "total_syncs" in analysis
    assert "success_rate" in analysis
    assert "average_confidence" in analysis
    assert "error_rate" in analysis

def test_invalid_model_sync(sync_engine, sample_models):
    sync_engine.register_model("model_a", sample_models["model_a"])
    
    with pytest.raises(ValueError):
        sync_engine.sync_parameters("model_a", "invalid_model")

def test_sync_trends(sync_engine):
    # Add sync history with improving confidence
    for i in range(5):
        sync_engine.sync_history.append({
            "source": "model_a",
            "target": "model_b",
            "timestamp": datetime.now(),
            "confidence": 0.90 + i*0.02,
            "synced_params": 1000,
            "total_params": 1000
        })
    
    analysis = sync_engine.analyze_sync_quality()
    trends = analysis["sync_trends"]
    
    assert "confidence_trend" in trends
    assert trends["trend_direction"] == "improving" 