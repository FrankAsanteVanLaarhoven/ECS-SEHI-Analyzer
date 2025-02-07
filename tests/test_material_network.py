import pytest
import numpy as np
import torch
from src.ecs_sehi_analyzer.core.neural.material_network import (
    MaterialAnalyzer,
    NetworkConfig,
    MaterialAnalysisNetwork
)

@pytest.fixture
def analyzer():
    config = NetworkConfig(
        input_dim=(64, 64),
        hidden_layers=[256, 128],
        epochs=2
    )
    return MaterialAnalyzer(config)

@pytest.fixture
def sample_data():
    # Create synthetic training data
    train_data = np.random.normal(0, 1, (100, 1, 64, 64))
    train_labels = np.random.normal(0, 1, (100, 1))
    
    # Create synthetic validation data
    val_data = np.random.normal(0, 1, (20, 1, 64, 64))
    val_labels = np.random.normal(0, 1, (20, 1))
    
    return train_data, train_labels, val_data, val_labels

def test_model_initialization(analyzer):
    assert isinstance(analyzer.model, MaterialAnalysisNetwork)
    assert analyzer.model.training
    assert len(analyzer.training_history) == 0

def test_training(analyzer, sample_data):
    train_data, train_labels, val_data, val_labels = sample_data
    
    result = analyzer.train(
        train_data,
        train_labels,
        val_data,
        val_labels
    )
    
    assert "final_loss" in result
    assert "epochs_completed" in result
    assert result["epochs_completed"] == analyzer.config.epochs
    assert len(analyzer.training_history) == analyzer.config.epochs

def test_prediction(analyzer, sample_data):
    train_data, train_labels, _, _ = sample_data
    
    # Train the model first
    analyzer.train(train_data, train_labels)
    
    # Make predictions
    predictions = analyzer.predict(train_data[:5])
    assert predictions.shape == (5, 1)

def test_validation(analyzer, sample_data):
    train_data, train_labels, val_data, val_labels = sample_data
    
    # Train the model first
    analyzer.train(train_data, train_labels)
    
    # Validate
    metrics = analyzer.validate(val_data, val_labels)
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert isinstance(metrics["loss"], float)
    assert isinstance(metrics["accuracy"], float)

def test_device_selection(analyzer):
    assert analyzer.config.device in ["cuda", "cpu"]
    assert next(analyzer.model.parameters()).device.type == analyzer.config.device 