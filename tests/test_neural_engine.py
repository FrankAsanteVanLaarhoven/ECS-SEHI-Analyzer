import pytest
import torch
import numpy as np
from src.ecs_sehi_analyzer.core.neural.neural_engine import (
    NeuralEngine,
    NeuralConfig,
    NetworkType
)

@pytest.fixture
def engine():
    return NeuralEngine()

def test_model_creation(engine):
    assert engine.model is not None
    assert isinstance(engine.model, torch.nn.Module)

def test_feedforward_architecture():
    config = NeuralConfig(
        architecture=NetworkType.FEEDFORWARD,
        layers=[32, 16, 8]
    )
    engine = NeuralEngine(config)
    
    # Check layer dimensions
    assert len(list(engine.model.children())) == 5  # 2 linear + 2 activation + 1 dropout
    layers = [l for l in engine.model.children() if isinstance(l, torch.nn.Linear)]
    assert layers[0].in_features == 32
    assert layers[0].out_features == 16
    assert layers[1].in_features == 16
    assert layers[1].out_features == 8

def test_activation_functions():
    # Test different activation functions
    for activation in ["relu", "tanh", "sigmoid"]:
        config = NeuralConfig(activation=activation)
        engine = NeuralEngine(config)
        
        activation_layer = [l for l in engine.model.children() 
                          if not isinstance(l, (torch.nn.Linear, torch.nn.Dropout))][0]
        
        if activation == "relu":
            assert isinstance(activation_layer, torch.nn.ReLU)
        elif activation == "tanh":
            assert isinstance(activation_layer, torch.nn.Tanh)
        elif activation == "sigmoid":
            assert isinstance(activation_layer, torch.nn.Sigmoid)

def test_invalid_architecture():
    config = NeuralConfig(architecture="invalid")
    with pytest.raises(ValueError):
        NeuralEngine(config)

def test_invalid_activation():
    config = NeuralConfig(activation="invalid")
    with pytest.raises(ValueError):
        NeuralEngine(config)

def test_model_forward_pass(engine):
    # Create sample input
    batch_size = 4
    input_size = engine.config.layers[0]
    x = torch.randn(batch_size, input_size)
    
    # Forward pass
    output = engine.model(x)
    
    assert output.shape == (batch_size, engine.config.layers[-1])

def test_dropout_configuration():
    config = NeuralConfig(dropout=0.5)
    engine = NeuralEngine(config)
    
    dropout_layers = [l for l in engine.model.children() if isinstance(l, torch.nn.Dropout)]
    assert len(dropout_layers) > 0
    assert dropout_layers[0].p == 0.5

def test_config_validation():
    # Test invalid layer configuration
    with pytest.raises(ValueError):
        NeuralConfig(layers=[])
        
    # Test invalid dropout
    with pytest.raises(ValueError):
        NeuralConfig(dropout=1.5)