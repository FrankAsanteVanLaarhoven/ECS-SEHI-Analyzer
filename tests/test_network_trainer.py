import pytest
import torch
import numpy as np
from src.ecs_sehi_analyzer.core.neural.neural_engine import NeuralEngine
from src.ecs_sehi_analyzer.core.neural.network_trainer import (
    NetworkTrainer,
    TrainerConfig
)

@pytest.fixture
def engine():
    return NeuralEngine()

@pytest.fixture
def trainer(engine):
    return NetworkTrainer(engine)

@pytest.fixture
def sample_data():
    # Create sample training data
    X = torch.randn(100, 64)  # 100 samples, 64 features
    y = torch.randn(100, 16)  # 100 samples, 16 outputs
    return X, y

def test_trainer_initialization(trainer):
    assert trainer.optimizer is not None
    assert trainer.loss_fn is not None
    assert len(trainer.training_history) == 0

def test_training_loop(trainer, sample_data):
    X, y = sample_data
    result = trainer.train(X, y)
    
    assert result["success"]
    assert "final_train_loss" in result
    assert "epochs_completed" in result
    assert result["epochs_completed"] == trainer.config.num_epochs

def test_validation_split(trainer, sample_data):
    X, y = sample_data
    val_size = int(len(X) * 0.2)
    X_val, y_val = X[:val_size], y[:val_size]
    
    result = trainer.train(X, y, X_val, y_val)
    
    assert result["success"]
    assert "final_val_loss" in result
    assert result["final_val_loss"] is not None

def test_optimizer_configuration():
    config = TrainerConfig(optimizer="sgd", learning_rate=0.1)
    engine = NeuralEngine()
    trainer = NetworkTrainer(engine, config)
    
    assert isinstance(trainer.optimizer, torch.optim.SGD)
    assert trainer.optimizer.param_groups[0]["lr"] == 0.1

def test_loss_function_configuration():
    config = TrainerConfig(loss_function="cross_entropy")
    engine = NeuralEngine()
    trainer = NetworkTrainer(engine, config)
    
    assert isinstance(trainer.loss_fn, torch.nn.CrossEntropyLoss)

def test_training_history(trainer, sample_data):
    X, y = sample_data
    trainer.train(X, y)
    
    assert len(trainer.training_history) == trainer.config.num_epochs
    for record in trainer.training_history:
        assert "epoch" in record
        assert "train_loss" in record

def test_invalid_optimizer():
    config = TrainerConfig(optimizer="invalid")
    engine = NeuralEngine()
    with pytest.raises(ValueError):
        NetworkTrainer(engine, config)

def test_invalid_loss_function():
    config = TrainerConfig(loss_function="invalid")
    engine = NeuralEngine()
    with pytest.raises(ValueError):
        NetworkTrainer(engine, config)

def test_batch_size_configuration(trainer, sample_data):
    X, y = sample_data
    trainer.config.batch_size = 10
    result = trainer.train(X, y)
    
    assert result["success"]
    # Should complete more steps per epoch due to smaller batch size
    assert len(trainer.training_history) == trainer.config.num_epochs