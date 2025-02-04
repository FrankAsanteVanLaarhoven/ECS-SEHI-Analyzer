import pytest
import torch
from models.degradation_cnn import DegradationCNN, ModelTrainer
from torch.utils.data import DataLoader, TensorDataset

class TestDegradationCNN:
    def test_model_architecture(self):
        """Test CNN architecture."""
        model = DegradationCNN(num_classes=10)
        x = torch.randn(1, 1, 64, 64)
        output = model(x)
        
        assert output.shape == (1, 10)
        
    def test_model_training(self, sample_cnn_data):
        """Test model training."""
        X, y = sample_cnn_data
        model = DegradationCNN(num_classes=10)
        trainer = ModelTrainer(model, device="cpu")
        
        # Create data loaders
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=4)
        val_loader = DataLoader(dataset, batch_size=4)
        
        history = trainer.train(
            train_loader,
            val_loader,
            epochs=2,
            lr=0.001
        )
        
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'val_acc' in history
        assert len(history['train_loss']) == 2 