from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .neural_engine import NeuralEngine

@dataclass
class TrainerConfig:
    """Neural network trainer configuration"""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "mse"
    validation_split: float = 0.2
    metadata: Dict = field(default_factory=dict)

class NetworkTrainer:
    def __init__(self, engine: NeuralEngine, config: Optional[TrainerConfig] = None):
        self.engine = engine
        self.config = config or TrainerConfig()
        self.optimizer = self._setup_optimizer()
        self.loss_fn = self._setup_loss_function()
        self.training_history: List[Dict] = []
        
    def train(self, 
             train_data: torch.Tensor,
             train_labels: torch.Tensor,
             validation_data: Optional[torch.Tensor] = None,
             validation_labels: Optional[torch.Tensor] = None) -> Dict:
        """Train neural network"""
        try:
            # Create data loaders
            train_loader = DataLoader(
                list(zip(train_data, train_labels)),
                batch_size=self.config.batch_size,
                shuffle=True
            )
            
            if validation_data is not None and validation_labels is not None:
                val_loader = DataLoader(
                    list(zip(validation_data, validation_labels)),
                    batch_size=self.config.batch_size
                )
            else:
                val_loader = None
                
            # Training loop
            for epoch in range(self.config.num_epochs):
                train_loss = self._train_epoch(train_loader)
                val_loss = self._validate_epoch(val_loader) if val_loader else None
                
                # Record metrics
                metrics = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                }
                self.training_history.append(metrics)
                
            return {
                "success": True,
                "final_train_loss": train_loss,
                "final_val_loss": val_loss,
                "epochs_completed": self.config.num_epochs
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer"""
        if self.config.optimizer == "adam":
            return torch.optim.Adam(
                self.engine.model.parameters(),
                lr=self.config.learning_rate
            )
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                self.engine.model.parameters(),
                lr=self.config.learning_rate
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
            
    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function"""
        if self.config.loss_function == "mse":
            return nn.MSELoss()
        elif self.config.loss_function == "cross_entropy":
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_function}")
            
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch"""
        self.engine.model.train()
        total_loss = 0
        
        for batch_data, batch_labels in train_loader:
            self.optimizer.zero_grad()
            outputs = self.engine.model(batch_data)
            loss = self.loss_fn(outputs, batch_labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
        
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate one epoch"""
        self.engine.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                outputs = self.engine.model(batch_data)
                loss = self.loss_fn(outputs, batch_labels)
                total_loss += loss.item()
                
        return total_loss / len(val_loader) 