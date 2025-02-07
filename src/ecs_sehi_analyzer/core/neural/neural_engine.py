from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from enum import Enum
import torch
import torch.nn as nn

class NetworkType(Enum):
    FEEDFORWARD = "feedforward"
    CONVOLUTIONAL = "convolutional"
    RECURRENT = "recurrent"
    TRANSFORMER = "transformer"

@dataclass
class NeuralConfig:
    """Neural network configuration"""
    architecture: NetworkType = NetworkType.FEEDFORWARD
    layers: List[int] = field(default_factory=lambda: [64, 32, 16])
    activation: str = "relu"
    dropout: float = 0.1
    metadata: Dict = field(default_factory=dict)

class NeuralEngine:
    def __init__(self, config: Optional[NeuralConfig] = None):
        self.config = config or NeuralConfig()
        self.model = self._build_model()
        self.training_history: List[Dict] = []
        
    def _build_model(self) -> nn.Module:
        """Build neural network model"""
        if self.config.architecture == NetworkType.FEEDFORWARD:
            return self._build_feedforward()
        elif self.config.architecture == NetworkType.CONVOLUTIONAL:
            return self._build_convolutional()
        elif self.config.architecture == NetworkType.RECURRENT:
            return self._build_recurrent()
        else:
            raise ValueError(f"Unsupported architecture: {self.config.architecture}")
            
    def _build_feedforward(self) -> nn.Module:
        """Build feedforward neural network"""
        layers = []
        for i in range(len(self.config.layers) - 1):
            layers.append(nn.Linear(self.config.layers[i], self.config.layers[i+1]))
            if i < len(self.config.layers) - 2:
                layers.append(self._get_activation())
                layers.append(nn.Dropout(self.config.dropout))
                
        return nn.Sequential(*layers)
        
    def _get_activation(self) -> nn.Module:
        """Get activation function"""
        if self.config.activation == "relu":
            return nn.ReLU()
        elif self.config.activation == "tanh":
            return nn.Tanh()
        elif self.config.activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {self.config.activation}") 