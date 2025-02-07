import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go

@dataclass
class NetworkConfig:
    """Neural network configuration"""
    input_dim: Tuple[int, int] = (512, 512)
    hidden_layers: List[int] = (1024, 512, 256)
    output_dim: int = 1
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    dropout_rate: float = 0.3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class MaterialAnalysisNetwork(nn.Module):
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Calculate flattened size
        with torch.no_grad():
            x = torch.randn(1, 1, *config.input_dim)
            x = self.features(x)
            flat_size = x.view(1, -1).size(1)
        
        # Prediction layers
        layers = []
        prev_size = flat_size
        
        for hidden_size in config.hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, config.output_dim))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class MaterialAnalyzer:
    def __init__(self, config: Optional[NetworkConfig] = None):
        self.config = config or NetworkConfig()
        self.model = MaterialAnalysisNetwork(self.config).to(self.config.device)
        self.optimizer = optim.Adam(self.model.parameters(), 
                                  lr=self.config.learning_rate)
        self.criterion = nn.MSELoss()
        self.training_history: List[Dict] = []
        self.validation_metrics: Dict[str, List[float]] = {
            "loss": [],
            "accuracy": []
        }
        
    def train(self, 
             train_data: np.ndarray, 
             train_labels: np.ndarray,
             validation_data: Optional[np.ndarray] = None,
             validation_labels: Optional[np.ndarray] = None) -> Dict:
        """Train the neural network"""
        self.model.train()
        total_batches = len(train_data) // self.config.batch_size
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            
            # Shuffle data
            indices = np.random.permutation(len(train_data))
            train_data = train_data[indices]
            train_labels = train_labels[indices]
            
            for batch in range(total_batches):
                start_idx = batch * self.config.batch_size
                end_idx = start_idx + self.config.batch_size
                
                batch_data = torch.tensor(
                    train_data[start_idx:end_idx],
                    dtype=torch.float32
                ).to(self.config.device)
                
                batch_labels = torch.tensor(
                    train_labels[start_idx:end_idx],
                    dtype=torch.float32
                ).to(self.config.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            # Validation
            if validation_data is not None and validation_labels is not None:
                val_metrics = self.validate(validation_data, validation_labels)
                self.validation_metrics["loss"].append(val_metrics["loss"])
                self.validation_metrics["accuracy"].append(val_metrics["accuracy"])
                
            # Record training history
            self.training_history.append({
                "epoch": epoch + 1,
                "loss": epoch_loss / total_batches,
                "timestamp": datetime.now()
            })
            
        return {
            "final_loss": epoch_loss / total_batches,
            "epochs_completed": self.config.epochs,
            "validation_metrics": self.validation_metrics
        }
        
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            tensor_data = torch.tensor(
                data,
                dtype=torch.float32
            ).to(self.config.device)
            
            predictions = self.model(tensor_data)
            return predictions.cpu().numpy()
            
    def validate(self, data: np.ndarray, labels: np.ndarray) -> Dict:
        """Validate model performance"""
        self.model.eval()
        with torch.no_grad():
            tensor_data = torch.tensor(
                data,
                dtype=torch.float32
            ).to(self.config.device)
            
            tensor_labels = torch.tensor(
                labels,
                dtype=torch.float32
            ).to(self.config.device)
            
            outputs = self.model(tensor_data)
            loss = self.criterion(outputs, tensor_labels)
            
            # Calculate accuracy (for regression, use R² score)
            accuracy = 1 - torch.mean((outputs - tensor_labels)**2) / \
                torch.var(tensor_labels)
                
            return {
                "loss": loss.item(),
                "accuracy": accuracy.item()
            }
            
    def render_training_interface(self):
        """Render Streamlit training interface"""
        st.markdown("### 🧠 Neural Network Training")
        
        # Training metrics
        if self.training_history:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Latest Loss",
                    f"{self.training_history[-1]['loss']:.4f}"
                )
                
            with col2:
                if self.validation_metrics["accuracy"]:
                    st.metric(
                        "Validation Accuracy",
                        f"{self.validation_metrics['accuracy'][-1]:.2%}"
                    )
                    
            # Training progress plot
            fig = go.Figure()
            
            # Training loss
            epochs = [h["epoch"] for h in self.training_history]
            losses = [h["loss"] for h in self.training_history]
            
            fig.add_trace(go.Scatter(
                x=epochs,
                y=losses,
                name="Training Loss",
                mode="lines+markers"
            ))
            
            # Validation metrics
            if self.validation_metrics["loss"]:
                fig.add_trace(go.Scatter(
                    x=epochs,
                    y=self.validation_metrics["loss"],
                    name="Validation Loss",
                    mode="lines+markers"
                ))
                
            fig.update_layout(
                title="Training Progress",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        # Model architecture
        with st.expander("Model Architecture"):
            st.code(str(self.model))
            
        # Training parameters
        with st.expander("Training Parameters"):
            st.markdown(f"""
            - Learning Rate: {self.config.learning_rate}
            - Batch Size: {self.config.batch_size}
            - Epochs: {self.config.epochs}
            - Device: {self.config.device}
            - Dropout Rate: {self.config.dropout_rate}
            """) 