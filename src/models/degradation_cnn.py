import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
import logging
from pathlib import Path
import numpy as np

class SEHIDataset(Dataset):
    """Custom dataset for SEHI images."""
    
    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = list(self.data_dir.glob("*.png"))
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.samples[idx]
        image = torch.from_numpy(np.load(img_path))
        
        # Extract label from filename (assuming format: "sample_label.png")
        label = int(img_path.stem.split("_")[1])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class DegradationCNN(nn.Module):
    """CNN for analyzing SEHI degradation patterns."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Classification
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.logger = logging.getLogger(__name__)
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        lr: float = 0.001
    ) -> dict:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }
        
        try:
            for epoch in range(epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                if val_loader:
                    val_loss, val_acc = self.evaluate(val_loader, criterion)
                    scheduler.step(val_loss)
                    
                    history['train_loss'].append(train_loss / len(train_loader))
                    history['val_loss'].append(val_loss)
                    history['val_acc'].append(val_acc)
                    
                    self.logger.info(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {train_loss/len(train_loader):.4f} - "
                        f"Val Loss: {val_loss:.4f} - "
                        f"Val Acc: {val_acc:.4f}"
                    )
                    
            return history
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
            
    def evaluate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return val_loss / len(val_loader), correct / total 