from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Any, Optional

class BaseFusionModel(ABC, nn.Module):
    """
    Abstract base class for all multimodal fusion models.
    Provides common interface and evaluation methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_metric': []}
        
    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the model"""
        pass
        
    @abstractmethod
    def get_fusion_strategy(self) -> str:
        """Return the fusion strategy name"""
        pass
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Make predictions on a batch"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch)
            if self.config.get('task_type') == 'classification':
                if outputs.size(1) == 1:  # Binary classification
                    predictions = (torch.sigmoid(outputs) > 0.5).long()
                else:  # Multi-class classification
                    predictions = torch.argmax(outputs, dim=1)
            else:  # Regression
                predictions = outputs.squeeze()
        return predictions
    
    def predict_proba(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get prediction probabilities for classification tasks"""
        if self.config.get('task_type') != 'classification':
            raise ValueError("predict_proba only available for classification tasks")
            
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch)
            if outputs.size(1) == 1:  # Binary classification
                probs = torch.sigmoid(outputs)
                probs = torch.cat([1 - probs, probs], dim=1)
            else:  # Multi-class classification
                probs = torch.softmax(outputs, dim=1)
        return probs
    
    def get_model_size(self) -> int:
        """Calculate model size in bytes"""
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        return param_size + buffer_size
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def measure_inference_time(self, batch: Dict[str, torch.Tensor], 
                             num_runs: int = 100) -> float:
        """Measure average inference time in seconds"""
        self.eval()
        times = []
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(10):
                _ = self.forward(batch)
        
        # Actual measurement
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.forward(batch)
                end_time = time.time()
                times.append(end_time - start_time)
                
        return float(np.mean(times))
    
    def save_model(self, path: str):
        """Save model state and config"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'model_class': self.__class__.__name__
        }, path)
    
    def load_model(self, path: str):
        """Load model state and config"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint.get('config', {})
        self.training_history = checkpoint.get('training_history', 
                                             {'train_loss': [], 'val_loss': [], 'val_metric': []})


def get_default_config(task_type: str = 'classification', **kwargs) -> Dict[str, Any]:
    """
    Get default configuration for fusion models
    
    Args:
        task_type: 'classification' or 'regression'
        **kwargs: Additional config parameters to override defaults
    
    Returns:
        Configuration dictionary
    """
    config = {
        # Task configuration
        'task_type': task_type,
        'num_classes': 2 if task_type == 'classification' else None,
        
        # Training parameters
        'learning_rate': 0.001,
        'batch_size': 32,
        'max_epochs': 100,
        'patience': 15,  # Early stopping patience
        
        # Model architecture
        'dropout': 0.2,
        'hidden_dim': 128,
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Override with provided kwargs
    config.update(kwargs)
    
    return config