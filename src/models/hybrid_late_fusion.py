import torch
import torch.nn as nn
from .base_fusion import BaseFusionModel

class HybridLateFusionModel(BaseFusionModel):
    """
    Production-quality Hybrid Late Fusion
    
    Technique: Individual modality predictions + original features → meta-learner
    Key insight: Combine the prediction power of individual models with raw feature information
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Individual modality predictors
        self.predictor1 = self._build_predictor(
            config['modality1_dim'], 
            config, 
            name="predictor1"
        )
        
        self.predictor2 = self._build_predictor(
            config['modality2_dim'], 
            config, 
            name="predictor2"
        )
        
        # Meta-model: combines predictions + original features
        if config['task_type'] == 'classification':
            pred_size = 2 * config['num_classes']  # Softmax outputs from both predictors
        else:
            pred_size = 2  # Raw predictions from both predictors
            
        # Total meta-input: predictions + raw features from both modalities
        meta_input_size = pred_size + config['modality1_dim'] + config['modality2_dim']
        
        # Meta-learner with dropout for regularization
        if config['task_type'] == 'classification':
            self.meta_model = nn.Sequential(
                nn.Linear(meta_input_size, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.4)),  # Higher dropout in meta-model
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.3)),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, config['num_classes'])
            )
        else:  # regression
            self.meta_model = nn.Sequential(
                nn.Linear(meta_input_size, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.4)),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.3)),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        
        self.task_type = config['task_type']
    
    def _build_predictor(self, input_dim, config, name):
        """Build individual modality predictor with adaptive architecture"""
        
        # Adaptive hidden layers based on input size
        if input_dim > 10000:  # Brain tumor images
            hidden_dims = [512, 256, 128]
        elif input_dim > 1000:  # Twitter text
            hidden_dims = [256, 128, 64]
        else:  # Job tabular or small text
            hidden_dims = [128, 64, 32]
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.2))
            ])
            prev_dim = hidden_dim
        
        # Output layer
        output_dim = config['num_classes'] if config['task_type'] == 'classification' else 1
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, batch):
        """
        Forward pass: individual predictions + raw features → meta-model
        """
        # Extract modalities
        keys = [k for k in batch.keys() if k != 'labels']
        x1, x2 = batch[keys[0]], batch[keys[1]]
        
        # Get individual predictions
        pred1_logits = self.predictor1(x1)
        pred2_logits = self.predictor2(x2)
        
        # Convert to probabilities for classification
        if self.task_type == 'classification':
            pred1 = torch.softmax(pred1_logits, dim=1)
            pred2 = torch.softmax(pred2_logits, dim=1)
        else:
            pred1 = pred1_logits
            pred2 = pred2_logits
        
        # Hybrid approach: predictions + original features from BOTH modalities
        meta_input = torch.cat([
            pred1,  # Predictions from modality 1
            pred2,  # Predictions from modality 2
            x1,     # Original features from modality 1
            x2      # Original features from modality 2
        ], dim=1)
        
        # Meta-model makes final prediction
        return self.meta_model(meta_input)
    
    def get_individual_predictions(self, batch):
        """Get individual modality predictions for analysis"""
        keys = [k for k in batch.keys() if k != 'labels']
        x1, x2 = batch[keys[0]], batch[keys[1]]
        
        with torch.no_grad():
            pred1_logits = self.predictor1(x1)
            pred2_logits = self.predictor2(x2)
            
            if self.task_type == 'classification':
                pred1 = torch.softmax(pred1_logits, dim=1)
                pred2 = torch.softmax(pred2_logits, dim=1)
            else:
                pred1 = pred1_logits
                pred2 = pred2_logits
        
        return {
            'predictor1': pred1,
            'predictor2': pred2,
            'predictor1_logits': pred1_logits,
            'predictor2_logits': pred2_logits
        }
    
    def forward_with_individual_preds(self, batch):
        """Forward pass that returns both individual and final predictions"""
        individual_preds = self.get_individual_predictions(batch)
        final_pred = self.forward(batch)
        
        return {
            'final_prediction': final_pred,
            **individual_preds
        }
    
    def get_fusion_strategy(self):
        return "hybrid_late_fusion"

def create_hybrid_late_fusion_model(config):
    return HybridLateFusionModel(config)