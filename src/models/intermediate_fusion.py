import torch
import torch.nn as nn
from .base_fusion import BaseFusionModel

class IntermediateFusionModel(BaseFusionModel):
    """
    Production-quality Intermediate Fusion
    
    Technique: Separate encoders → intermediate representations → fusion
    Key insight: Each modality learns its own representation, then fused at intermediate level
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Adaptive encoder sizing based on input dimensions
        dim1, dim2 = config['modality1_dim'], config['modality2_dim']
        
        # Smart intermediate sizing: larger inputs get more compression
        intermediate_dim = 128
        if max(dim1, dim2) > 10000:  # Brain tumor case
            hidden1, hidden2 = 512, 512
        elif max(dim1, dim2) > 1000:  # Twitter text case  
            hidden1, hidden2 = 256, 128
        else:  # Job salary case
            hidden1, hidden2 = 128, 64
        
        # Modality 1 encoder (e.g., text, CT images)
        self.encoder1 = nn.Sequential(
            nn.Linear(dim1, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.2)),
            nn.Linear(hidden1, intermediate_dim),
            nn.ReLU()
        )
        
        # Modality 2 encoder (e.g., tabular, MRI images)
        self.encoder2 = nn.Sequential(
            nn.Linear(dim2, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.2)),
            nn.Linear(hidden2, intermediate_dim),
            nn.ReLU()
        )
        
        # Fusion network - combines intermediate representations
        fusion_input = intermediate_dim * 2
        
        if config['task_type'] == 'classification':
            self.fusion_head = nn.Sequential(
                nn.Linear(fusion_input, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.3)),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, config['num_classes'])
            )
        else:  # regression
            self.fusion_head = nn.Sequential(
                nn.Linear(fusion_input, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.3)),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
    
    def forward(self, batch):
        """
        Forward pass: modality → encoder → intermediate repr → fusion
        """
        # Extract modalities (flexible key handling)
        keys = [k for k in batch.keys() if k != 'labels']
        x1, x2 = batch[keys[0]], batch[keys[1]]
        
        # Learn intermediate representations
        repr1 = self.encoder1(x1)  # Any size → 128
        repr2 = self.encoder2(x2)  # Any size → 128
        
        # Fuse at intermediate level (this is the "intermediate" in intermediate fusion)
        fused = torch.cat([repr1, repr2], dim=1)
        
        return self.fusion_head(fused)
    
    def get_intermediate_representations(self, batch):
        """Get intermediate representations for analysis"""
        keys = [k for k in batch.keys() if k != 'labels']
        x1, x2 = batch[keys[0]], batch[keys[1]]
        
        with torch.no_grad():
            repr1 = self.encoder1(x1)
            repr2 = self.encoder2(x2)
            
        return {'repr1': repr1, 'repr2': repr2}
    
    def get_fusion_strategy(self):
        return "intermediate_fusion"

def create_intermediate_fusion_model(config):
    return IntermediateFusionModel(config)