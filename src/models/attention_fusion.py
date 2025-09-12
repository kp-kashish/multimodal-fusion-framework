import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_fusion import BaseFusionModel

class AttentionFusionModel(BaseFusionModel):
    """
    Production-quality Attention Fusion
    
    Technique: Cross-modal attention between modalities
    Key insight: Each modality learns to attend to relevant parts of the other modality
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Universal projection dimension for attention
        self.attention_dim = 256
        self.num_heads = 8
        
        # Project both modalities to same dimension for attention
        self.proj1 = nn.Sequential(
            nn.Linear(config['modality1_dim'], self.attention_dim),
            nn.LayerNorm(self.attention_dim),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1))
        )
        
        self.proj2 = nn.Sequential(
            nn.Linear(config['modality2_dim'], self.attention_dim),
            nn.LayerNorm(self.attention_dim),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1))
        )
        
        # Multi-head cross-attention
        self.cross_attention_1_to_2 = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.num_heads,
            dropout=config.get('dropout', 0.1),
            batch_first=True
        )
        
        self.cross_attention_2_to_1 = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.num_heads,
            dropout=config.get('dropout', 0.1),
            batch_first=True
        )
        
        # Self-attention for within-modality relationships
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim * 2,  # Concatenated features
            num_heads=self.num_heads,
            dropout=config.get('dropout', 0.1),
            batch_first=True
        )
        
        # Output layers
        output_dim = self.attention_dim * 2  # Concatenated attended features
        
        if config['task_type'] == 'classification':
            self.output_head = nn.Sequential(
                nn.Linear(output_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.3)),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.2)),
                nn.Linear(128, config['num_classes'])
            )
        else:  # regression
            self.output_head = nn.Sequential(
                nn.Linear(output_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.3)),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.2)),
                nn.Linear(128, 1)
            )
    
    def forward(self, batch):
        """
        Forward pass: project → cross-attend → self-attend → predict
        """
        # Extract modalities
        keys = [k for k in batch.keys() if k != 'labels']
        x1, x2 = batch[keys[0]], batch[keys[1]]
        batch_size = x1.size(0)
        
        # Project to attention space
        h1 = self.proj1(x1).unsqueeze(1)  # [B, 1, D]
        h2 = self.proj2(x2).unsqueeze(1)  # [B, 1, D]
        
        # Cross-modal attention: each modality attends to the other
        # h1 attending to h2 (h1 as query, h2 as key/value)
        attended_h1, attn_weights_1 = self.cross_attention_1_to_2(h1, h2, h2)
        
        # h2 attending to h1 (h2 as query, h1 as key/value)  
        attended_h2, attn_weights_2 = self.cross_attention_2_to_1(h2, h1, h1)
        
        # Combine attended features
        combined = torch.cat([attended_h1, attended_h2], dim=-1)  # [B, 1, 2*D]
        
        # Self-attention on combined features for final refinement
        final_attended, _ = self.self_attention(combined, combined, combined)
        
        # Flatten for classification/regression head
        final_features = final_attended.squeeze(1)  # [B, 2*D]
        
        return self.output_head(final_features)
    
    def get_attention_weights(self, batch):
        """Get attention weights for interpretability"""
        keys = [k for k in batch.keys() if k != 'labels']
        x1, x2 = batch[keys[0]], batch[keys[1]]
        
        with torch.no_grad():
            h1 = self.proj1(x1).unsqueeze(1)
            h2 = self.proj2(x2).unsqueeze(1)
            
            _, attn_weights_1 = self.cross_attention_1_to_2(h1, h2, h2)
            _, attn_weights_2 = self.cross_attention_2_to_1(h2, h1, h1)
            
        return {
            'modality1_to_modality2': attn_weights_1,
            'modality2_to_modality1': attn_weights_2
        }
    
    def get_fusion_strategy(self):
        return "attention_fusion"

def create_attention_fusion_model(config):
    return AttentionFusionModel(config)