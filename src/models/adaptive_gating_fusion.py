import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_fusion import BaseFusionModel

class AdaptiveGatingFusion(BaseFusionModel):
    """
    Production-quality Adaptive Gating Fusion
    
    Technique: Dynamic gating mechanism that learns optimal modality weights per sample
    Key insight: Different samples benefit from different modality combinations
    
    Architecture:
    1. Encode each modality
    2. Learn sample-specific gates for each modality + interaction
    3. Adaptively combine features based on learned gates
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Universal encoding dimension
        self.encoding_dim = 256
        
        # Modality encoders - project to common space
        self.encoder1 = self._build_encoder(
            config['modality1_dim'], 
            self.encoding_dim,
            config,
            name="encoder1"
        )
        
        self.encoder2 = self._build_encoder(
            config['modality2_dim'],
            self.encoding_dim, 
            config,
            name="encoder2"
        )
        
        # Interaction encoder - learns cross-modal interactions
        interaction_input = self.encoding_dim * 2
        self.interaction_encoder = nn.Sequential(
            nn.Linear(interaction_input, self.encoding_dim),
            nn.LayerNorm(self.encoding_dim),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.2)),
            nn.Linear(self.encoding_dim, self.encoding_dim // 2),
            nn.ReLU()
        )
        
        # Adaptive gating network - learns per-sample importance
        gate_input = interaction_input  # Uses concatenated encoded features
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 3 gates: modality1, modality2, interaction
            nn.Sigmoid()  # Gates in [0,1] range
        )
        
        # Final prediction network
        final_input = self.encoding_dim * 2 + self.encoding_dim // 2  # gated features
        
        if config['task_type'] == 'classification':
            self.prediction_head = nn.Sequential(
                nn.Linear(final_input, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.3)),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.2)),
                nn.Linear(128, config['num_classes'])
            )
        else:  # regression
            self.prediction_head = nn.Sequential(
                nn.Linear(final_input, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.3)),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.2)),
                nn.Linear(128, 1)
            )
    
    def _build_encoder(self, input_dim, output_dim, config, name):
        """Build adaptive encoder based on input dimension"""
        
        # Adaptive architecture based on input size
        if input_dim > 10000:  # Large inputs (images)
            hidden_dims = [512, 384, output_dim]
        elif input_dim > 1000:  # Medium inputs (text)
            hidden_dims = [384, 256, output_dim]
        else:  # Small inputs (tabular)
            hidden_dims = [128, output_dim]
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.2))
            ])
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def forward(self, batch):
        """
        Forward pass: encode → gate → adaptively combine → predict
        """
        # Extract modalities
        keys = [k for k in batch.keys() if k != 'labels']
        x1, x2 = batch[keys[0]], batch[keys[1]]
        
        # Encode each modality to common space
        encoded1 = self.encoder1(x1)  # [B, encoding_dim]
        encoded2 = self.encoder2(x2)  # [B, encoding_dim]
        
        # Learn interaction features
        concatenated = torch.cat([encoded1, encoded2], dim=1)
        interaction_features = self.interaction_encoder(concatenated)  # [B, encoding_dim//2]
        
        # Compute adaptive gates (per-sample importance weights)
        gates = self.gate_network(concatenated)  # [B, 3]
        gate1, gate2, gate_interaction = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]
        
        # Apply adaptive gating
        gated_encoded1 = encoded1 * gate1  # Element-wise scaling
        gated_encoded2 = encoded2 * gate2
        gated_interaction = interaction_features * gate_interaction
        
        # Combine gated features
        final_features = torch.cat([
            gated_encoded1,
            gated_encoded2, 
            gated_interaction
        ], dim=1)
        
        return self.prediction_head(final_features)
    
    def get_gating_weights(self, batch):
        """Get gating weights for interpretability"""
        keys = [k for k in batch.keys() if k != 'labels']
        x1, x2 = batch[keys[0]], batch[keys[1]]
        
        with torch.no_grad():
            encoded1 = self.encoder1(x1)
            encoded2 = self.encoder2(x2)
            concatenated = torch.cat([encoded1, encoded2], dim=1)
            gates = self.gate_network(concatenated)
        
        return {
            'modality1_importance': gates[:, 0],
            'modality2_importance': gates[:, 1], 
            'interaction_importance': gates[:, 2],
            'total_gates': gates
        }
    
    def forward_with_analysis(self, batch):
        """Forward pass with detailed component analysis"""
        keys = [k for k in batch.keys() if k != 'labels']
        x1, x2 = batch[keys[0]], batch[keys[1]]
        
        # Forward pass components
        encoded1 = self.encoder1(x1)
        encoded2 = self.encoder2(x2)
        concatenated = torch.cat([encoded1, encoded2], dim=1)
        interaction_features = self.interaction_encoder(concatenated)
        gates = self.gate_network(concatenated)
        
        gate1, gate2, gate_interaction = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]
        
        gated_encoded1 = encoded1 * gate1
        gated_encoded2 = encoded2 * gate2
        gated_interaction = interaction_features * gate_interaction
        
        final_features = torch.cat([gated_encoded1, gated_encoded2, gated_interaction], dim=1)
        predictions = self.prediction_head(final_features)
        
        return {
            'predictions': predictions,
            'encoded_features': {'modality1': encoded1, 'modality2': encoded2},
            'interaction_features': interaction_features,
            'gates': {'modality1': gate1, 'modality2': gate2, 'interaction': gate_interaction},
            'gated_features': {'modality1': gated_encoded1, 'modality2': gated_encoded2, 'interaction': gated_interaction}
        }
    
    def analyze_modality_importance(self, dataloader):
        """Analyze average modality importance across dataset"""
        all_gates = []
        
        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    batch_dict = {'modality_0': batch[0], 'modality_1': batch[1]}
                else:
                    batch_dict = batch
                    
                gates_info = self.get_gating_weights(batch_dict)
                all_gates.append(gates_info['total_gates'])
        
        all_gates = torch.cat(all_gates, dim=0)
        
        return {
            'avg_modality1_importance': all_gates[:, 0].mean().item(),
            'avg_modality2_importance': all_gates[:, 1].mean().item(),
            'avg_interaction_importance': all_gates[:, 2].mean().item(),
            'std_modality1_importance': all_gates[:, 0].std().item(),
            'std_modality2_importance': all_gates[:, 1].std().item(),
            'std_interaction_importance': all_gates[:, 2].std().item()
        }
    
    def get_fusion_strategy(self):
        return "adaptive_gating_fusion"

def create_adaptive_gating_fusion_model(config):
    return AdaptiveGatingFusion(config)