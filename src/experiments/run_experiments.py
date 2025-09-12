import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import modules
from data.universal_data_loader import UniversalDataLoader
from models.base_fusion import get_default_config
from models.intermediate_fusion import create_intermediate_fusion_model
from models.attention_fusion import create_attention_fusion_model
from models.hybrid_late_fusion import create_hybrid_late_fusion_model
from models.adaptive_gating_fusion import create_adaptive_gating_fusion_model
from utils.early_stopping import EarlyStopping
from utils.evaluation import ModelEvaluator
from utils.dashboard import SimpleDashboard

class FusionExperimentRunner:
    """Clean experiment runner - minimal output, saves 1 txt + 4 png files"""
    
    def __init__(self, dataset_name, debug_mode=False, batch_size=32):
        self.dataset_name = dataset_name
        self.debug_mode = debug_mode
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.data_loader = UniversalDataLoader(dataset_name, debug_mode)
        self.evaluator = ModelEvaluator()
        self.results = {}
        
        # Initialize data containers
        self.dataloaders = None
        self.dataset_info = None
        
        # Create results directory
        self.results_dir = f"./results/{dataset_name}/"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"Initialized {dataset_name} experiment runner")
    
    def prepare_data(self):
        """Load and prepare dataset"""
        self.dataloaders = self.data_loader.prepare_dataset(self.batch_size)
        self.dataset_info = self.data_loader.get_info()
        print(f"Dataset prepared: {self.dataset_info['task_type']}, {self.dataset_info['num_samples']} samples")
    
    def get_base_config(self):
        """Get configuration"""
        modality_shapes = {k: v for k, v in self.dataset_info.items() if k.endswith('_shape')}
        modality_names = [k.replace('_shape', '') for k in modality_shapes.keys()]
        
        config = get_default_config(
            task_type=self.dataset_info['task_type'],
            modality1_dim=modality_shapes[f'{modality_names[0]}_shape'][1],
            modality2_dim=modality_shapes[f'{modality_names[1]}_shape'][1],
            num_classes=self.dataset_info.get('num_classes', 2),
            learning_rate=0.0005,
            dropout=0.4,
            patience=25
        )
        
        if 'class_weights' in self.dataset_info:
            config['class_weights'] = self.dataset_info['class_weights']
        
        return config
    
    def train_model(self, model, model_name, max_epochs=100):
        """Train model with clean output"""
        print(f"\nTraining {model_name}...")
        
        # Create dashboard for this model
        dashboard = SimpleDashboard(f"{model_name} Training")
        
        # Setup training
        model = model.to(self.device)
        early_stopping = EarlyStopping(patience=25, verbose=False)  # No verbose output
        
        if self.dataset_info['task_type'] == 'classification':
            if 'class_weights' in self.dataset_info:
                weights = torch.FloatTensor(list(self.dataset_info['class_weights'].values())).to(self.device)
                criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        
        # Training loop
        for epoch in range(max_epochs):
            model.train()
            train_loss = 0.0
            
            for batch_data in self.dataloaders['train']:
                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 3:
                    batch = {
                        'modality_0': batch_data[0].to(self.device),
                        'modality_1': batch_data[1].to(self.device),
                        'labels': batch_data[2].to(self.device)
                    }
                    labels = batch['labels']
                else:
                    batch = batch_data
                    labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch)
                
                if self.dataset_info['task_type'] == 'classification':
                    loss = criterion(outputs, labels.long())
                else:
                    loss = criterion(outputs.squeeze(), labels.float())
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(self.dataloaders['train'])
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_labels = []
            
            with torch.no_grad():
                for batch_data in self.dataloaders['val']:
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 3:
                        batch = {
                            'modality_0': batch_data[0].to(self.device),
                            'modality_1': batch_data[1].to(self.device),
                            'labels': batch_data[2].to(self.device)
                        }
                        labels = batch['labels']
                    else:
                        batch = batch_data
                        labels = batch['labels'].to(self.device)
                    
                    outputs = model(batch)
                    
                    if self.dataset_info['task_type'] == 'classification':
                        loss = criterion(outputs, labels.long())
                        predictions = torch.argmax(outputs, dim=1)
                    else:
                        loss = criterion(outputs.squeeze(), labels.float())
                        predictions = outputs.squeeze()
                    
                    val_loss += loss.item()
                    val_predictions.extend(predictions.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = val_loss / len(self.dataloaders['val'])
            
            # Calculate validation metric
            if self.dataset_info['task_type'] == 'classification':
                val_metric = f1_score(val_labels, val_predictions, average='weighted')
            else:
                val_metric = r2_score(val_labels, val_predictions)
            
            # Update dashboard (no print output)
            dashboard.update(avg_train_loss, val_metric)
            
            # Print progress every 10 epochs only
            if (epoch + 1) % 10 == 0:
                if self.dataset_info['task_type'] == 'classification':
                    print(f"  Epoch {epoch + 1:3d} - Val Loss: {avg_val_loss:.4f}, F1: {val_metric:.4f}")
                else:
                    print(f"  Epoch {epoch + 1:3d} - Val Loss: {avg_val_loss:.4f}, R2: {val_metric:.4f}")
            
            # Early stopping check (silent)
            if early_stopping(avg_val_loss, model):
                print(f"  Stopped at epoch {epoch + 1}")
                break
        
        # Save dashboard plot for this model
        dashboard.save_plots(
            save_dir=self.results_dir, 
            filename_prefix=f"{model_name}"
        )
        
        # Evaluate model
        test_metrics = self.evaluator.evaluate_model(
            model, self.dataloaders['test'], 
            self.dataset_info['task_type'], self.device
        )
        
        self.results[model_name] = test_metrics
        
        # Print final result
        if self.dataset_info['task_type'] == 'classification':
            print(f"  Result: Acc={test_metrics['test_accuracy']:.3f}, F1={test_metrics['test_f1']:.3f}")
        else:
            print(f"  Result: MAE={test_metrics['test_mae']:.3f}, R2={test_metrics['test_r2']:.3f}")
        
        return model, test_metrics
    
    def run_all_experiments(self):
        """Run all experiments with minimal output"""
        if self.dataloaders is None:
            self.prepare_data()
        
        base_config = self.get_base_config()
        
        print(f"\nRunning 4 fusion models for {self.dataset_name}")
        print("-" * 50)
        
        # Train all 4 models
        models_to_train = [
            ("Intermediate_Fusion", create_intermediate_fusion_model),
            ("Attention_Fusion", create_attention_fusion_model),
            ("Hybrid_Late_Fusion", create_hybrid_late_fusion_model),
            ("Adaptive_Gating_Fusion", create_adaptive_gating_fusion_model)
        ]
        
        for model_name, create_fn in models_to_train:
            config = base_config.copy()
            model = create_fn(config)
            self.train_model(model, model_name)
        
        # Save results to text file
        self.save_results()
        
        print(f"\nExperiments complete! Results saved to {self.results_dir}")
        return self.results
    
    def save_results(self):
        """Save clean results to text file"""
        with open(f"{self.results_dir}/results.txt", "w") as f:
            f.write(f"MULTIMODAL FUSION RESULTS: {self.dataset_name.upper()}\n")
            f.write("=" * 60 + "\n\n")
            
            # Dataset info
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Task: {self.dataset_info['task_type']}\n")
            f.write(f"Samples: {self.dataset_info['num_samples']:,}\n")
            f.write(f"Device: {self.device}\n\n")
            
            # Results table
            if self.dataset_info['task_type'] == 'classification':
                f.write("MODEL RESULTS (Classification):\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Model':<25} {'Accuracy':<10} {'F1 Score':<10} {'Parameters':<12}\n")
                f.write("-" * 40 + "\n")
                for model_name, metrics in self.results.items():
                    f.write(f"{model_name:<25} {metrics['test_accuracy']:<10.3f} {metrics['test_f1']:<10.3f} {metrics['model_parameters']:<12,}\n")
            else:
                f.write("MODEL RESULTS (Regression):\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Model':<25} {'MAE':<10} {'R2 Score':<10} {'Parameters':<12}\n")
                f.write("-" * 40 + "\n")
                for model_name, metrics in self.results.items():
                    f.write(f"{model_name:<25} {metrics['test_mae']:<10.3f} {metrics['test_r2']:<10.3f} {metrics['model_parameters']:<12,}\n")
            
            f.write(f"\nFiles generated:\n")
            f.write(f"- results.txt (this file)\n")
            for model_name in self.results.keys():
                f.write(f"- {model_name}_plots.png\n")

def main():
    """Run experiments for one dataset"""
    dataset_name = 'job_salary'  
    
    try:
        runner = FusionExperimentRunner(dataset_name=dataset_name, debug_mode=False)
        results = runner.run_all_experiments()
        print(f"\n{dataset_name} experiments completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()