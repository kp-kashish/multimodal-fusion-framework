import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error

class ModelEvaluator:
    """Simple model evaluation for multimodal fusion"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model, dataloader, task_type, device='cpu'):
        """
        Evaluate model on test data
        
        Args:
            model: Trained model
            dataloader: Test dataloader
            task_type: 'classification' or 'regression'
            device: Device to run on
            
        Returns:
            dict: Evaluation metrics
        """
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in dataloader:
                # Handle different batch formats
                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 3:
                    batch = {
                        'modality_0': batch_data[0].to(device),
                        'modality_1': batch_data[1].to(device)
                    }
                    labels = batch_data[2]
                else:
                    batch = {k: v.to(device) for k, v in batch_data.items() if k != 'labels'}
                    labels = batch_data['labels']
                
                # Get predictions
                outputs = model(batch)
                
                if task_type == 'classification':
                    predictions = torch.argmax(outputs, dim=1)
                else:
                    predictions = outputs.squeeze()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        
        if task_type == 'classification':
            metrics = {
                'test_accuracy': accuracy_score(labels, predictions),
                'test_f1': f1_score(labels, predictions, average='weighted'),
                'model_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
        else:
            metrics = {
                'test_mae': mean_absolute_error(labels, predictions),
                'test_r2': r2_score(labels, predictions),
                'model_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
        
        return metrics
    
    def compare_models(self, results, task_type):
        """Print model comparison table"""
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        
        if task_type == 'classification':
            print(f"{'Model':<25} {'Parameters':<12} {'Test Accuracy':<15} {'Test F1':<10}")
        else:
            print(f"{'Model':<25} {'Parameters':<12} {'Test MAE':<12} {'Test R2':<10}")
        
        print("-" * 80)
        
        for model_name, metrics in results.items():
            params = metrics.get('model_parameters', 0)
            
            if task_type == 'classification':
                acc = metrics.get('test_accuracy', 0)
                f1 = metrics.get('test_f1', 0)
                print(f"{model_name:<25} {params:<12,} {acc:<15.4f} {f1:<10.4f}")
            else:
                mae = metrics.get('test_mae', 0)
                r2 = metrics.get('test_r2', 0)
                print(f"{model_name:<25} {params:<12,} {mae:<12.4f} {r2:<10.4f}")
        
        print("="*80)