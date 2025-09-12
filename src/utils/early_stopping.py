import torch

class EarlyStopping:
    """Early stopping with cleaner output - prints every 5 epochs"""
    
    def __init__(self, patience=25, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_loss, model=None):
        """Check if training should stop"""
        
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.counter = 0
            
            # Save best model state
            if model is not None:
                self.best_model_state = model.state_dict().copy()
                
            if self.verbose:
                print(f"  ✓ Validation improved to {val_loss:.4f} - saving model")
                
        else:
            # No improvement
            self.counter += 1
            
            # Only print every 5 epochs to reduce noise
            if self.verbose and (self.counter % 5 == 0 or self.counter >= self.patience):
                print(f"  • No improvement for {self.counter}/{self.patience} epochs (best: {self.best_loss:.4f})")
            
            # Check if we should stop
            if self.counter >= self.patience:
                self.early_stop = True
                
                # Restore best model if available
                if model is not None and self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)
                    if self.verbose:
                        print(f"  🛑 Early stopping! Restored best model (val_loss: {self.best_loss:.4f})")
                else:
                    if self.verbose:
                        print(f"  🛑 Early stopping! (best val_loss: {self.best_loss:.4f})")
        
        return self.early_stop
    
    def reset(self):
        """Reset early stopping state"""
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None