import time
import psutil
import matplotlib.pyplot as plt
import os

class SimpleDashboard:
    """Training dashboard with controlled output frequency"""
    
    def __init__(self, title="Training Dashboard"):
        self.title = title
        self.start_time = time.time()
        self.process = psutil.Process()
        
        # History tracking
        self.time_history = []
        self.memory_history = []
        self.loss_history = []
        self.metric_history = []
        self.phase_markers = []
        
        # Control output frequency
        self.update_count = 0
        self.last_progress_time = 0
        
    def add_phase_marker(self, phase_name):
        """Add a phase marker to track training stages"""
        current_time = (time.time() - self.start_time) / 60
        self.phase_markers.append((current_time, phase_name))
        print(f"Phase: {phase_name} at {current_time:.1f} min")
    
    def update(self, loss, metric=None):
        """Update with current training metrics - controlled frequency"""
        current_time = (time.time() - self.start_time) / 60
        memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
        # Store history
        self.time_history.append(current_time)
        self.memory_history.append(memory_mb)
        self.loss_history.append(loss)
        
        if metric is not None:
            self.metric_history.append(metric)
        
        self.update_count += 1
        
        # Only print progress every 30 seconds to avoid spam
        if current_time - self.last_progress_time > 0.5:  # 30 seconds
            if metric is not None:
                print(f"    Progress - Time: {current_time:.1f}min | Loss: {loss:.4f} | Metric: {metric:.4f} | Memory: {memory_mb:.0f}MB")
            else:
                print(f"    Progress - Time: {current_time:.1f}min | Loss: {loss:.4f} | Memory: {memory_mb:.0f}MB")
            self.last_progress_time = current_time
    
    def show(self):
        """Show current status"""
        current_time = (time.time() - self.start_time) / 60
        print(f"\n{self.title} - Running for {current_time:.1f} minutes")
        
        if self.phase_markers:
            print("Phases completed:")
            for time_marker, phase in self.phase_markers:
                print(f"  - {phase} ({time_marker:.1f} min)")
        print()
    
    def save_plots(self, save_dir="./results", filename_prefix="dashboard"):
        """Save dashboard plots to results directory"""
        if not self.time_history:
            print("No data to plot")
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Determine number of subplots
            num_plots = 2 if not self.metric_history else 3
            fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))
            
            # Handle single subplot case
            if num_plots == 1:
                axes = [axes]
            elif num_plots == 2:
                axes = list(axes)
            
            # Memory usage plot
            axes[0].plot(self.time_history, self.memory_history, 'g-', linewidth=2, label='Memory Usage')
            axes[0].fill_between(self.time_history, self.memory_history, alpha=0.3, color='green')
            
            # Add phase markers
            for marker_time, phase_name in self.phase_markers:
                axes[0].axvline(x=marker_time, color='red', linestyle='--', alpha=0.7)
                axes[0].text(marker_time, max(self.memory_history) * 0.9, phase_name,
                           rotation=90, fontsize=8, ha='right')
            
            axes[0].set_title(f'{self.title} - Memory Usage')
            axes[0].set_xlabel('Time (minutes)')
            axes[0].set_ylabel('Memory (MB)')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            # Loss plot
            axes[1].plot(self.time_history, self.loss_history, 'b-', linewidth=2, label='Training Loss')
            axes[1].set_title(f'{self.title} - Training Loss')
            axes[1].set_xlabel('Time (minutes)')
            axes[1].set_ylabel('Loss')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            
            # Metric plot (if available)
            if self.metric_history and num_plots == 3:
                axes[2].plot(self.time_history[:len(self.metric_history)], self.metric_history, 'r-', linewidth=2, label='Validation Metric')
                axes[2].set_title(f'{self.title} - Validation Metric')
                axes[2].set_xlabel('Time (minutes)')
                axes[2].set_ylabel('Metric')
                axes[2].grid(True, alpha=0.3)
                axes[2].legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(save_dir, f"{filename_prefix}_plots.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()  # Important: close to free memory
            
            print(f"Dashboard plots saved to {plot_path}")
            return plot_path
            
        except Exception as e:
            print(f"Could not save dashboard plots: {e}")
            return None
    
    def get_summary(self):
        """Get summary statistics"""
        if not self.time_history:
            return "No data collected"
        
        total_time = self.time_history[-1]
        max_memory = max(self.memory_history) if self.memory_history else 0
        min_loss = min(self.loss_history) if self.loss_history else 0
        final_loss = self.loss_history[-1] if self.loss_history else 0
        
        summary = f"""
Dashboard Summary:
- Total Runtime: {total_time:.1f} minutes
- Peak Memory: {max_memory:.0f} MB
- Best Loss: {min_loss:.4f}
- Final Loss: {final_loss:.4f}
- Total Updates: {self.update_count}
- Phases: {len(self.phase_markers)}
"""
        return summary