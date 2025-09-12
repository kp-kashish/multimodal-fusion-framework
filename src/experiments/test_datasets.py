import sys
import os
import time

# Add the parent directory (src) to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from data.universal_data_loader import UniversalDataLoader

def test_dataset(dataset_name, batch_size=32):
    """Test a single dataset with full data"""
    print(f"\n{'='*60}")
    print(f"TESTING {dataset_name.upper()} DATASET - FULL DATA")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        # Create loader without debug mode for full dataset
        loader = UniversalDataLoader(dataset_name, debug_mode=False)
        
        # Load and process data
        print("Loading raw data...")
        loader.load_data()
        
        print("Processing features...")
        loader.process_data()
        
        # Get dataset info before splitting
        info = loader.get_info()
        print(f"\nDataset Info:")
        print(f"  Total samples: {info['num_samples']:,}")
        print(f"  Task type: {info['task_type']}")
        
        # Print modality shapes and memory usage
        for key, value in info.items():
            if key.endswith('_shape'):
                modality = key.replace('_shape', '')
                memory_key = f'{modality}_memory_mb'
                print(f"  {modality.capitalize()} shape: {value}")
                if memory_key in info:
                    print(f"  {modality.capitalize()} memory: {info[memory_key]:.1f} MB")
        
        if 'total_memory_mb' in info:
            print(f"  Total feature memory: {info['total_memory_mb']:.1f} MB")
        
        if 'system_memory_mb' in info:
            print(f"  System memory: {info['system_memory_mb']:.1f} MB")
        
        # Print class distribution for classification
        if info['task_type'] == 'classification':
            print(f"  Classes: {info['num_classes']}")
            print(f"  Distribution: {info['class_distribution']}")
            if 'imbalance_ratio' in info:
                print(f"  Imbalance ratio: {info['imbalance_ratio']}")
            if 'class_weights' in info:
                print(f"  Class weights: {info['class_weights']}")
        
        # Print feature statistics
        if 'feature_stats' in info:
            print(f"\nFeature Statistics:")
            for modality, stats in info['feature_stats'].items():
                print(f"  {modality.capitalize()}:")
                print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
                print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        print("\nCreating train/val/test splits...")
        loader.create_splits()
        
        print("Scaling features...")
        loader.scale_features()
        
        print("Creating DataLoaders...")
        dataloaders = loader.create_dataloaders(batch_size)
        
        # Test DataLoaders
        print(f"\nDataLoader Info:")
        for split_name, dataloader in dataloaders.items():
            print(f"  {split_name}: {len(dataloader)} batches, {len(dataloader.dataset)} samples")
        
        # Test a batch from training set
        print("\nTesting batch loading...")
        train_loader = dataloaders['train']
        batch = next(iter(train_loader))
        
        print(f"Batch shapes:")
        if dataset_name == 'brain_tumor':
            ct_batch, mri_batch, labels = batch
            print(f"  CT: {ct_batch.shape}")
            print(f"  MRI: {mri_batch.shape}")
            print(f"  Labels: {labels.shape}")
            print(f"  Label distribution in batch: {labels.unique(return_counts=True)}")
            
        elif dataset_name in ['twitter_sentiment', 'job_salary']:
            text_batch, tabular_batch, labels = batch
            print(f"  Text: {text_batch.shape}")
            print(f"  Tabular: {tabular_batch.shape}")
            print(f"  Labels: {labels.shape}")
            
            if dataset_name == 'twitter_sentiment':
                print(f"  Label distribution in batch: {labels.unique(return_counts=True)}")
            else:  # job_salary
                print(f"  Salary range in batch: [{labels.min():.0f}, {labels.max():.0f}]")
        
        # Test inference speed on a few batches
        print("\nTesting batch iteration speed...")
        batch_times = []
        for i, batch in enumerate(train_loader):
            if i >= 5:  # Test first 5 batches
                break
            start_batch = time.time()
            # Simulate minimal processing
            _ = batch
            batch_time = time.time() - start_batch
            batch_times.append(batch_time)
        
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        print(f"  Average batch load time: {avg_batch_time*1000:.2f} ms")
        
        total_time = time.time() - start_time
        print(f"\nTotal processing time: {total_time:.2f} seconds")
        print(f"✓ {dataset_name} dataset test PASSED")
        
        return True, info
        
    except Exception as e:
        print(f"\n✗ {dataset_name} dataset test FAILED")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Test all three datasets with full data"""
    print("Testing Universal Data Loader with FULL datasets...")
    print("This will take several minutes and use significant memory.")
    
    datasets_to_test = [
        ('brain_tumor', 16),      # Start with smallest 
        ('twitter_sentiment', 32),
        ('job_salary', 64)        # Largest dataset last
    ]
    
    results = {}
    dataset_info = {}
    
    for dataset_name, batch_size in datasets_to_test:
        success, info = test_dataset(dataset_name, batch_size)
        results[dataset_name] = success
        if info:
            dataset_info[dataset_name] = info
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("COMPREHENSIVE DATASET COMPARISON")
    print(f"{'='*80}")
    
    print(f"{'Dataset':<20} {'Status':<10} {'Samples':<10} {'Memory(MB)':<12} {'Task':<12}")
    print("-" * 80)
    
    for dataset_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        if dataset_name in dataset_info:
            info = dataset_info[dataset_name]
            samples = f"{info['num_samples']:,}"
            memory = f"{info.get('total_memory_mb', 0):.1f}"
            task = info['task_type']
        else:
            samples = "N/A"
            memory = "N/A"
            task = "N/A"
        
        print(f"{dataset_name:<20} {status:<10} {samples:<10} {memory:<12} {task:<12}")
    
    # Feature scale comparison
    print(f"\n{'='*80}")
    print("FEATURE SCALE ANALYSIS")
    print(f"{'='*80}")
    
    for dataset_name, info in dataset_info.items():
        print(f"\n{dataset_name.upper()}:")
        if dataset_name == 'brain_tumor':
            ct_shape = info.get('ct_shape', (0,0))
            mri_shape = info.get('mri_shape', (0,0))
            print(f"  CT features: {ct_shape[1]:,}")
            print(f"  MRI features: {mri_shape[1]:,}")
            print(f"  Scale ratio: 1:1 (balanced)")
        else:
            text_shape = info.get('text_shape', (0,0))
            tabular_shape = info.get('tabular_shape', (0,0))
            if text_shape[1] > 0 and tabular_shape[1] > 0:
                ratio = text_shape[1] / tabular_shape[1]
                print(f"  Text features: {text_shape[1]:,}")
                print(f"  Tabular features: {tabular_shape[1]:,}")
                print(f"  Scale ratio: {ratio:.1f}:1")
    
    # Class imbalance analysis
    print(f"\n{'='*80}")
    print("CLASS IMBALANCE ANALYSIS")
    print(f"{'='*80}")
    
    for dataset_name, info in dataset_info.items():
        if info['task_type'] == 'classification':
            print(f"\n{dataset_name.upper()}:")
            print(f"  Classes: {info['num_classes']}")
            print(f"  Distribution: {info['class_distribution']}")
            if 'imbalance_ratio' in info:
                print(f"  Imbalance: {info['imbalance_ratio']}")
    
    total_passed = sum(results.values())
    print(f"\n{'='*80}")
    print(f"FINAL RESULT: {total_passed}/3 datasets working")
    print(f"{'='*80}")
    
    if total_passed == 3:
        print("All datasets ready for fusion experiments!")
    else:
        print("Some datasets failed - check errors above.")

if __name__ == "__main__":
    main()