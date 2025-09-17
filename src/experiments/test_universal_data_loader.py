import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
data_dir = os.path.join(parent_dir, 'data')

sys.path.insert(0, root_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, data_dir)

try:
    from src.data.universal_data_loaderv2 import UniversalDataLoader
except ImportError:
    try:
        from data.universal_data_loaderv2 import UniversalDataLoader
    except ImportError:
        try:
            from src.data.universal_data_loaderv2 import UniversalDataLoader
        except ImportError:
            print("Cannot find universal_data_loaderv2.py")
            print(f"Current dir: {current_dir}")
            print(f"Data dir: {data_dir}")
            print(f"Files in data dir: {os.listdir(data_dir) if os.path.exists(data_dir) else 'Not found'}")
            sys.exit(1)

def get_dataset_config(dataset_name):
    configs = {
        'twitter_sentiment': {
            'data_source': 'huggingface',
            'dataset_id': 'mteb/tweet_sentiment_extraction',
            'target_column': 'label_text',
            'task_type': 'classification',
            'sample_limit': 1000,
            'expected_samples': 1000
        },
        'job_salary': {
            'data_source': 'huggingface',
            'dataset_id': 'lukebarousse/data_jobs',
            'target_column': 'salary_in_usd',
            'task_type': 'regression',
            'sample_limit': 1000,
            'expected_samples': 1000
        }
    }
    return configs.get(dataset_name)

def test_dataset(dataset_name):
    print(f"Testing {dataset_name}")
    
    config = get_dataset_config(dataset_name)
    if config is None:
        print(f"No config for {dataset_name}")
        return False
        
    try:
        loader = UniversalDataLoader(config)
        df = loader.load_dataset()
        print(f"Loaded {len(df)} samples")
        
        modalities = loader.auto_detect_modalities(df)
        processed_data = loader.process_modalities(df, modalities)
        labels = loader.prepare_labels(df)
        splits = loader.create_splits(processed_data, labels)
        pytorch_loaders = loader.get_pytorch_loaders(batch_size=16)
        
        train_loader = pytorch_loaders['train']
        batch = next(iter(train_loader))
        
        print(f"Batch shapes:")
        for key, value in batch.items():
            print(f"  {key}: {value.shape}")
        
        info = loader.get_info()
        print(f"Processing mode: {info['processing_mode']}")
        print(f"Total features: {info['total_features']}")
        
        print(f"{dataset_name} PASSED")
        return True
        
    except Exception as e:
        print(f"{dataset_name} FAILED: {e}")
        return False

def test_all():
    datasets = ['twitter_sentiment', 'job_salary']
    results = {}
    
    for dataset_name in datasets:
        results[dataset_name] = test_dataset(dataset_name)
        print("-" * 40)
    
    print("Results:")
    for dataset_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{dataset_name}: {status}")

if __name__ == "__main__":
    import torch
    import numpy as np
    test_all()