import sys
import os
import urllib.request
import zipfile
import shutil
import pandas as pd

sys.path.insert(0, 'src')
from src.data.universal_data_loaderv2 import UniversalDataLoader

def test_dataset(name, source, target=None, limit=200):
    print(f"Testing {name}")
    try:
        loader = UniversalDataLoader(source, target_column=target, sample_limit=limit)
        processed_data = loader.process_data()
        splits = loader.create_splits(processed_data)
        loaders = loader.get_pytorch_loaders(batch_size=8)
        
        batch = next(iter(loaders['train']))
        print("Batch shapes:")
        for key, value in batch.items():
            print(f"  {key}: {value.shape}")
        
        info = loader.get_info()
        print(f"Modalities: {list(set(info['modalities'].values()))}")
        print(f"Total features: {info['n_features']}")
        print(f"{name} PASSED\n")
        return True
    except Exception as e:
        print(f"{name} FAILED: {e}\n")
        return False

def download_petfinder():
    print("Downloading PetFinder dataset...")
    url = "https://automl-mm-bench.s3.amazonaws.com/petfinder_kaggle.zip"
    urllib.request.urlretrieve(url, "petfinder.zip")
    
    with zipfile.ZipFile("petfinder.zip", 'r') as zip_ref:
        zip_ref.extractall("petfinder_data")
    
    # Load and preprocess
    train_data = pd.read_csv("petfinder_data/petfinder_processed/train.csv", index_col=0)
    train_data['Images'] = train_data['Images'].apply(lambda x: str(x).split(';')[0] if pd.notna(x) else 'missing.jpg')
    train_data['Images'] = train_data['Images'].apply(lambda x: f"petfinder_data/petfinder_processed/{x}" if x != 'missing.jpg' else x)
    
    train_data.head(150).to_csv("petfinder_processed.csv")
    return "petfinder_processed.csv"

def main():
    try:
        petfinder_path = download_petfinder()
        
        datasets = [
            ("Twitter Sentiment", 'osanseviero/twitter-airline-sentiment', 'airline_sentiment'),
            ("Job Salary", 'lukebarousse/data_jobs', 'salary_in_usd'),
            ("PetFinder Multimodal", petfinder_path, 'AdoptionSpeed')
        ]
        
        results = []
        for name, source, target in datasets:
            results.append(test_dataset(name, source, target))
        
        print("=" * 40)
        print(f"Results: {sum(results)}/{len(results)} passed")
        
    finally:
        # Cleanup
        for file in ['petfinder.zip', 'petfinder_processed.csv']:
            if os.path.exists(file):
                os.remove(file)
        if os.path.exists('petfinder_data'):
            shutil.rmtree('petfinder_data')

if __name__ == "__main__":
    main()