import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import ast
import re
import pickle
import psutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.class_weight import compute_class_weight
from scipy.sparse import hstack, csr_matrix
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

class UniversalDataLoader:
    """Universal loader matching the three Colab experiments exactly with enhanced features"""
    
    def __init__(self, dataset_name, debug_mode=False):
        self.dataset_name = dataset_name
        self.debug_mode = debug_mode
        self.raw_data = None
        self.processed_features = {}
        self.labels = None
        self.splits = {}
        self.scalers = {}
        self.class_weights = None
        self.feature_stats = {}
        
        # Dataset configurations based on your Colab experiments
        self.configs = {
            'twitter_sentiment': {
                'task_type': 'classification',
                'data_source': 'huggingface',
                'dataset_id': 'osanseviero/twitter-airline-sentiment',
                'text_features': 5000,
                'binary_target': True,
                'normalize_features': True,
                'handle_imbalance': True
            },
            'job_salary': {
                'task_type': 'regression', 
                'data_source': 'huggingface',
                'dataset_id': 'lukebarousse/data_jobs',
                'text_features': 10000,
                'text_reducer_dim': 300,
                'filter_salary': True,
                'normalize_features': True,
                'salary_log_transform': True
            },
            'brain_tumor': {
                'task_type': 'classification',
                'data_source': 'kaggle',
                'dataset_id': 'murtozalikhon/brain-tumor-multimodal-image-ct-and-mri',
                'image_size': (128, 128),
                'max_samples': 500,
                'normalize_features': True,
                'handle_imbalance': False
            }
        }
        
        if dataset_name not in self.configs:
            raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(self.configs.keys())}")
            
        self.config = self.configs[dataset_name]
    
    def load_data(self):
        """Load raw data based on your Colab implementations"""
        print(f"Loading {self.dataset_name} dataset...")
        
        if self.config['data_source'] == 'huggingface':
            dataset = load_dataset(self.config['dataset_id'])
            self.raw_data = pd.DataFrame(dataset['train'])
            
        elif self.config['data_source'] == 'kaggle':
            import kagglehub
            path = kagglehub.dataset_download(self.config['dataset_id'])
            self.raw_data = path
            
        print(f"Data loaded from {self.config['data_source']}")
    
    def process_data(self):
        """Process data exactly like your Colab experiments"""
        if self.dataset_name == 'twitter_sentiment':
            self._process_twitter_sentiment()
        elif self.dataset_name == 'job_salary':
            self._process_job_salary()
        elif self.dataset_name == 'brain_tumor':
            self._process_brain_tumor()
        
        # Compute feature statistics and handle imbalances
        self._compute_feature_stats()
        if self.config.get('handle_imbalance') and self.config['task_type'] == 'classification':
            self._compute_class_weights()
    
    def _process_twitter_sentiment(self):
        """Process Twitter data exactly like Colab file 1"""
        df = self.raw_data.copy()
        
        if self.debug_mode:
            df = df.sample(min(1000, len(df)), random_state=42)
        
        # Extract text and labels (binary conversion)
        texts = df['text'].values
        labels = (df['airline_sentiment'] == 'positive').astype(int).values
        
        # Feature engineering like in your Colab
        le_airline = LabelEncoder()
        df['airline_encoded'] = le_airline.fit_transform(df['airline'])
        df['negativereason_confidence'].fillna(0, inplace=True)
        df['tweet_datetime'] = pd.to_datetime(df['tweet_created'])
        df['hour'] = df['tweet_datetime'].dt.hour
        df['day_of_week'] = df['tweet_datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
        df['is_peak_travel'] = df['hour'].isin([6, 7, 8, 17, 18, 19, 20]).astype(int)
        df['has_location'] = (~df['tweet_location'].isna()).astype(int)
        df['has_timezone'] = (~df['user_timezone'].isna()).astype(int)
        
        tabular_columns = [
            'airline_encoded', 'airline_sentiment_confidence', 'negativereason_confidence',
            'retweet_count', 'has_location', 'has_timezone', 'hour', 'day_of_week',
            'is_weekend', 'is_business_hours', 'is_peak_travel'
        ]
        tabular_features = df[tabular_columns].values
        
        # Text preprocessing like in your Colab
        def preprocess_text(text):
            text = text.lower()
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        texts_clean = [preprocess_text(text) for text in texts]
        
        # TF-IDF exactly like your Colab
        tfidf = TfidfVectorizer(
            max_features=self.config['text_features'],
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        text_features = tfidf.fit_transform(texts_clean).toarray()
        
        # Normalize text features to [0,1] for consistency with images
        if self.config.get('normalize_features'):
            text_scaler = MinMaxScaler()
            text_features = text_scaler.fit_transform(text_features)
            self.scalers['text_normalizer'] = text_scaler
        
        self.processed_features = {
            'text': text_features,
            'tabular': tabular_features
        }
        self.labels = labels
        
        print(f"Twitter: Text {text_features.shape}, Tabular {tabular_features.shape}")
        print(f"Positive ratio: {labels.sum()/len(labels)*100:.1f}%")
        print(f"Feature scale ratio: {text_features.shape[1]}:{tabular_features.shape[1]} = {text_features.shape[1]/tabular_features.shape[1]:.1f}:1")
    
    def _process_job_salary(self):
        """Process job data exactly like Colab file 2"""
        # Filter to salary subset like in your Colab
        salary_df = self.raw_data[self.raw_data['salary_year_avg'].notna()].copy()
        print(f"Filtered to {len(salary_df):,} samples with salary data")
        
        if self.debug_mode:
            salary_df = salary_df.sample(min(1000, len(salary_df)), random_state=42)
        
        # Text processing exactly like your Colab
        def safe_parse_skills(x):
            if pd.isna(x) or str(x).strip() in ['', 'None']:
                return []
            s = str(x).strip()
            if s.startswith('[') and s.endswith(']'):
                try:
                    parsed = ast.literal_eval(s)
                    return parsed if isinstance(parsed, list) else [str(parsed)]
                except:
                    pass
            return [t.strip() for t in re.split(r"[;,/|]", s) if t.strip()]
        
        def build_text_features(row):
            parts = []
            for col in ['job_title', 'job_title_short', 'company_name', 'job_location']:
                if pd.notna(row.get(col)):
                    parts.append(str(row[col]))
            if pd.notna(row.get('job_skills')):
                parts.extend(safe_parse_skills(row.get('job_skills')))
            return ' '.join(parts)
        
        salary_df['text_features'] = salary_df.apply(build_text_features, axis=1)
        
        # Tech-aware tokenizer like your Colab
        TECH_TOKENIZER = re.compile(r"[A-Za-z0-9][A-Za-z0-9\+\#\.\-_]*")
        
        def tech_tokenizer(s):
            s = s.lower()
            s = re.sub(r"[\\,;]+", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return TECH_TOKENIZER.findall(s)
        
        # TF-IDF with dimensionality reduction like your Colab
        tfidf = TfidfVectorizer(
            tokenizer=tech_tokenizer,
            preprocessor=None,
            lowercase=False,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            max_features=self.config['text_features'],
            sublinear_tf=True
        )
        
        text_tfidf = tfidf.fit_transform(salary_df['text_features'])
        text_reducer = TruncatedSVD(n_components=self.config['text_reducer_dim'], random_state=42)
        text_features = text_reducer.fit_transform(text_tfidf)
        
        # Normalize text features to [0,1] for consistency
        if self.config.get('normalize_features'):
            text_scaler = MinMaxScaler()
            text_features = text_scaler.fit_transform(text_features)
            self.scalers['text_normalizer'] = text_scaler
        
        # Tabular processing exactly like your Colab
        def target_encode_smoothed(series, target, alpha=50):
            s = series.fillna('unknown')
            gm = target.mean()
            agg = pd.DataFrame({'c': s, 'y': target}).groupby('c')['y'].agg(['mean','count'])
            enc = (agg['count']*agg['mean'] + alpha*gm) / (agg['count'] + alpha)
            return s.map(enc).fillna(gm).values.reshape(-1,1)
        
        def parse_skills(x):
            if pd.isna(x) or x == 'None': 
                return []
            try:
                obj = ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith('{') else x
                if isinstance(obj, dict):
                    out = []
                    for v in obj.values():
                        if isinstance(v, list):
                            out += [str(s).strip().lower() for s in v]
                    return out
            except Exception:
                return []
            return []
        
        y = salary_df['salary_year_avg'].values
        
        # Apply log transformation to reduce salary variance
        if self.config.get('salary_log_transform'):
            y = np.log1p(y)  # log(1+x) to handle zeros safely
            self.scalers['salary_log_transform'] = True
        
        # Target encoded columns
        te_cols = {
            'company_name': salary_df['company_name'],
            'job_location': salary_df['job_location'],
            'job_country': salary_df['job_country'],
            'job_via': salary_df['job_via'],
            'job_schedule_type': salary_df['job_schedule_type'],
        }
        te_blocks = [csr_matrix(target_encode_smoothed(col, y)) for col in te_cols.values()]
        
        # One-hot encode job title
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        X_title_short = ohe.fit_transform(salary_df[['job_title_short']])
        
        # Skills processing
        from sklearn.preprocessing import MultiLabelBinarizer
        skills_lists = salary_df['job_type_skills'].apply(parse_skills)
        mlb = MultiLabelBinarizer(sparse_output=True)
        X_skills = mlb.fit_transform(skills_lists)
        svd = TruncatedSVD(n_components=10, random_state=42)
        X_skills_red = csr_matrix(svd.fit_transform(X_skills))
        
        # Temporal features
        d = pd.to_datetime(salary_df['job_posted_date'])
        m = d.dt.month.values
        month_sin = csr_matrix(np.sin(2*np.pi*m/12).reshape(-1,1))
        month_cos = csr_matrix(np.cos(2*np.pi*m/12).reshape(-1,1))
        
        # Boolean features
        bool_blocks = csr_matrix(np.column_stack([
            salary_df['job_work_from_home'].astype(int).values,
            salary_df['job_no_degree_mention'].astype(int).values,
            salary_df['job_health_insurance'].astype(int).values
        ]))
        
        # Combine all tabular features
        tabular_features = hstack(
            te_blocks + [X_title_short, X_skills_red, month_sin, month_cos, bool_blocks],
            format='csr'
        ).toarray()
        
        self.processed_features = {
            'text': text_features,
            'tabular': tabular_features
        }
        self.labels = y
        
        print(f"Job Salary: Text {text_features.shape}, Tabular {tabular_features.shape}")
        print(f"Feature scale ratio: {text_features.shape[1]}:{tabular_features.shape[1]} = {text_features.shape[1]/tabular_features.shape[1]:.1f}:1")
        print(f"Salary range: ${np.min(salary_df['salary_year_avg'].values):,.0f} - ${np.max(salary_df['salary_year_avg'].values):,.0f}")
        print(f"Variance explained: {text_reducer.explained_variance_ratio_.sum():.3f}")
    
    def _process_brain_tumor(self):
        """Process brain tumor data exactly like Colab file 3"""
        dataset_path = Path(self.raw_data) / 'Dataset'
        ct_path = dataset_path / 'Brain Tumor CT scan Images'
        mri_path = dataset_path / 'Brain Tumor MRI images'
        
        def load_images(modality_path, class_name, target_size=(128, 128), max_samples=500):
            path = modality_path / class_name
            images = []
            files = sorted([f for f in path.iterdir() if f.suffix.lower() in ['.jpg', '.png']])
            
            # Sample evenly if too many files
            if len(files) > max_samples:
                step = len(files) // max_samples
                files = files[::step][:max_samples]
            
            for file in files:
                img = cv2.imread(str(file))
                img = cv2.resize(img, target_size)
                img = img.astype(np.float32) / 255.0  # Already normalized to [0,1]
                images.append(img.flatten())
            
            return np.array(images)
        
        # Load exactly like your Colab
        max_samples = self.config['max_samples'] if not self.debug_mode else 50
        size = self.config['image_size']
        
        ct_tumor = load_images(ct_path, 'Tumor', size, max_samples)
        ct_healthy = load_images(ct_path, 'Healthy', size, max_samples)
        mri_tumor = load_images(mri_path, 'Tumor', size, max_samples)
        mri_healthy = load_images(mri_path, 'Healthy', size, max_samples)
        
        ct_data = np.vstack([ct_tumor, ct_healthy])
        mri_data = np.vstack([mri_tumor, mri_healthy])
        labels = np.hstack([np.ones(len(ct_tumor)), np.zeros(len(ct_healthy))])
        
        self.processed_features = {
            'ct': ct_data,
            'mri': mri_data
        }
        self.labels = labels
        
        print(f"Brain Tumor: CT {ct_data.shape}, MRI {mri_data.shape}")
        print(f"Feature scale ratio: {ct_data.shape[1]}:{mri_data.shape[1]} = 1:1 (balanced)")
        print(f"Tumor ratio: {labels.sum()/len(labels)*100:.1f}%")
    
    def _compute_feature_stats(self):
        """Compute feature statistics for analysis"""
        self.feature_stats = {}
        
        for modality_name, features in self.processed_features.items():
            self.feature_stats[modality_name] = {
                'shape': features.shape,
                'mean': np.mean(features),
                'std': np.std(features),
                'min': np.min(features),
                'max': np.max(features),
                'memory_mb': features.nbytes / (1024 * 1024)
            }
    
    def _compute_class_weights(self):
        """Compute class weights for imbalanced datasets"""
        if self.config['task_type'] == 'classification':
            unique_classes = np.unique(self.labels)
            weights = compute_class_weight('balanced', classes=unique_classes, y=self.labels)
            self.class_weights = dict(zip(unique_classes, weights))
            print(f"Class weights computed: {self.class_weights}")
    
    def create_splits(self, test_size=0.2, val_size=0.1, random_state=42):
        """Create splits exactly like your Colab experiments"""
        if not self.processed_features or self.labels is None:
            raise ValueError("Must process data first")
        
        # Input validation
        if not (0 < test_size < 1) or not (0 <= val_size < 1):
            raise ValueError("test_size must be in (0,1), val_size must be in [0,1)")
        if test_size + val_size >= 1:
            raise ValueError("test_size + val_size must be < 1")
        
        # Get all modalities and indices for splitting
        modality_names = list(self.processed_features.keys())
        n_samples = len(self.labels)
        
        # Create indices for splitting
        indices = np.arange(n_samples)
        
        # Stratify for classification tasks
        stratify = self.labels if self.config['task_type'] == 'classification' else None
        
        # First split: train+val vs test (split indices)
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # Second split: train vs val (if val_size > 0)
        if val_size > 0:
            val_size_adj = val_size / (1 - test_size)
            train_val_labels = self.labels[train_val_idx]
            stratify_tv = train_val_labels if self.config['task_type'] == 'classification' else None
            
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_size_adj,
                random_state=random_state,
                stratify=stratify_tv
            )
        else:
            train_idx, val_idx = train_val_idx, np.array([])
        
        # Create splits using indices
        # Create val split data
        if len(val_idx) > 0:
            val_data = {name: self.processed_features[name][val_idx] for name in modality_names}
            val_labels = self.labels[val_idx]
        else:
            val_data = {name: np.array([]) for name in modality_names}
            val_labels = np.array([])
        
        self.splits = {
            'train': {
                **{name: self.processed_features[name][train_idx] for name in modality_names},
                'labels': self.labels[train_idx]
            },
            'val': {
                **val_data,
                'labels': val_labels
            },
            'test': {
                **{name: self.processed_features[name][test_idx] for name in modality_names},
                'labels': self.labels[test_idx]
            }
        }
        
        print(f"Splits: Train {len(train_idx)}, Val {len(val_idx)}, Test {len(test_idx)}")
    
    def scale_features(self):
        """Scale features using StandardScaler like your Colab"""
        modality_names = list(self.processed_features.keys())
        
        for modality_name in modality_names:
            # Skip if already normalized (images and optionally text)
            if modality_name in ['ct', 'mri']:  # Images already normalized
                continue
            if modality_name == 'text' and 'text_normalizer' in self.scalers:
                continue
                
            scaler = StandardScaler()
            
            # Fit on training data
            train_features = self.splits['train'][modality_name]
            if len(train_features) > 0:
                self.splits['train'][modality_name] = scaler.fit_transform(train_features)
                
                # Transform val and test
                for split_name in ['val', 'test']:
                    features = self.splits[split_name][modality_name]
                    if len(features) > 0:
                        self.splits[split_name][modality_name] = scaler.transform(features)
                
                self.scalers[modality_name] = scaler
    
    def create_dataloaders(self, batch_size=32):
        """Create DataLoaders exactly like your Colab setup"""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
            
        dataloaders = {}
        modality_names = list(self.processed_features.keys())
        
        for split_name, split_data in self.splits.items():
            if len(split_data['labels']) == 0:
                continue
            
            # Convert to tensors
            feature_tensors = []
            for modality_name in modality_names:
                features = split_data[modality_name]
                feature_tensors.append(torch.FloatTensor(features))
            
            # Convert labels
            labels = split_data['labels']
            if self.config['task_type'] == 'classification':
                label_tensor = torch.LongTensor(labels.astype(int))
            else:
                label_tensor = torch.FloatTensor(labels.astype(float))
            
            # Create DataLoader
            dataset = TensorDataset(*feature_tensors, label_tensor)
            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split_name == 'train'),
                num_workers=0 if self.debug_mode else 2,  # Laptop friendly
                pin_memory=torch.cuda.is_available(),
                drop_last=(split_name == 'train')
            )
        
        return dataloaders
    
    def prepare_dataset(self, batch_size=32):
        """Complete pipeline exactly matching your Colab workflow"""
        print(f"Preparing {self.dataset_name} dataset...")
        
        # Full pipeline
        self.load_data()
        self.process_data()
        self.create_splits()
        self.scale_features()
        
        # Create DataLoaders
        dataloaders = self.create_dataloaders(batch_size)
        
        print("Dataset preparation complete!")
        return dataloaders
    
    def get_info(self):
        """Get dataset information like your Colab outputs with memory usage"""
        info = {
            'dataset': self.dataset_name,
            'task_type': self.config['task_type'],
            'num_samples': len(self.labels) if self.labels is not None else 0,
        }
        
        if self.processed_features:
            total_memory = 0
            for name, features in self.processed_features.items():
                memory_mb = features.nbytes / (1024 * 1024)
                info[f'{name}_shape'] = features.shape
                info[f'{name}_memory_mb'] = memory_mb
                total_memory += memory_mb
            info['total_memory_mb'] = total_memory
        
        if self.feature_stats:
            info['feature_stats'] = self.feature_stats
        
        if self.class_weights:
            info['class_weights'] = self.class_weights
        
        if self.config['task_type'] == 'classification' and self.labels is not None:
            unique_labels = np.unique(self.labels)
            info['num_classes'] = len(unique_labels)
            info['class_distribution'] = dict(zip(*np.unique(self.labels, return_counts=True)))
            
            # Add imbalance ratio
            counts = list(info['class_distribution'].values())
            if len(counts) == 2:
                info['imbalance_ratio'] = f"{max(counts)}:{min(counts)} = {max(counts)/min(counts):.1f}:1"
        
        # Add system memory usage
        process = psutil.Process()
        info['system_memory_mb'] = process.memory_info().rss / (1024 * 1024)
        
        return info
    
    def save_processed_data(self, filepath):
        """Save processed features for faster re-runs"""
        save_dict = {
            'processed_features': self.processed_features,
            'labels': self.labels,
            'scalers': self.scalers,
            'config': self.config,
            'feature_stats': self.feature_stats,
            'class_weights': self.class_weights
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Processed data saved to {filepath}")
    
    def load_processed_data(self, filepath):
        """Load processed features for faster re-runs"""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.processed_features = save_dict['processed_features']
        self.labels = save_dict['labels']
        self.scalers = save_dict['scalers']
        self.config = save_dict['config']
        self.feature_stats = save_dict.get('feature_stats', {})
        self.class_weights = save_dict.get('class_weights', None)
        
        print(f"Processed data loaded from {filepath}")

# Factory function for easy usage
def load_multimodal_dataset(dataset_name, debug_mode=False, batch_size=32):
    """
    Load and prepare dataset exactly like your Colab experiments
    
    Args:
        dataset_name: 'twitter_sentiment', 'job_salary', or 'brain_tumor'
        debug_mode: Use smaller samples for testing
        batch_size: Batch size for DataLoaders
    
    Returns:
        dict: DataLoaders for train/val/test splits
    """
    loader = UniversalDataLoader(dataset_name, debug_mode)
    return loader.prepare_dataset(batch_size)