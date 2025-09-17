import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from datasets import load_dataset
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class DataDetector:
    """Automatic data source and modality detection"""
    
    @staticmethod
    def load_data(source) -> pd.DataFrame:
        """Smart data loading from various sources"""
        if isinstance(source, pd.DataFrame):
            return source
            
        if isinstance(source, str):
            if source.endswith('.csv'):
                return pd.read_csv(source)
            elif source.endswith('.json'):
                return pd.read_json(source)
            elif source.endswith('.parquet'):
                return pd.read_parquet(source)
            elif os.path.isdir(source):
                return DataDetector._load_image_folder(source)
            else:
                # Try as HuggingFace dataset
                try:
                    dataset = load_dataset(source, split='train')
                    return pd.DataFrame(dataset)
                except:
                    raise ValueError(f"Cannot load data from: {source}")
        
        raise ValueError(f"Unsupported data source type: {type(source)}")
    
    @staticmethod
    def _load_image_folder(folder_path: str) -> pd.DataFrame:
        """Load images from folder structure"""
        image_paths = []
        labels = []
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_paths.append(os.path.join(root, file))
                    # Use folder name as label
                    labels.append(os.path.basename(root))
        
        return pd.DataFrame({'image_path': image_paths, 'label': labels})
    
    @staticmethod
    def detect_target_column(df: pd.DataFrame, target_hint: Optional[str] = None) -> str:
        """Detect target column automatically"""
        # ALWAYS prioritize user-provided target column
        if target_hint and target_hint in df.columns:
            print(f"Using user-specified target column: {target_hint}")
            return target_hint
            
        # Common target column names (ordered by priority)
        candidates = [
            'label', 'target', 'class', 'output', 'y', 'sentiment', 
            'price', 'score', 'rating', 'category', 'outcome', 'prediction'
        ]
        
        # Look for exact matches first
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
                
        # Look for columns containing these keywords
        for candidate in candidates:
            for col in df.columns:
                if candidate in col.lower():
                    return col
                    
        # Avoid obviously non-target columns
        avoid_columns = ['id', 'index', 'name', 'date', 'time', 'url', 'path']
        potential_targets = []
        
        for col in df.columns:
            if not any(avoid in col.lower() for avoid in avoid_columns):
                potential_targets.append(col)
        
        # Return first potential target or last column as fallback
        return potential_targets[0] if potential_targets else df.columns[-1]
    
    @staticmethod
    def detect_modalities(df: pd.DataFrame, target_column: str) -> Dict[str, str]:
        """Detect modality for each column"""
        modalities = {}
        
        for column in df.columns:
            if column == target_column:
                continue
                
            # Sample non-null values
            sample_data = df[column].dropna()
            if len(sample_data) == 0:
                continue
                
            sample = sample_data.iloc[0]
            modality = DataDetector._classify_column(sample_data, column)
            modalities[column] = modality
            
        return modalities
    
    @staticmethod
    def _classify_column(series: pd.Series, column_name: str) -> str:
        """Classify column into text/tabular/image"""
        sample = series.iloc[0]
        
        # Image detection
        if isinstance(sample, str):
            if any(ext in str(sample).lower() for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
                return 'image'
            if str(sample).startswith(('http', 'data:image')):
                return 'image'
        
        # Text detection - check multiple samples
        if series.dtype == 'object':
            text_samples = series.dropna().head(10).astype(str)
            avg_length = text_samples.str.len().mean()
            avg_words = text_samples.str.split().str.len().mean()
            
            # Text if average length > 20 and has multiple words
            if avg_length > 20 and avg_words > 2:
                return 'text'
        
        return 'tabular'
    
    @staticmethod
    def detect_task_type(df: pd.DataFrame, target_column: str) -> str:
        """Detect classification vs regression"""
        target = df[target_column]
        
        if target.dtype in ['object', 'category']:
            return 'classification'
        
        if target.dtype in ['int64', 'int32']:
            unique_ratio = target.nunique() / len(target)
            if unique_ratio < 0.1 and target.nunique() <= 20:
                return 'classification'
        
        return 'regression'

class TextProcessor:
    """Robust text preprocessing with fallback strategies"""
    
    def __init__(self, max_features: int = 5000, use_transformers: bool = True):
        self.max_features = max_features
        self.use_transformers = use_transformers
        self.processor = None
        self.method = None
        
    def fit(self, texts: List[str]) -> 'TextProcessor':
        """Fit with fallback strategy"""
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        # Try sentence transformers first
        if self.use_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                self.processor = SentenceTransformer('all-MiniLM-L6-v2')
                self.method = 'transformers'
                print("Using sentence transformers for text processing")
                return self
            except Exception:
                print("Sentence transformers failed, falling back to TF-IDF")
        
        # Fallback to TF-IDF
        self.processor = TfidfVectorizer(
            max_features=self.max_features,
            max_df=0.95,
            min_df=2,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.processor.fit(cleaned_texts)
        self.method = 'tfidf'
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to features"""
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        if self.method == 'transformers':
            return self.processor.encode(cleaned_texts, show_progress_bar=False, batch_size=32)
        else:
            features = self.processor.transform(cleaned_texts).toarray()
            
            # Apply SVD if too many features
            if features.shape[1] > self.max_features:
                svd = TruncatedSVD(n_components=self.max_features)
                features = svd.fit_transform(features)
                
            return features
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

class TabularProcessor:
    """Robust tabular data preprocessing"""
    
    def __init__(self):
        self.numeric_processors = {}
        self.categorical_processors = {}
        self.feature_names = []
        self.column_types = {}
        
    def fit(self, df: pd.DataFrame) -> 'TabularProcessor':
        """Fit processors for each column"""
        for column in df.columns:
            column_type = self._detect_column_type(df[column])
            self.column_types[column] = column_type
            
            if column_type == 'numeric':
                self._fit_numeric(df[column], column)
            else:
                self._fit_categorical(df[column], column)
        
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform tabular data"""
        processed_features = []
        
        for column in df.columns:
            if column not in self.column_types:
                continue
                
            if self.column_types[column] == 'numeric':
                features = self._transform_numeric(df[column], column)
            else:
                features = self._transform_categorical(df[column], column)
            
            if features is not None:
                processed_features.append(features)
        
        return np.hstack(processed_features) if processed_features else np.array([]).reshape(len(df), 0)
    
    def _detect_column_type(self, series: pd.Series) -> str:
        """Detect if column is numeric or categorical"""
        if series.dtype in ['int64', 'float64']:
            # Check if it's actually categorical
            unique_ratio = series.nunique() / len(series)
            if series.nunique() <= 10 or unique_ratio < 0.05:
                return 'categorical'
            return 'numeric'
        return 'categorical'
    
    def _fit_numeric(self, series: pd.Series, column: str):
        """Fit numeric column processors"""
        imputer = SimpleImputer(strategy='median')
        scaler = RobustScaler()
        
        # Fit imputer
        values = series.values.reshape(-1, 1)
        imputed = imputer.fit_transform(values)
        
        # Fit scaler
        scaler.fit(imputed)
        
        self.numeric_processors[column] = {
            'imputer': imputer,
            'scaler': scaler
        }
        self.feature_names.append(column)
    
    def _fit_categorical(self, series: pd.Series, column: str):
        """Fit categorical column processors"""
        # Handle missing values and convert to string
        filled_series = series.fillna('missing').astype(str)
        unique_count = filled_series.nunique()
        
        if unique_count <= 10:
            # One-hot encoding for low cardinality
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoder.fit(filled_series.values.reshape(-1, 1))
            self.categorical_processors[column] = ('onehot', encoder)
            categories = encoder.categories_[0]
            self.feature_names.extend([f"{column}_{cat}" for cat in categories])
        else:
            # Label encoding for high cardinality
            encoder = LabelEncoder()
            encoder.fit(filled_series)
            self.categorical_processors[column] = ('label', encoder)
            self.feature_names.append(f"{column}_encoded")
    
    def _transform_numeric(self, series: pd.Series, column: str) -> np.ndarray:
        """Transform numeric column"""
        if column not in self.numeric_processors:
            return None
            
        processors = self.numeric_processors[column]
        values = series.values.reshape(-1, 1)
        
        # Apply imputer then scaler
        imputed = processors['imputer'].transform(values)
        scaled = processors['scaler'].transform(imputed)
        
        return scaled
    
    def _transform_categorical(self, series: pd.Series, column: str) -> np.ndarray:
        """Transform categorical column"""
        if column not in self.categorical_processors:
            return None
            
        encoder_type, encoder = self.categorical_processors[column]
        filled_series = series.fillna('missing').astype(str)
        
        if encoder_type == 'onehot':
            return encoder.transform(filled_series.values.reshape(-1, 1))
        else:
            return encoder.transform(filled_series).reshape(-1, 1)

class ImageProcessor:
    """Robust image processing with fallback"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.feature_extractor = None
        self.method = None
        
    def fit(self, image_paths: List[str]) -> 'ImageProcessor':
        """Fit with fallback strategy"""
        # Try CLIP first
        try:
            import clip
            self.feature_extractor, self.preprocess = clip.load("ViT-B/32", device="cpu")
            self.method = 'clip'
            print("Using CLIP for image processing")
        except:
            # Fallback to basic processing
            self.method = 'basic'
            print("Using basic image processing")
        
        return self
    
    def transform(self, image_paths: List[str]) -> np.ndarray:
        """Transform images to features"""
        if self.method == 'clip':
            return self._transform_clip(image_paths)
        else:
            return self._transform_basic(image_paths)
    
    def _transform_clip(self, image_paths: List[str]) -> np.ndarray:
        """Transform using CLIP"""
        from PIL import Image
        features = []
        
        with torch.no_grad():
            for path in image_paths:
                try:
                    image = self.preprocess(Image.open(path)).unsqueeze(0)
                    feature = self.feature_extractor.encode_image(image)
                    features.append(feature.cpu().numpy().flatten())
                except:
                    # Dummy feature for failed images
                    features.append(np.zeros(512))
        
        return np.array(features)
    
    def _transform_basic(self, image_paths: List[str]) -> np.ndarray:
        """Basic image feature extraction"""
        features = []
        target_dim = 1024
        
        for path in image_paths:
            try:
                img = cv2.imread(str(path))
                if img is None:
                    features.append(np.zeros(target_dim))
                    continue
                    
                img = cv2.resize(img, self.target_size)
                img_features = img.flatten().astype(np.float32) / 255.0
                
                # Downsample to fixed size
                if len(img_features) > target_dim:
                    stride = len(img_features) // target_dim
                    img_features = img_features[::stride][:target_dim]
                else:
                    img_features = np.pad(img_features, (0, target_dim - len(img_features)))
                
                features.append(img_features)
            except:
                features.append(np.zeros(target_dim))
        
        return np.array(features)

class UniversalDataLoader:
    """Universal multimodal data loader with automatic detection"""
    
    def __init__(self, data_source, target_column: Optional[str] = None, 
                 sample_limit: Optional[int] = None, test_size: float = 0.2, 
                 val_size: float = 0.1, random_state: int = 42):
        
        # Load and detect
        print("Loading data...")
        self.raw_data = DataDetector.load_data(data_source)
        
        if sample_limit:
            self.raw_data = self.raw_data.head(sample_limit)
        
        print(f"Loaded {len(self.raw_data)} samples with {len(self.raw_data.columns)} columns")
        
        # Auto-detection
        self.target_column = DataDetector.detect_target_column(self.raw_data, target_column)
        self.modalities = DataDetector.detect_modalities(self.raw_data, self.target_column)
        self.task_type = DataDetector.detect_task_type(self.raw_data, self.target_column)
        
        print(f"Target column: {self.target_column}")
        print(f"Task type: {self.task_type}")
        print(f"Detected modalities: {self.modalities}")
        
        # Initialize processors
        self.processors = {}
        self.splits = {}
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
    def process_data(self) -> Dict[str, np.ndarray]:
        """Process all modalities"""
        processed_data = {}
        
        # Group columns by modality
        modality_groups = {}
        for column, modality in self.modalities.items():
            if modality not in modality_groups:
                modality_groups[modality] = []
            modality_groups[modality].append(column)
        
        # Process each modality
        for modality_type, columns in modality_groups.items():
            print(f"Processing {modality_type} columns: {columns}")
            
            if modality_type == 'text':
                processor = TextProcessor()
                text_data = self.raw_data[columns[0]].fillna('').astype(str).tolist()
                processor.fit(text_data)
                features = processor.transform(text_data)
                
            elif modality_type == 'tabular':
                processor = TabularProcessor()
                tabular_df = self.raw_data[columns]
                processor.fit(tabular_df)
                features = processor.transform(tabular_df)
                
            elif modality_type == 'image':
                processor = ImageProcessor()
                image_paths = self.raw_data[columns[0]].tolist()
                processor.fit(image_paths)
                features = processor.transform(image_paths)
            
            self.processors[modality_type] = processor
            processed_data[modality_type] = features
            print(f"  -> {features.shape} features")
        
        # Prepare labels
        labels = self._prepare_labels()
        processed_data['labels'] = labels
        
        return processed_data
    
    def _prepare_labels(self) -> np.ndarray:
        """Prepare target labels"""
        labels = self.raw_data[self.target_column]
        
        if self.task_type == 'classification':
            if labels.dtype == 'object':
                label_encoder = LabelEncoder()
                labels = label_encoder.fit_transform(labels.astype(str))
                print(f"Encoded {len(label_encoder.classes_)} classes")
        else:
            labels = pd.to_numeric(labels, errors='coerce').fillna(0)
            
        return labels.values if hasattr(labels, 'values') else labels
    
    def create_splits(self, processed_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Create train/val/test splits with robust stratification"""
        splits = {'train': {}, 'val': {}, 'test': {}}
        
        labels = processed_data['labels']
        
        # Check if stratification is possible for classification
        stratify = None
        if self.task_type == 'classification':
            unique_labels, counts = np.unique(labels, return_counts=True)
            min_count = np.min(counts)
            
            # Only use stratification if all classes have at least 2 samples
            if min_count >= 2:
                stratify = labels
            else:
                print(f"Warning: Some classes have only {min_count} sample(s). Skipping stratification.")
        
        # Create splits
        indices = np.arange(len(labels))
        train_val_idx, test_idx = train_test_split(
            indices, test_size=self.test_size, random_state=self.random_state, stratify=stratify
        )
        
        # Validation split
        if len(train_val_idx) > 1:
            train_labels = labels[train_val_idx]
            val_stratify = None
            
            if self.task_type == 'classification' and stratify is not None:
                unique_train_labels, train_counts = np.unique(train_labels, return_counts=True)
                if np.min(train_counts) >= 2:
                    val_stratify = train_labels
                    
            val_split_size = self.val_size / (1 - self.test_size)
            
            try:
                train_idx, val_idx = train_test_split(
                    train_val_idx, test_size=val_split_size, random_state=self.random_state, stratify=val_stratify
                )
            except ValueError:
                # Fallback without stratification
                train_idx, val_idx = train_test_split(
                    train_val_idx, test_size=val_split_size, random_state=self.random_state
                )
        else:
            train_idx = train_val_idx
            val_idx = np.array([])
        
        # Split all data
        for modality, features in processed_data.items():
            splits['train'][modality] = features[train_idx]
            if len(val_idx) > 0:
                splits['val'][modality] = features[val_idx]
            else:
                splits['val'][modality] = features[:0]  # Empty array with correct shape
            splits['test'][modality] = features[test_idx]
        
        print(f"Created splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        self.splits = splits
        return splits
    
    def get_pytorch_loaders(self, batch_size: int = 32) -> Dict[str, DataLoader]:
        """Create PyTorch DataLoaders"""
        if not self.splits:
            raise ValueError("Must create splits first")
        
        loaders = {}
        
        for split_name, split_data in self.splits.items():
            tensors = {}
            
            for modality in split_data:
                if modality != 'labels':
                    tensors[modality] = torch.FloatTensor(split_data[modality])
            
            if self.task_type == 'classification':
                tensors['labels'] = torch.LongTensor(split_data['labels'])
            else:
                tensors['labels'] = torch.FloatTensor(split_data['labels'])
            
            dataset = MultiModalDataset(tensors)
            shuffle = (split_name == 'train')
            loaders[split_name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return loaders
    
    def get_sklearn_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get data in sklearn format"""
        return self.splits
    
    def get_info(self) -> Dict[str, Any]:
        """Get loader information"""
        return {
            'task_type': self.task_type,
            'target_column': self.target_column,
            'modalities': self.modalities,
            'n_samples': len(self.raw_data),
            'n_features': sum(data.shape[1] for data in self.splits.get('train', {}).values() if isinstance(data, np.ndarray) and data.ndim == 2)
        }

class MultiModalDataset(Dataset):
    """PyTorch dataset for multimodal data"""
    
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self.tensors = tensors
        self.length = len(list(tensors.values())[0])
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {key: tensor[idx] for key, tensor in self.tensors.items()}
