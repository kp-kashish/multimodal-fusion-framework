import os

def create_framework():
    """Create multimodal fusion framework structure"""
    
    # Define directories
    dirs = [
        "multimodal_fusion/src/models",
        "multimodal_fusion/src/data", 
        "multimodal_fusion/src/utils",
        "multimodal_fusion/src/experiments"
    ]
    
    # Define files
    files = [
        "multimodal_fusion/requirements.txt",
        "multimodal_fusion/__init__.py",
        "multimodal_fusion/src/__init__.py",
        "multimodal_fusion/src/models/__init__.py",
        "multimodal_fusion/src/models/base_fusion.py",
        "multimodal_fusion/src/models/intermediate_fusion.py",
        "multimodal_fusion/src/models/hybrid_late_fusion.py",
        "multimodal_fusion/src/models/attention_fusion.py",
        "multimodal_fusion/src/data/__init__.py",
        "multimodal_fusion/src/data/base_loader.py",
        "multimodal_fusion/src/data/brain_tumor_loader.py",
        "multimodal_fusion/src/data/job_salary_loader.py",
        "multimodal_fusion/src/utils/__init__.py",
        "multimodal_fusion/src/utils/early_stopping.py",
        "multimodal_fusion/src/utils/evaluation.py",
        "multimodal_fusion/src/utils/dashboard.py",
        "multimodal_fusion/src/experiments/__init__.py",
        "multimodal_fusion/src/experiments/run_brain_tumor.py",
        "multimodal_fusion/src/experiments/run_job_salary.py"
    ]
    
    # Create directories
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Create files
    for file_path in files:
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                if file_path.endswith('requirements.txt'):
                    f.write("""torch>=1.11.0
torchvision>=0.12.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
tensorboard>=2.8.0
transformers>=4.15.0
tqdm>=4.62.0
psutil>=5.8.0
""")
                elif file_path.endswith('__init__.py'):
                    f.write("")
                else:
                    f.write("# " + os.path.basename(file_path) + "\n")
    
    print("Framework created successfully")
    print("Structure:")
    for root, dirs, files in os.walk("."):
        if root == ".":
            continue
        level = root.count(os.sep) - 1
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = "  " * (level + 1)
        for file in files:
            print(f"{subindent}{file}")

if __name__ == "__main__":
    create_framework()