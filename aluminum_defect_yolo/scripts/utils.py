"""
Utility functions for the aluminum defect detection project.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


class ImageAnalyzer:
    """Analyze image dataset properties."""
    
    @staticmethod
    def analyze_dataset(data_path):
        """
        Analyze dataset statistics.
        
        Args:
            data_path: Path to dataset folder with images
        """
        data_path = Path(data_path)
        
        images = list(data_path.glob('**/*.jpg')) + list(data_path.glob('**/*.png'))
        
        if not images:
            print("No images found!")
            return
        
        print("\n" + "="*60)
        print("DATASET ANALYSIS")
        print("="*60)
        
        # Basic stats
        print(f"\nTotal images: {len(images)}")
        
        # Image dimensions
        heights = []
        widths = []
        file_sizes = []
        
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                heights.append(h)
                widths.append(w)
                file_sizes.append(os.path.getsize(img_path) / 1024)  # KB
        
        print(f"\nImage Dimensions:")
        print(f"  Width range: {min(widths)} - {max(widths)} pixels")
        print(f"  Height range: {min(heights)} - {max(heights)} pixels")
        print(f"  Most common size: {max(Counter([(w,h) for w,h in zip(widths, heights)]).items(), key=lambda x: x[1])[0]}")
        
        print(f"\nFile Sizes:")
        print(f"  Average: {np.mean(file_sizes):.1f} KB")
        print(f"  Total: {sum(file_sizes)/1024:.1f} MB")
        print(f"  Range: {min(file_sizes):.1f} - {max(file_sizes):.1f} KB")
        
        print("\n" + "="*60)
    
    @staticmethod
    def visualize_dataset_distribution(dataset_yaml):
        """Visualize dataset class distribution."""
        import yaml
        
        with open(dataset_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        names = config.get('names', {})
        
        # Count images per class
        class_counts = {}
        data_dir = Path(config['path'])
        train_dir = data_dir / config['train']
        
        for class_id, class_name in names.items():
            # Count image files
            img_files = list((train_dir / class_name).glob('*')) if (train_dir / class_name).exists() else []
            class_counts[class_name] = len(img_files)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(class_counts)), list(class_counts.values()))
        plt.xticks(range(len(class_counts)), list(class_counts.keys()), rotation=45, ha='right')
        plt.ylabel('Number of Images')
        plt.title('Dataset Class Distribution')
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=100)
        print("✓ Saved: class_distribution.png")
        plt.close()


class ModelUtils:
    """Utility functions for model management."""
    
    @staticmethod
    def get_model_info(model_path):
        """Get information about a trained model."""
        from ultralytics import YOLO
        
        model = YOLO(str(model_path))
        
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        print(f"Model path: {model_path}")
        print(f"Model type: {model.model}")
        
        # Get model summary
        print("\n" + "-"*60)
        print("Model Summary:")
        print("-"*60)
        # model.info()
        
        print("="*60)
    
    @staticmethod
    def compare_models(model_paths):
        """Compare multiple trained models."""
        from ultralytics import YOLO
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        for model_path in model_paths:
            model = YOLO(str(model_path))
            print(f"\nModel: {Path(model_path).name}")
            # Add comparison metrics here
        
        print("="*60)


class DataAugmentation:
    """Data augmentation utilities."""
    
    @staticmethod
    def augment_image(image_path, augmentation_type='random', num_variants=3):
        """
        Create augmented versions of an image.
        
        Args:
            image_path: Path to image
            augmentation_type: Type of augmentation
            num_variants: Number of augmented versions
        """
        img = cv2.imread(str(image_path))
        
        augmented = []
        
        for i in range(num_variants):
            if augmentation_type in ['random', 'rotation']:
                angle = np.random.randint(-15, 15)
                h, w = img.shape[:2]
                matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                rotated = cv2.warpAffine(img, matrix, (w, h))
                augmented.append(rotated)
            
            if augmentation_type in ['random', 'brightness']:
                brightness = np.random.uniform(0.7, 1.3)
                brightened = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
                augmented.append(brightened)
            
            if augmentation_type in ['random', 'blur']:
                kernel_size = np.random.choice([3, 5, 7])
                blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
                augmented.append(blurred)
        
        return augmented


class ResultsAnalyzer:
    """Analyze and visualize training results."""
    
    @staticmethod
    def plot_training_metrics(csv_path):
        """
        Plot training metrics from results.csv.
        
        Args:
            csv_path: Path to results.csv from training
        """
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot training/validation loss
        axes[0, 0].plot(df['epoch'], df['train/loss'], label='Train Loss')
        axes[0, 0].plot(df['epoch'], df['val/loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss over Epochs')
        axes[0, 0].legend()
        axes[0, 0].grid()
        
        # Plot accuracy
        if 'metrics/accuracy_top1' in df.columns:
            axes[0, 1].plot(df['epoch'], df['metrics/accuracy_top1'], label='Top-1 Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Model Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid()
        
        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=100)
        print("✓ Saved: training_metrics.png")
        plt.close()
    
    @staticmethod
    def confusion_matrix_from_predictions(y_true, y_pred):
        """
        Create confusion matrix from predictions.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=100)
        print("✓ Saved: confusion_matrix.png")
        plt.close()
        
        return cm


def setup_project():
    """Initialize project structure."""
    dirs = [
        'data/raw',
        'data/images',
        'data/labels',
        'models/trained',
        'models/configs',
        'results',
        'notebooks',
        'scripts'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✓ Project structure initialized")


if __name__ == '__main__':
    setup_project()
