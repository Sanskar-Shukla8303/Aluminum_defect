"""
Training script for YOLO defect detection model.

Usage:
    python train.py --model yolov8m-cls --epochs 50 --batch 32
    python train.py --model yolov8l-cls --epochs 100 --batch 16
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
import json
from datetime import datetime


class YOLODefectTrainer:
    def __init__(self, config_path):
        """Initialize trainer with config file."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.device = self._get_device()
        
    def _load_config(self):
        """Load configuration from yaml file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_device(self):
        """Get available device (GPU or CPU)."""
        if torch.cuda.is_available():
            device = 0  # Use first GPU
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
        else:
            device = 'cpu'
            print("⚠ GPU not available. Using CPU (training will be slow)")
        return device
    
    def train(self, model_type='yolov8m-cls', epochs=50, batch_size=32, 
              imgsz=640, patience=20, lr0=0.001, lrf=0.01):
        """
        Train YOLO model for defect classification.
        
        Args:
            model_type: YOLO model variant (e.g., 'yolov8n-cls', 'yolov8m-cls', 'yolov8l-cls')
            epochs: Number of training epochs
            batch_size: Batch size for training
            imgsz: Image size for training
            patience: Early stopping patience (epochs)
            lr0: Initial learning rate
            lrf: Final learning rate ratio
        
        Returns:
            results: Training results
        """
        
        print("\n" + "="*60)
        print("ALUMINUM DEFECT DETECTION - YOLO TRAINING")
        print("="*60)
        
        # Load model
        print(f"\n→ Loading model: {model_type}")
        model = YOLO(model_type)
        
        # Create results directory
        results_dir = Path(self.config['results_dir']) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset path
        data_path = r"D:\Aluminum_defect\aluminum_defect_yolo\data_cls"
        print(f"\n→ Training configuration:")
        print(f"  Model: {model_type}")
        print(f"  Dataset: {data_path}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {imgsz}")
        print(f"  Device: {self.device}")
        print(f"  Results directory: {results_dir}")
        
        # Train
        print("\n→ Starting training...")
        results = model.train(
            data=str(data_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=self.device,
            patience=patience,
            lr0=lr0,
            lrf=lrf,
            project=str(results_dir.parent),
            name=results_dir.name,
            save=True,
            save_period=5,  # Save checkpoint every 5 epochs
            plots=True,
            mosaic=1.0,
            augment=True,
            flipud=0.5,
            fliplr=0.5,
            degrees=10,
            translate=0.1,
            scale=0.5,
            optimizer='SGD',
            verbose=True,
            seed=42,
            deterministic=True
        )
        
        print("\n" + "="*60)
        print("✓ TRAINING COMPLETE")
        print("="*60)
        
        # Save training info
        self._save_training_info(results_dir, {
            'model': model_type,
            'epochs': epochs,
            'batch_size': batch_size,
            'imgsz': imgsz,
            'device': str(self.device)
        })
        
        return results
    
    def _save_training_info(self, results_dir, training_params):
        """Save training information to file."""
        info = {
            'timestamp': datetime.now().isoformat(),
            'training_parameters': training_params,
            'device_info': {
                'cuda_available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        info_path = results_dir / 'training_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
        
        print(f"\n→ Training info saved to: {info_path}")


def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for aluminum defect detection')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='yolov8m-cls',
                       choices=['yolov8n-cls', 'yolov8s-cls', 'yolov8m-cls', 
                               'yolov8l-cls', 'yolov8x-cls'],
                       help='YOLO model size (default: yolov8m-cls)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size (default: 640)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience in epochs (default: 20)')
    parser.add_argument('--lr0', type=float, default=0.001,
                       help='Initial learning rate (default: 0.001)')
    parser.add_argument('--lrf', type=float, default=0.01,
                       help='Final learning rate ratio (default: 0.01)')
    
    # Config
    parser.add_argument('--config', type=str,
                       default=r'D:\Aluminum_defect\aluminum_defect_yolo\config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = YOLODefectTrainer(args.config)
    
    # Train model
    trainer.train(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        lr0=args.lr0,
        lrf=args.lrf
    )


if __name__ == '__main__':
    main()
