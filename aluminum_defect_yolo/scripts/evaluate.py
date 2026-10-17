"""
Evaluation script for testing model performance on test dataset.

Usage:
    python evaluate.py --model path/to/best.pt --data ./data/dataset.yaml
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import json
from datetime import datetime
import cv2


class ModelEvaluator:
    def __init__(self, model_path, dataset_yaml):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            dataset_yaml: Path to dataset.yaml configuration
        """
        self.model_path = Path(model_path)
        self.dataset_yaml = Path(dataset_yaml)
        self.model = YOLO(str(self.model_path))
        
        print(f"✓ Model loaded: {self.model_path}")
        print(f"✓ Dataset config: {self.dataset_yaml}")
    
    def evaluate(self, conf=0.5, imgsz=640):
        """
        Evaluate model on test set.
        
        Args:
            conf: Confidence threshold
            imgsz: Image size
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Validate on test set
        results = self.model.val(
            data=str(self.dataset_yaml),
            conf=conf,
            imgsz=imgsz,
            verbose=True
        )
        
        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)
        
        # Print detailed results
        if hasattr(results, 'top1'):
            print(f"Top-1 Accuracy: {results.top1:.2%}")
        if hasattr(results, 'top5'):
            print(f"Top-5 Accuracy: {results.top5:.2%}")
        
        return results
    
    def test_on_directory(self, test_dir, conf=0.5, imgsz=640):
        """
        Test model on directory of images.
        
        Args:
            test_dir: Path to test images
            conf: Confidence threshold
            imgsz: Image size
        """
        test_dir = Path(test_dir)
        
        print(f"\n→ Testing on directory: {test_dir}")
        
        y_true = []
        y_pred = []
        confidences = []
        
        image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
        
        print(f"Found {len(image_files)} images")
        
        for img_path in image_files:
            # Get true label from parent folder name
            true_class = img_path.parent.name
            
            # Predict
            result = self.model.predict(
                source=str(img_path),
                conf=conf,
                imgsz=imgsz,
                verbose=False
            )[0]
            
            pred_class_id = int(result.top1)
            confidence = result.probs.top1conf.item()
            
            y_true.append(true_class)
            y_pred.append(pred_class_id)
            confidences.append(confidence)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred)
        
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        self._print_metrics(metrics)
        
        return metrics, y_true, y_pred
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        return metrics
    
    def _print_metrics(self, metrics):
        """Print evaluation metrics."""
        print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print("="*60)
    
    def save_evaluation_report(self, metrics, output_path='evaluation_report.json'):
        """Save evaluation report to file."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'dataset': str(self.dataset_yaml),
            'metrics': metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"\n✓ Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO defect detection model')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset.yaml')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size (default: 640)')
    parser.add_argument('--test_dir', type=str,
                       help='Optional: Path to test directory')
    parser.add_argument('--output', type=str, default='evaluation_report.json',
                       help='Output report path')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model, args.data)
    
    # Run evaluation
    if args.test_dir:
        metrics, y_true, y_pred = evaluator.test_on_directory(
            args.test_dir,
            conf=args.conf,
            imgsz=args.imgsz
        )
        evaluator.save_evaluation_report(metrics, args.output)
    else:
        results = evaluator.evaluate(conf=args.conf, imgsz=args.imgsz)
        
        # Save results
        report = {
            'timestamp': datetime.now().isoformat(),
            'model': str(args.model),
            'dataset': str(args.data)
        }
        
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"\n✓ Report saved: {args.output}")


if __name__ == '__main__':
    main()
