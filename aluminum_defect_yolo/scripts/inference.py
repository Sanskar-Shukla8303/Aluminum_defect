"""
Inference script for running predictions on new aluminum sheets.

Usage:
    python inference.py --model path/to/model.pt --source path/to/image.jpg
    python inference.py --model model.pt --source path/to/folder
    python inference.py --model model.pt --source video.mp4
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import json


class DefectInference:
    def __init__(self, model_path):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained YOLO model
        """
        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))
        
        # Class names
        self.class_names = {
            0: 'Clean sample',
            1: 'Be injured by a collision',
            2: 'Coating cracking',
            3: 'Convex powder',
            4: 'Dirty spot',
            5: 'Drain bottom',
            6: 'Non-conducting',
            7: 'Orange peel',
            8: 'Pitting',
            9: 'Scuffing',
            10: 'Transverse strip dented'
        }
        
        print(f"✓ Model loaded: {self.model_path}")
    
    def predict(self, source, conf=0.5, imgsz=640):
        """
        Run prediction on image/video/folder.
        
        Args:
            source: Path to image, video, or folder
            conf: Confidence threshold
            imgsz: Image size
            
        Returns:
            results: List of prediction results
        """
        source = Path(source)
        
        if source.is_dir():
            return self._predict_folder(source, conf, imgsz)
        elif source.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            return self._predict_image(source, conf, imgsz)
        elif source.suffix.lower() in ['.mp4', '.avi', '.mov']:
            return self._predict_video(source, conf, imgsz)
        else:
            raise ValueError(f"Unsupported file type: {source.suffix}")
    
    def _predict_image(self, image_path, conf, imgsz):
        """Predict on single image."""
        print(f"\n→ Processing image: {image_path}")
        
        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            imgsz=imgsz,
            verbose=False
        )
        
        # Process results
        for result in results:
            self._print_result(image_path.name, result)
            self._save_annotated_image(image_path, result)
        
        return results
    
    def _predict_folder(self, folder_path, conf, imgsz):
        """Predict on all images in folder and subfolders."""
        folder_path = Path(folder_path)
        print(f"\n→ Processing folder: {folder_path}")
        
        # Recursive search (rglob) across all common image extensions
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')
        image_files = []
        for ext in extensions:
            image_files.extend(folder_path.rglob(ext))
            
        print(f"  Found {len(image_files)} images")
        
        if len(image_files) == 0:
            print("  ⚠ No images found in directory or subdirectories.")
            return []
        
        results = []
        for img_path in tqdm(image_files, desc="Processing images"):
            result = self.model.predict(
                source=str(img_path),
                conf=conf,
                imgsz=imgsz,
                verbose=False
            )
            results.extend(result)
            self._print_result(img_path.name, result[0])
        
        return results
    
    def _predict_video(self, video_path, conf, imgsz):
        """Predict on video file."""
        print(f"\n→ Processing video: {video_path}")
        
        results = self.model.predict(
            source=str(video_path),
            conf=conf,
            imgsz=imgsz,
            verbose=False,
            save=True,
            save_txt=True
        )
        
        print(f"✓ Video processed. Results saved.")
        return results
    
    def _print_result(self, filename, result):
        """Print prediction result."""
        top1_index = int(result.probs.top1)
        
        # Get class name directly from model metadata or fallback dictionary
        if hasattr(result, 'names') and top1_index in result.names:
            class_name = result.names[top1_index]
        else:
            class_name = self.class_names.get(top1_index, f"Unknown (ID: {top1_index})")
            
        confidence = result.probs.top1conf.item()
        
        print(f"  {filename}: {class_name} ({confidence:.2%})")
    
    def _save_annotated_image(self, image_path, result):
        """Save image with predictions annotated."""
        img = cv2.imread(str(image_path))
        
        # Get prediction
        top1_index = int(result.probs.top1)
        confidence = result.probs.top1conf.item()
        
        if hasattr(result, 'names') and top1_index in result.names:
            class_name = result.names[top1_index]
        else:
            class_name = self.class_names.get(top1_index, "Unknown")
        
        # Add text to image
        text = f"{class_name}: {confidence:.1%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (10, 30), font, 1, (0, 255, 0), 2)
        
        # Save
        output_path = image_path.parent / f"predicted_{image_path.name}"
        cv2.imwrite(str(output_path), img)
        print(f"  Annotated image saved: {output_path}")
        
        return output_path
    
    def evaluate_folder(self, folder_path, conf=0.5, imgsz=640):
        """
        Evaluate model on folder of images and generate statistics.
        """
        folder_path = Path(folder_path)
        print(f"\n→ Evaluating on folder: {folder_path}")
        
        results = self._predict_folder(folder_path, conf, imgsz)
        
        if not results:
            print("No images were evaluated.")
            return {}
        
        # Collect statistics
        stats = {
            'total_images': len(results),
            'class_predictions': {},
            'confidences': []
        }
        
        for result in results:
            class_id = int(result.probs.top1)
            if hasattr(result, 'names') and class_id in result.names:
                class_name = result.names[class_id]
            else:
                class_name = self.class_names.get(class_id, "Unknown")
                
            confidence = result.probs.top1conf.item()
            
            if class_name not in stats['class_predictions']:
                stats['class_predictions'][class_name] = {'count': 0, 'avg_conf': []}
            
            stats['class_predictions'][class_name]['count'] += 1
            stats['class_predictions'][class_name]['avg_conf'].append(confidence)
            stats['confidences'].append(confidence)
        
        # Calculate averages
        for class_name in stats['class_predictions']:
            confs = stats['class_predictions'][class_name]['avg_conf']
            stats['class_predictions'][class_name]['avg_confidence'] = sum(confs) / len(confs)
            del stats['class_predictions'][class_name]['avg_conf']
        
        # Overall statistics
        stats['avg_confidence'] = sum(stats['confidences']) / len(stats['confidences'])
        stats['min_confidence'] = min(stats['confidences'])
        stats['max_confidence'] = max(stats['confidences'])
        
        # Print statistics
        self._print_evaluation_stats(stats)
        
        return stats
    
    def _print_evaluation_stats(self, stats):
        """Print evaluation statistics."""
        print("\n" + "="*60)
        print("EVALUATION STATISTICS")
        print("="*60)
        
        print(f"\nTotal images: {stats['total_images']}")
        print(f"Average confidence: {stats['avg_confidence']:.2%}")
        print(f"Min confidence: {stats['min_confidence']:.2%}")
        print(f"Max confidence: {stats['max_confidence']:.2%}")
        
        print("\nPredictions by class:")
        print("-" * 60)
        for class_name, counts in stats['class_predictions'].items():
            pct = 100 * counts['count'] / stats['total_images']
            print(f"{class_name:30} | Count: {counts['count']:4} ({pct:5.1f}%) | Avg Conf: {counts['avg_confidence']:.2%}")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Run inference with YOLO defect detection model')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image, video, or folder')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size (default: 640)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation and print statistics')
    
    args = parser.parse_args()
    
    # Initialize inference
    detector = DefectInference(args.model)
    
    # Run predictions
    if args.evaluate and Path(args.source).is_dir():
        stats = detector.evaluate_folder(args.source, args.conf, args.imgsz)
    else:
        results = detector.predict(args.source, args.conf, args.imgsz)
        print(f"\n✓ Inference complete. Processed {len(results)} result(s)")


if __name__ == '__main__':
    main()