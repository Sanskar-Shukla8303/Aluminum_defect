"""
Data Preparation Script for Aluminum Defect Detection YOLO Project

This script:
1. Organizes raw images from ali2018 dataset into train/val/test splits
2. Creates YOLO format labels (for classification or detection)
3. Generates dataset.yaml configuration file
"""

import os
import shutil
import yaml
import random
from pathlib import Path
from tqdm import tqdm
import argparse


class DataPreparer:
    def __init__(self, raw_data_path, output_path, train_split=0.7, val_split=0.15, test_split=0.15):
        """
        Initialize data preparation.
        
        Args:
            raw_data_path: Path to ali2018 folder with defect categories
            output_path: Path to output dataset directory
            train_split: Training data percentage
            val_split: Validation data percentage
            test_split: Testing data percentage
        """
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Create output directories
        self.setup_directories()
        
        # Class mapping - defect types
        self.classes = {
            'Clean sample': 0,
            'Be injured by a collision': 1,
            'Coating cracking': 2,
            'Convex powder': 3,
            'Dirty spot': 4,
            'Drain bottom': 5,
            'non-conducting': 6,
            'Orange peel': 7,
            'pitting': 8,
            'scuffing': 9,
            'The transverse strip is dented': 10
        }
        
    def setup_directories(self):
        """Create necessary output directories."""
        dirs = [
            self.output_path / 'images' / 'train',
            self.output_path / 'images' / 'val',
            self.output_path / 'images' / 'test',
            self.output_path / 'labels' / 'train',
            self.output_path / 'labels' / 'val',
            self.output_path / 'labels' / 'test',
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_class_id(self, class_name):
        """Get class ID from class name."""
        return self.classes.get(class_name, -1)
    
    def prepare_classification_data(self):
        """
        Prepare classification data: organize images by defect type into train/val/test.
        Each folder represents a class.
        """
        print("Preparing classification dataset...")
        
        class_stats = {}
        
        # Iterate through defect categories
        for class_name, class_id in self.classes.items():
            class_path = self.raw_data_path / class_name
            
            if not class_path.exists():
                print(f"Warning: {class_name} folder not found")
                continue
            
            # Get all images in this class
            images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
            
            if len(images) == 0:
                print(f"No images found in {class_name}")
                continue
            
            print(f"\n{class_name}: {len(images)} images")
            
            # Shuffle images
            random.shuffle(images)
            
            # Split into train/val/test
            n_train = int(len(images) * self.train_split)
            n_val = int(len(images) * self.val_split)
            
            train_imgs = images[:n_train]
            val_imgs = images[n_train:n_train + n_val]
            test_imgs = images[n_train + n_val:]
            
            # Copy images and create labels
            self._copy_and_label(train_imgs, 'train', class_id, class_name)
            self._copy_and_label(val_imgs, 'val', class_id, class_name)
            self._copy_and_label(test_imgs, 'test', class_id, class_name)
            
            class_stats[class_name] = {
                'total': len(images),
                'train': len(train_imgs),
                'val': len(val_imgs),
                'test': len(test_imgs)
            }
        
        return class_stats
    
    def _copy_and_label(self, image_paths, split, class_id, class_name):
        """Copy images and create YOLO format labels."""
        for img_path in tqdm(image_paths, desc=f"Processing {class_name} - {split}"):
            # Copy image
            dst_img = self.output_path / 'images' / split / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Create YOLO format label file (classification: just class ID)
            label_path = self.output_path / 'labels' / split / img_path.stem
            with open(f'{label_path}.txt', 'w') as f:
                f.write(str(class_id))
    
    def create_dataset_yaml(self):
        """Create dataset.yaml file for YOLO training."""
        dataset_config = {
            'path': str(self.output_path),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.classes),
            'names': {v: k for k, v in self.classes.items()}
        }
        
        yaml_path = self.output_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\nDataset configuration saved to {yaml_path}")
        return dataset_config
    
    def print_statistics(self, stats):
        """Print dataset statistics."""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        total_train = sum(s['train'] for s in stats.values())
        total_val = sum(s['val'] for s in stats.values())
        total_test = sum(s['test'] for s in stats.values())
        total_all = sum(s['total'] for s in stats.values())
        
        print(f"\nTotal images: {total_all}")
        print(f"Training set: {total_train} images ({100*total_train/total_all:.1f}%)")
        print(f"Validation set: {total_val} images ({100*total_val/total_all:.1f}%)")
        print(f"Test set: {total_test} images ({100*total_test/total_all:.1f}%)")
        
        print("\nPer-class breakdown:")
        print("-" * 60)
        for class_name, counts in stats.items():
            print(f"{class_name:30} | Total: {counts['total']:4} | Train: {counts['train']:4} | Val: {counts['val']:3} | Test: {counts['test']:3}")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Prepare aluminum defect dataset for YOLO')
    parser.add_argument('--raw_data', type=str, 
                       default=r'D:\Aluminum_defect\ali2018',
                       help='Path to raw ali2018 dataset')
    parser.add_argument('--output', type=str,
                       default=r'D:\Aluminum_defect\aluminum_defect_yolo\data',
                       help='Path to output dataset')
    parser.add_argument('--train_split', type=float, default=0.7,
                       help='Training data percentage (default: 0.7)')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Validation data percentage (default: 0.15)')
    
    args = parser.parse_args()
    
    # Calculate test split
    test_split = 1.0 - args.train_split - args.val_split
    
    # Initialize data preparer
    preparer = DataPreparer(
        args.raw_data,
        args.output,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=test_split
    )
    
    # Prepare data
    print("Starting data preparation...")
    stats = preparer.prepare_classification_data()
    
    # Create dataset configuration
    preparer.create_dataset_yaml()
    
    # Print statistics
    preparer.print_statistics(stats)
    
    print("\n✓ Data preparation complete!")
    print(f"Output directory: {preparer.output_path}")


if __name__ == '__main__':
    main()
