import shutil
from pathlib import Path
from tqdm import tqdm

def reformat_for_classification():
    source_dir = Path(r"D:\Aluminum_defect\aluminum_defect_yolo\data")
    dest_dir = Path(r"D:\Aluminum_defect\aluminum_defect_yolo\data_cls")
    
    # Class mapping from your original script
    classes = {
        0: 'Clean sample', 1: 'Be injured by a collision', 2: 'Coating cracking',
        3: 'Convex powder', 4: 'Dirty spot', 5: 'Drain bottom', 6: 'non-conducting',
        7: 'Orange peel', 8: 'pitting', 9: 'scuffing', 10: 'The transverse strip is dented'
    }
    
    for split in ['train', 'val', 'test']:
        print(f"\nConverting {split} data...")
        img_split_dir = source_dir / 'images' / split
        lbl_split_dir = source_dir / 'labels' / split
        
        if not img_split_dir.exists():
            continue
            
        images = list(img_split_dir.glob('*.*'))
        
        for img_path in tqdm(images):
            # Find the corresponding text label
            lbl_path = lbl_split_dir / (img_path.stem + '.txt')
            if not lbl_path.exists():
                continue
                
            # Read class ID from text file
            with open(lbl_path, 'r') as f:
                class_id = int(f.read().strip())
                
            class_name = classes.get(class_id, 'Unknown')
            
            # Create target class folder
            target_dir = dest_dir / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy image to its new class folder
            shutil.copy2(img_path, target_dir / img_path.name)
            
    print(f"\n✓ Dataset successfully reformatted to: {dest_dir}")

if __name__ == '__main__':
    reformat_for_classification()