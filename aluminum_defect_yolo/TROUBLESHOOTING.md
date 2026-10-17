# TROUBLESHOOTING GUIDE

Common issues and their solutions.

## 🔧 Installation Issues

### Issue: "python: command not found"
**Cause:** Python not installed or not in PATH
**Solution:**
```bash
# Download Python 3.10+ from python.org
# During installation, CHECK: "Add Python to PATH"
# Verify:
python --version
```

### Issue: "ModuleNotFoundError: No module named 'ultralytics'"
**Cause:** Dependencies not installed
**Solution:**
```bash
# Make sure virtual environment is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "from ultralytics import YOLO; print('✓')"
```

### Issue: "CUDA not found" but I have NVIDIA GPU
**Cause:** CUDA-enabled PyTorch not installed
**Solution:**
```bash
# Remove current PyTorch
pip uninstall torch torchvision torchaudio -y

# Install CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

---

## 📊 Data Preparation Issues

### Issue: "No images found in [folder]"
**Cause:** Folder path is incorrect or images not in specified location
**Solution:**
```bash
# Check path exists
dir C:\Users\akash\Desktop\Aluminum_defect\ali2018

# Run with correct path
python scripts/prepare_data.py --raw_data "C:\Users\akash\Desktop\Aluminum_defect\ali2018"
```

### Issue: "FileNotFoundError: dataset.yaml"
**Cause:** Data preparation didn't complete successfully
**Solution:**
```bash
# Re-run data preparation
python scripts/prepare_data.py

# Verify output
dir data/  # Should contain images/ and labels/ folders
```

### Issue: Dataset split is wrong (not 70/15/15)
**Cause:** Custom split values didn't match requirements
**Solution:**
```bash
# Use default split (70/15/15)
python scripts/prepare_data.py

# Or specify exact split
python scripts/prepare_data.py --train_split 0.7 --val_split 0.15
```

---

## 🎯 Training Issues

### Issue: "RuntimeError: CUDA out of memory"
**Cause:** GPU doesn't have enough VRAM for batch size
**Solution:**
```bash
# Option 1: Reduce batch size
python scripts/train.py --batch 16  # Default is 32

# Option 2: Use smaller model
python scripts/train.py --model yolov8s-cls --batch 32

# Option 3: Reduce image size
python scripts/train.py --imgsz 416  # Default is 640

# Option 4: Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

**Quick reference by GPU:**
```
GPU Memory Recommendation:
- 2GB: --model yolov8n-cls --batch 8 --imgsz 416
- 4GB: --model yolov8s-cls --batch 16 --imgsz 640
- 6GB: --model yolov8m-cls --batch 32 --imgsz 640
- 8GB+: --model yolov8l-cls --batch 64 --imgsz 640
```

### Issue: Training is very slow
**Cause:** 
1. Using CPU instead of GPU
2. Batch size too small
3. Image size too large

**Solutions:**
```bash
# Check if GPU is being used
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"

# If GPU available but not used:
# - Check CUDA installation (see above)
# - Try forcing GPU with CUDA_VISIBLE_DEVICES=0

# If using CPU, use smaller model/batch/image size
python scripts/train.py --model yolov8n-cls --batch 16 --imgsz 416 --epochs 20
```

### Issue: Validation loss not decreasing
**Cause:** Learning rate too high, model not learning
**Solution:**
```bash
# Reduce learning rate
python scripts/train.py --lr0 0.0005  # Default is 0.001

# Or increase patience and epochs
python scripts/train.py --epochs 100 --patience 30

# Check if data is properly prepared
python -c "
from pathlib import Path
import yaml
with open('data/dataset.yaml') as f:
    cfg = yaml.safe_load(f)
    print(f'Classes: {cfg[\"nc\"]}')
    print(f'Train images: {len(list(Path(cfg[\"path\"])/cfg[\"train\"]).glob(\"**/*.jpg\"))}')
"
```

### Issue: "RuntimeError: Expected 3D (unbatched) or 4D (batched) input"
**Cause:** Image format issue
**Solution:**
```bash
# Verify image format
python scripts/inference.py --model best.pt --source image.jpg

# Make sure image is valid
# Try converting to standard format
python -c "
import cv2
img = cv2.imread('image.jpg')
if img is None:
    print('Image corrupted!')
else:
    print('Image OK')
"
```

---

## 🔍 Inference Issues

### Issue: "FileNotFoundError: best.pt not found"
**Cause:** Training didn't complete or model not in expected location
**Solution:**
```bash
# Find trained model
dir /s runs\detect\*\weights\best.pt

# Use full path
python scripts/inference.py --model "C:\full\path\to\best.pt" --source image.jpg
```

### Issue: Model predicts all images as same class
**Cause:** 
1. Model underfitted (trained for too few epochs)
2. Dataset imbalanced
3. Poor quality training data

**Solutions:**
```bash
# Re-train with more epochs
python scripts/train.py --model yolov8m-cls --epochs 100

# Check dataset balance
python scripts/prepare_data.py  # Will show class distribution

# Check prediction confidence
python scripts/inference.py --model best.pt --source image.jpg --conf 0.3
```

### Issue: Prediction confidence very low
**Cause:** Model uncertainty - not trained well on this image type
**Solution:**
```bash
# Lower confidence threshold to see predictions
python scripts/inference.py --model best.pt --source image.jpg --conf 0.3

# Or re-train with better data
python scripts/train.py --epochs 100 --imgsz 768
```

### Issue: "CUDA out of memory" during inference
**Cause:** Batch size too large for inference
**Solution:**
```bash
# YOLOv8 automatically batches - reduce image size
python scripts/inference.py --model best.pt --source folder/ --imgsz 416
```

---

## 📁 File & Path Issues

### Issue: "FileNotFoundError" or path issues
**Cause:** Backslash interpretation in paths
**Solution:**
```bash
# Use forward slashes
python scripts/train.py --data "data/dataset.yaml"

# Or use raw string
python scripts/train.py --data "C:\\full\\path\\to\\dataset.yaml"

# Best practice - use in Python
from pathlib import Path
path = Path('data') / 'dataset.yaml'
```

### Issue: "Permission denied" errors
**Cause:** File/folder permissions
**Solution:**
```bash
# Run command prompt as Administrator (right-click → Run as administrator)
# Then try again

# Or change directory permissions
icacls "c:\Users\akash\Desktop\aluminum_defect_yolo" /grant:r "%username%":F /t
```

---

## 🆘 Advanced Debugging

### Enable Verbose Logging
```bash
# Add verbose flag to see detailed output
python scripts/train.py --model yolov8m-cls --epochs 10 --verbose
```

### Check Configuration
```bash
# Verify config file
python -c "
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
    for key, value in config.items():
        print(f'{key}: {value}')
"
```

### Verify Dataset
```bash
# Check YOLO dataset config
python -c "
import yaml
with open('data/dataset.yaml') as f:
    cfg = yaml.safe_load(f)
    print('Dataset Config:')
    for key, value in cfg.items():
        print(f'  {key}: {value}')
"
```

### Test Model on Sample
```bash
# Simple test
python -c "
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')
results = model.predict('test_image.jpg', verbose=True)
for r in results:
    print(f'Top prediction: {r.top1} with confidence {r.probs.top1conf}')
"
```

---

## 🔄 Recovery Procedures

### Reset Everything
```bash
# Remove all outputs
rmdir /s /q runs/
rmdir /s /q data/images
rmdir /s /q data/labels
del data/dataset.yaml

# Start fresh
python scripts/prepare_data.py
python scripts/train.py --epochs 30
```

### Backup Important Files
```bash
# Copy best model
copy "runs\detect\train\weights\best.pt" "best_backup.pt"

# Copy results
copy "runs\detect\train\results.csv" "results_backup.csv"
```

### Restore from Backup
```bash
# Copy backup back
copy "best_backup.pt" "best.pt"

# Use backup model
python scripts/inference.py --model "best_backup.pt" --source image.jpg
```

---

## 📞 Getting Help

### Check These First
1. **QUICKSTART.md** - Fast solutions
2. **README.md** - Detailed explanations
3. This file - Specific issues
4. Script help: `python scripts/train.py --help`

### Collect Information for Support
```bash
# System info
python --version
pip list | grep -E "torch|ultralytics|opencv"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Error details
python scripts/train.py --model yolov8m-cls --epochs 5 2>&1 > error.log
# Share error.log with support
```

### Useful Commands
```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Check all dependencies
pip list

# Update packages
pip install --upgrade ultralytics torch torchvision

# Clear cache
python -c "import torch; torch.cuda.empty_cache()"

# Reinstall from scratch
pip install -r requirements.txt --force-reinstall --no-cache-dir
```

---

## ✅ Pre-Training Checklist

Before running training, verify:
- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Data prepared: `data/dataset.yaml` exists
- [ ] GPU detected (optional): `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Disk space available: > 50GB
- [ ] No other heavy processes running (GPU training)

---

**Still stuck?** Check the main README.md file or visit:
- https://docs.ultralytics.com/help/
- https://github.com/ultralytics/ultralytics/issues
