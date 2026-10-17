



# QUICK START GUIDE - Aluminum Defect Detection YOLO

Get your defect detection model running in **5 minutes**!

## Prerequisites

- Python 3.8+ installed
- ~30GB disk space available
- (Optional) NVIDIA GPU with CUDA 11.8+

## Installation (2 minutes)

```bash
# 1. Navigate to project directory
cd aluminum_defect_yolo

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "from ultralytics import YOLO; print('✓ YOLO ready')"
```

---

## Step-by-Step Training

### 1. Prepare Data (1-2 minutes)

```bash
python scripts/prepare_data.py
```

**What it does:**
- ✓ Organizes 4000+ images from ali2018 dataset
- ✓ Splits into train/val/test (70/15/15)
- ✓ Creates YOLO format labels
- ✓ Generates dataset.yaml configuration

**Output:**
```
data/
├── images/
│   ├── train/ (2800 images)
│   ├── val/ (600 images)
│   └── test/ (600 images)
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml ✓
```

### 2. Train Model (20-60 minutes, depending on GPU)

```bash
# Quick training (recommended for first run)
python scripts/train.py --model yolov8m-cls --epochs 30 --batch 32

# Better accuracy (takes longer)
python scripts/train.py --model yolov8m-cls --epochs 50 --batch 32

# Production quality (for final model)
python scripts/train.py --model yolov8l-cls --epochs 100 --batch 16
```

**What happens:**
- ✓ Downloads YOLOv8 weights (first time only: 100MB)
- ✓ Trains on your dataset
- ✓ Saves best model to `runs/detect/train/weights/best.pt`
- ✓ Shows training progress and loss plots

**Expected output after training:**
```
→ Training complete!
✓ Best model: runs/detect/train/weights/best.pt
✓ Accuracy: ~85-90% (depends on training size)
✓ Training time: 30-60 minutes
```

### 3. Test on Single Image (30 seconds)

```bash
python scripts/inference.py \
    --model runs/detect/train/weights/best.pt \
    --source "path/to/aluminum_sheet.jpg"
```

**Output:**
```
→ Processing image: aluminum_sheet.jpg
  aluminum_sheet.jpg: Orange peel (92%)
  
✓ Annotated image saved: predicted_aluminum_sheet.jpg
```

### 4. Batch Test on Folder (1-2 minutes)

```bash
python scripts/inference.py \
    --model runs/detect/train/weights/best.pt \
    --source data/images/test/ \
    --evaluate
```

**Output:**
```
EVALUATION STATISTICS
====================================================
Total images: 600
Average confidence: 87.5%

Predictions by class:
Clean sample          | Count: 65 (10.8%) | Avg Conf: 94%
Coating cracking      | Count: 48 (8.0%)  | Avg Conf: 91%
Orange peel           | Count: 52 (8.7%)  | Avg Conf: 88%
...
====================================================
```

---

## Model Sizes & Speed

Choose based on your needs:

```bash
# 1. Fast (Real-time inference)
python scripts/train.py --model yolov8n-cls --epochs 20

# 2. Balanced (Recommended)
python scripts/train.py --model yolov8m-cls --epochs 50

# 3. Accurate (Best results)
python scripts/train.py --model yolov8l-cls --epochs 100
```

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| nano | 3MB | 2ms | 75% | Mobile/Edge |
| small | 11MB | 5ms | 82% | Real-time |
| **medium** | **26MB** | **10ms** | **87%** | **Balanced** |
| large | 44MB | 15ms | 89% | High accuracy |
| xlarge | 71MB | 20ms | 91% | Maximum accuracy |

---

## Common Commands Reference

```bash
# 📊 Data preparation
python scripts/prepare_data.py

# 🎯 Training
python scripts/train.py --model yolov8m-cls --epochs 50

# 🔍 Single image prediction
python scripts/inference.py --model best.pt --source image.jpg

# 📁 Batch prediction with stats
python scripts/inference.py --model best.pt --source folder/ --evaluate

# 📈 Evaluate on test set
python scripts/evaluate.py --model best.pt --data data/dataset.yaml

# 🎬 Video analysis
python scripts/inference.py --model best.pt --source video.mp4
```

---

## Troubleshooting

### Problem: "CUDA not found" but I have a GPU

```bash
# Install CUDA-enabled PyTorch
pip uninstall torch torchvision -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Problem: "Out of Memory" error during training

```bash
# Reduce batch size
python scripts/train.py --model yolov8m-cls --batch 16

# Or use smaller model
python scripts/train.py --model yolov8s-cls --batch 32
```

### Problem: Training is too slow

```bash
# Reduce epochs and use smaller model
python scripts/train.py --model yolov8n-cls --epochs 20 --batch 64

# Check if GPU is being used
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Problem: Poor accuracy on predictions

- ✓ Train for more epochs: `--epochs 100`
- ✓ Use larger model: `yolov8l-cls`
- ✓ Check if test images are similar to training data

---

## Performance Expectations

On a typical setup:

| Stage | Time | Machine |
|-------|------|---------|
| Data prep | 2-5 min | Any |
| Training (30 epochs) | 30 min | RTX 3070 |
| Training (50 epochs) | 50 min | RTX 3070 |
| Single prediction | 50ms | GPU / 500ms CPU |
| Batch 100 images | 5-10s | GPU / 50s CPU |

**Accuracy expectations:**
- First training: 75-80%
- After 50 epochs: 85-90%
- With data augmentation: 90-95%

---

## Next Steps After Training

### 1. Deploy Your Model

```python
# Use model in your application
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
results = model.predict(source='image.jpg', conf=0.5)
```

### 2. Export to Different Formats

```bash
# Export to ONNX (for cross-platform inference)
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='onnx')"

# Export to TensorFlow Lite (for mobile/edge)
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='tflite')"
```

### 3. Fine-tune on New Data

```bash
# If you get new defective samples, fine-tune the model
python scripts/train.py --model yolov8m-cls --epochs 20 --lr0 0.0001
```

---

## File Locations

After training, your files are here:

```
runs/detect/train/
├── weights/
│   ├── best.pt ← Use this for predictions
│   └── last.pt
├── results.csv ← Training metrics
├── results.png ← Loss/accuracy graphs
└── confusion_matrix.png ← Per-class performance
```

---

## Support

**Stuck?** Check these:

1. **README.md** - Full documentation
2. **Logs** - Check `runs/detect/train/` for detailed logs
3. **UltraLytics Docs** - https://docs.ultralytics.com/
4. **Common Issues** - See README.md Troubleshooting section

---

## Success Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Data prepared (`data/dataset.yaml` exists)
- [ ] Model trained (`runs/detect/train/weights/best.pt` exists)
- [ ] Single image prediction works
- [ ] Batch prediction works
- [ ] Accuracy meets expectations

---

**You're ready to detect aluminum defects! 🎉**

Next time, just run:
```bash
python scripts/inference.py --model best.pt --source image.jpg
```

Happy detecting! 🚀
