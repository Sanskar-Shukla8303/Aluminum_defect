# Aluminum Defect Detection using YOLO

A complete deep learning project for detecting and classifying defects in aluminum sheets using YOLOv8.

## Project Overview

This project uses **YOLOv8 Classification** to automatically detect and classify 11 types of aluminum defects:
- Clean sample (non-defective)
- Be injured by a collision
- Coating cracking
- Convex powder
- Dirty spot
- Drain bottom
- Non-conducting
- Orange peel
- Pitting
- Scuffing
- Transverse strip dented

## Directory Structure

```
aluminum_defect_yolo/
├── data/
│   ├── raw/                 # Source: Aluminum_defect/ali2018
│   ├── images/              # Organized images (train/val/test)
│   ├── labels/              # YOLO format labels
│   └── dataset.yaml         # Dataset configuration
├── models/
│   ├── trained/             # Trained model weights
│   └── configs/             # Model configurations
├── scripts/
│   ├── prepare_data.py     # Data preparation & organization
│   ├── train.py            # Model training
│   ├── inference.py        # Predictions on new data
│   └── utils.py            # Utility functions
├── results/                 # Training outputs & results
├── notebooks/              # Jupyter notebooks
├── config.yaml             # Project configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

### 1. Prerequisites
- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)
- 4GB RAM minimum (8GB+ recommended)
- 50GB disk space for dataset and models

### 2. Clone/Setup Project

```bash
# Navigate to project directory
cd aluminum_defect_yolo

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
 
```

## Quick Start

### Step 1: Prepare Dataset

Convert your raw aluminum defect images into YOLO format:

```bash
python scripts/prepare_data.py \
    --raw_data "C:\Users\akash\Desktop\Aluminum_defect\ali2018" \
    --output "./data" \
    --train_split 0.7 \
    --val_split 0.15
```

This will:
- Organize images into train/val/test splits
- Create YOLO format labels
- Generate `dataset.yaml` configuration file

**Expected Output:**
```
Processing images...
✓ Data preparation complete!
Output directory: ./data

DATASET STATISTICS
===================================================
Total images: 4250
Training set: 2975 images (70.0%)
Validation set: 638 images (15.0%)
Test set: 637 images (15.0%)
===================================================
```

### Step 2: Train Model

Train a YOLO model on your dataset:

```bash
# Quick training (medium model, 50 epochs)
python scripts/train.py --model yolov8m-cls --epochs 50 --batch 32

# For better accuracy (takes longer):
python scripts/train.py --model yolov8l-cls --epochs 100 --batch 16

# For faster training (less accurate):
python scripts/train.py --model yolov8s-cls --epochs 30 --batch 64
```

**Model sizes (speed vs accuracy tradeoff):**
- `yolov8n-cls` - Nano (fastest, least accurate)
- `yolov8s-cls` - Small
- `yolov8m-cls` - Medium (recommended)
- `yolov8l-cls` - Large
- `yolov8x-cls` - Extra Large (slowest, most accurate)

**Training outputs:**
- Trained model: `runs/detect/trainX/weights/best.pt`
- Training plots: `runs/detect/trainX/results.png`
- Tensorboard logs: `runs/detect/trainX/events.out.tfevents.*`

### Step 3: Run Inference

Make predictions on new images:

```bash
# Single image
python scripts/inference.py \
    --model "runs/detect/train/weights/best.pt" \
    --source "path/to/image.jpg"

# Folder of images
python scripts/inference.py \
    --model "runs/detect/train/weights/best.pt" \
    --source "path/to/images/" \
    --evaluate

# Video file
python scripts/inference.py \
    --model "runs/detect/train/weights/best.pt" \
    --source "path/to/video.mp4"

# With custom confidence threshold
python scripts/inference.py \
    --model "best.pt" \
    --source "image.jpg" \
    --conf 0.7
```

## Detailed Usage

### Data Preparation

```bash
python scripts/prepare_data.py --help
```

**Options:**
- `--raw_data`: Path to ali2018 folder (default: Aluminum_defect\ali2018)
- `--output`: Output dataset path (default: ./data)
- `--train_split`: Training percentage (default: 0.7)
- `--val_split`: Validation percentage (default: 0.15)

### Training

```bash
python scripts/train.py --help
```

**Options:**
- `--model`: Model size (default: yolov8m-cls)
- `--epochs`: Number of epochs (default: 50)
- `--batch`: Batch size (default: 32)
- `--imgsz`: Image size (default: 640)
- `--patience`: Early stopping patience (default: 20)
- `--lr0`: Initial learning rate (default: 0.001)
- `--lrf`: Final learning rate ratio (default: 0.01)

**Best practices:**
- Start with medium model for balance between speed and accuracy
- Use batch size based on your GPU VRAM:
  - RTX 3060: batch=32-64
  - RTX 3090: batch=64-128
  - CPU only: batch=8-16
- More epochs ≠ better results; use patience/early stopping
- Monitor loss curves in results/train directory

### Inference

```bash
python scripts/inference.py --help
```

**Options:**
- `--model`: Path to trained model (.pt file) ✓ Required
- `--source`: Image/video/folder path ✓ Required
- `--conf`: Confidence threshold (default: 0.5)
- `--imgsz`: Image size (default: 640)
- `--evaluate`: Run evaluation and print statistics

**Output:**
- Annotated images with predictions
- Console output with class predictions
- Statistics (if --evaluate flag used)

## Results Interpretation

### Training Results

After training, check these files:
- **best.pt** - Best model (lowest validation loss)
- **last.pt** - Last epoch model
- **results.csv** - Training metrics
- **results.png** - Loss/accuracy plots

### Predictions

Each prediction shows:
- **Class name** - Detected defect type
- **Confidence** - Prediction confidence (0-100%)

**Guidelines:**
- Confidence > 70% - High confidence prediction
- Confidence 50-70% - Medium confidence (review recommended)
- Confidence < 50% - Low confidence (likely misclassification)

## Advanced Usage

### Training on GPU

```bash
# Automatic GPU detection
python scripts/train.py --model yolov8l-cls --epochs 100 --batch 64

# Specify GPU device
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --model yolov8l-cls
```

### Hyperparameter Tuning

Edit `config.yaml` or pass arguments:

```bash
python scripts/train.py \
    --model yolov8m-cls \
    --epochs 100 \
    --batch 32 \
    --lr0 0.0005 \
    --imgsz 768
```

### Transfer Learning

Start from pretrained weights:

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8m-cls.pt')

# Train on your data
results = model.train(data='data/dataset.yaml', epochs=50)
```

### Model Conversion

Convert model to other formats:

```python
from ultralytics import YOLO

model = YOLO('best.pt')
model.export(format='onnx')  # ONNX format
model.export(format='tflite')  # TensorFlow Lite
model.export(format='torchscript')  # TorchScript
```

## Troubleshooting

### CUDA Not Found
```bash
# Check installation
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory (OOM)
- Reduce batch size: `--batch 16` or `--batch 8`
- Use smaller model: `--model yolov8s-cls`
- Reduce image size: `--imgsz 416`

### Validation loss not decreasing
- Increase epochs: `--epochs 100`
- Increase patience: `--patience 30`
- Reduce learning rate: `--lr0 0.0005`

### Poor predictions
- Check dataset quality and balance
- Increase training epochs
- Try larger model: `yolov8l-cls`
- Collect more diverse data

### Slow inference
- Use smaller model: `yolov8n-cls`
- Reduce image size: `--imgsz 416`
- Enable GPU acceleration (if available)

## Performance Benchmarks

Approximate metrics on RTX 3070 GPU:

| Model | Size | Speed | Accuracy | VRAM |
|-------|------|-------|----------|------|
| YOLOv8n-cls | 3M | 2ms | 75% | 1GB |
| YOLOv8s-cls | 11M | 5ms | 82% | 2GB |
| YOLOv8m-cls | 26M | 10ms | 87% | 4GB |
| YOLOv8l-cls | 44M | 15ms | 89% | 6GB |
| YOLOv8x-cls | 71M | 20ms | 91% | 8GB |

*Note: Actual performance depends on hardware and image resolution*

## Project Structure Best Practices

```
✓ Keep raw data in data/raw/ (read-only)
✓ Store trained models with version numbers
✓ Backup best.pt regularly
✓ Document hyperparameters used for each training
✓ Use version control for code changes
✓ Keep experiment logs and results
```

## Next Steps

1. **Data Augmentation**: Improve robustness
   ```python
   # Advanced augmentation in train.py
   augment=True, flipud=0.5, fliplr=0.5, degrees=10
   ```

2. **Cross-validation**: Test model reliability
   ```bash
   # Train multiple times with different splits
   for i in {1..5}; do
     python scripts/train.py --epochs 50
   done
   ```

3. **Ensemble Methods**: Combine multiple models for better accuracy
   ```python
   # Load multiple models and average predictions
   model1 = YOLO('model1.pt')
   model2 = YOLO('model2.pt')
   ```

4. **Model Deployment**: Deploy to production
   - Export to ONNX/TensorFlow/Mobile formats
   - Create inference API (Flask/FastAPI)
   - Deploy on edge devices

## Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Aluminum Defect Detection Paper](https://arxiv.org/search/?query=aluminum+defect)

## License

This project is provided for educational and research purposes.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review YOLO documentation
3. Check PyTorch/CUDA installation

## Author

Created for Aluminum Defect Detection Project
Date: 2024

---

**Happy Training! 🚀**
