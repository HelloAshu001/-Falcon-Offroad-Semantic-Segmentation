# 🏆 Falcon Offroad Semantic Segmentation

Production-ready **Semantic Segmentation pipeline** built for **Hackathon-level evaluation** using **Deep Learning (PyTorch)**.  
This project trains a model to classify **every pixel** in off-road scene images into predefined terrain/object classes.

---

## 🚀 Key Highlights 
- End-to-end pipeline: **Data → Training → Evaluation → Inference → App**
- Modular & clean codebase (easy to reproduce & extend)
- Strict dataset separation (train / val / test)
- Automatic checkpointing & best-model saving
- IoU-based evaluation (industry-standard metric)
- Ready for **deployment / demo during hackathon**

---

## 📁 Project Structure

```
project-root/
│
├── app/                    
│
├── configs/                
│   └── config.yaml
│
├── data/                   
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── testImages/
│
├── models/                 
│
├── utils/                 
│
├── scripts/                
│
├── runs/                  
│   ├── checkpoints/       
│   ├── logs/               
│   └── results/            
│
├── train.py               
├── check_data.py           
├── best_model.pth         
├── requirements.txt       
└── README.md
```

---

## 🧠 Model Overview
- **Task**: Semantic Segmentation
- **Classes**: 10
- **Architecture**: Encoder–Decoder based (via `segmentation-models-pytorch`)
- **Loss**: Pixel-wise segmentation loss
- **Metric**: Intersection over Union (IoU)

---

## ⚙️ Environment & Dependency Requirements

### Software
- Python **3.10.00 – 3.10.09**
- OS: Windows/ macOS

### Core Libraries
```
torch
torchvision
opencv-python
albumentations
segmentation-models-pytorch
numpy
streamlit
plotly
PyYAML
```

Install all dependencies:
```
pip install -r requirements.txt
```

---

## 💻 System Requirements 

| Component | Requirement |
|--------|-------------|
| CPU | 4+ cores |
| RAM | 8 GB (16 GB recommended) |
| GPU | NVIDIA GPU (CUDA supported) |
| VRAM | ≥ 4 GB |
| Disk | 12+ GB free space |

> ⚠️ CPU training is also supported but slower.

---

## 🧪 Dataset Preparation

- Images and masks must have **same filename**
- Mask pixels must contain **class indices (0–9)**

Verify dataset integrity:
```
python check_data.py
```

---

## ▶️ Step-by-Step: Train the Model

1. **Update config**
   - Edit `configs/config.yaml`
   - Set dataset paths, batch size, epochs

2. **Start Training**
```
python train.py
```

3. **During Training**
- Training & validation IoU printed per epoch
- Best model automatically saved

---

## 📦 Model Checkpoints & Outputs

After training:

```
runs/checkpoints/
 └── best_model.pth
```

- `best_model.pth` → Highest validation IoU model
- Logs → Stored in `runs/logs/`
- Metrics → Stored in `runs/results/`

---

## 🔁 Reproducing Final Results

To reproduce hackathon results:

1. Use **same config.yaml**
2. Keep dataset split unchanged
3. Run training with same seed
4. Load `best_model.pth` for inference

This ensures **deterministic & reproducible results**.

---

## 🔍 Running Inference / Demo App

Launch Streamlit app:
```
streamlit run app/app.py
```

Flow:
1. Upload image
2. Model loads `best_model.pth`
3. Segmentation mask generated
4. Mask overlaid on input image

---

## 📊 Expected Outputs & Interpretation

### Training Output
- **Loss ↓** → Model learning
- **IoU ↑** → Better segmentation

### IoU Values
| IoU Range | Interpretation |
|--------|----------------|
| < 0.5 | Poor segmentation |
| 0.5 – 0.7 | Acceptable |
| 0.7 – 0.9 | Strong |
| > 0.9 | Excellent / Overfitting possible |

### Visual Output
- Clean class boundaries
- Correct pixel-level classification

---

## 📝 Notes 
- Fully reproducible pipeline
- Clean & scalable codebase
- Industry-standard metrics
- Real-time demo ready
- Production-aligned design

---

## 🏁 Conclusion
This project demonstrates **strong ML fundamentals**, **engineering discipline**, and **deployment readiness** 
---

📌 *The program executed successfully.*

