# 🏆 Falcon Offroad Semantic Segmentation

Production-ready **Semantic Segmentation pipeline** for Falcon off-road environments using **DeepLabV3+**.  
Built for the **Duality AI Falcon Hackathon**, with strict dataset separation, automated checkpoints, IoU logging, batch inference, and a Streamlit dashboard.

---

## 🚀 Key Features

- DeepLabV3+ with ResNet-101 backbone
- Pixel-wise semantic segmentation
- Strict **train / val / test** separation
- Automatic checkpoint saving (best & latest)
- IoU-based validation
- Batch prediction on test images
- Streamlit-based training dashboard

---

<h2>📁 Project Structure</h2>

<pre>
Falcon-Offroad-Semantic-Segmentation/
│
├── app/
│   └── main.py                 # Streamlit dashboard
│
├── configs/
│   └── config.yaml             # Configuration file
│
├── data/
│   ├── train/
│   │   ├── images/             # From train/color
│   │   └── masks/              # From train/segmentation
│   │
│   ├── val/
│   │   ├── images/             # From val/color
│   │   └── masks/              # From val/segmentation
│   │
│   └── testImages/
│       └── images/             # From testImages/color ONLY
│
├── models/
│   └── deeplabv3plus.py        # Model architecture
│
├── utils/
│   ├── dataset.py              # Dataset loader
│   ├── trainer.py              # Training logic
│   ├── metrics.py              # IoU calculation
│   ├── logger.py               # Training logs
│   └── checkpoint.py           # Model checkpoints
│
├── scripts/
│   └── predict_all.py          # Batch inference
│
├── runs/
│   ├── checkpoints/            # Saved models
│   ├── logs/                   # Training logs
│   └── results/                # Predictions
│
├── train.py                    # Training entry point
├── requirements.txt            # Dependencies
└── README.md
</pre>
## 📊 Dataset Explanation

The Falcon dataset follows this naming convention:

| Folder Name     | Description                     |
|-----------------|---------------------------------|
| `color/`        | RGB input images                |
| `segmentation/` | Pixel-wise ground truth masks   |

### Dataset Usage Rules

- **Training & Validation**
  - Use both `color` and `segmentation`
- **Test Dataset**
  - Use **ONLY** `color`
  - ❌ Never use test segmentation for training

---

## ▶️ Step-by-Step Instructions to Run and Test the Model

Follow the steps below to train the model and test it on the Falcon off-road dataset.

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/Tushar7902/Falcon-Offroad-Semantic-Segmentation.git
cd Falcon-Offroad-Semantic-Segmentation
```
### Step 2: Set Up the Environment

Ensure Python 3.11 is installed. Create and activate the environment, then install dependencies:
```bash
conda create -n falcon python=3.11 -y
conda activate falcon
pip install -r requirements.txt
```

### Step 3: Prepare the Dataset
Place the dataset in the following structure:
```text
train/color        → data/train/images
train/segmentation → data/train/masks

val/color          → data/val/images
val/segmentation   → data/val/masks

testImages/color   → data/testImages/images
```
⚠️ Test segmentation masks must not be used for training.

### Step 4: Train the Model
Run the training script:
```bash
python train.py
```
Outputs generated:

- Best model checkpoint: runs/checkpoints/best_model.pth

- Training logs: runs/logs/training_log.csv

### Step 5: Test the Model (Inference)
Run inference on the test images:
```bash
python scripts/predict_all.py
```
Outputs generated:

- Predicted segmentation masks saved in: runs/results/
### Step 6: Verify the Outputs
- Ensure output masks are generated for each test image.

- Pixel values in output masks correspond to semantic classes.

- Higher validation IoU indicates better segmentation performance.
## Optional: Visualize Training Progress
```bash
streamlit run app/main.py
```
## 🔁 Reproducing the Final Results

This section explains how to reproduce the final results using the trained model checkpoint.

---

### Step 1: Set Up the Environment

Ensure the environment and dependencies are installed as described in the **Environment Setup** section.

```bash
conda activate falcon
```
## Step 2: Prepare the Dataset
Verify that the dataset is placed correctly:
```bash
data/
├── train/images
├── train/masks
├── val/images
├── val/masks
└── testImages/images
```

⚠️ Only test images are used during this step.
Test segmentation masks are not used.
## Step 3: Use the Trained Model
After training, the best-performing model is saved automatically at:
```bash
runs/checkpoints/best_model.pth
```
This checkpoint is selected based on validation IoU.
## Step 4: Run Inference on Test Images
Generate the final segmentation results by running:
```bash
python scripts/predict_all.py
```
## Step 5: Locate the Final Outputs
The reproduced results are saved in:
```bash
runs/results/
```
Each output file corresponds to one input test image and contains pixel-wise semantic predictions.
## Step 6: Result Interpretation
- Output images represent predicted segmentation masks.

- Each pixel value corresponds to a semantic class defined by the dataset.

- Model performance is evaluated using Intersection over Union (IoU) on the validation set.

- Final test evaluation is performed by the challenge organizers using hidden ground truth.
## Reproducibility Statement
- All results are reproducible by following the steps above.

- No external data or test annotations are used during training or inference.

- The model behavior is deterministic given the same dataset and configuration.
## 🛠️ Environment & Dependency Requirements

The project was developed and tested using the following environment configuration.

### System Requirements

- **Operating System:** Linux / macOS / Windows  
- **Python Version:** 3.11 (tested and supported)  
- **Framework:** PyTorch  
- **GPU:** Optional (CUDA-enabled GPU recommended for faster training)

> ⚠️ Python versions **3.12 and above** are not fully supported by PyTorch at the time of development and may cause installation or build issues.

---

### Environment Setup

Create and activate a dedicated Conda environment and install all required dependencies using the commands below:

```bash
conda create -n falcon python=3.11 -y
conda activate falcon
pip install -r requirements.txt
```
## Python Dependencies
All required Python packages are listed in the requirements.txt file, including but not limited to:
- torch
- torchvision
- numpy
- opencv-python
- matplotlib
- albumentations
- segmentation-models-pytorch
- streamlit
- Installing dependencies from requirements.txt ensures a consistent and reproducible environment across different systems.
## 📊 Notes on Expected Outputs and How to Interpret Them

This section explains the outputs generated by the model and how to interpret them correctly.

---

### 📁 Output Directory Structure

After training and inference, the following output directories are created automatically:

```text
runs/
├── checkpoints/
│   ├── best_model.pth      # Best model based on validation IoU
│   └── latest.pth          # Most recent training checkpoint
│
├── logs/
│   └── training_log.csv    # Training loss and IoU per epoch
│
└── results/
    └── *.png               # Predicted segmentation masks for test images
```
## 🧠 Model Checkpoints
- best_model.pth

  - Represents the model with the highest validation IoU.

  - Used to generate final test predictions.

- latest.pth

  - Stores the most recent training state.

  - Useful for resuming training.
## 📈 Training Logs
The file training_log.csv contains:
- Epoch number
- Training loss
- Validation IoU
Interpretation:
- Decreasing loss indicates improved learning.
- Increasing IoU indicates better segmentation performance.
- The epoch with the highest IoU corresponds to best_model.pth.
## 🖼️ Predicted Segmentation Outputs
- Predicted masks are saved in:
```bash
runs/results/
```
- Each output file corresponds to a test input image.
- Output images are single-channel segmentation masks.
Pixel Interpretation:
- Each pixel value represents a semantic class.
- Pixel values map directly to class labels defined by the Falcon dataset.
- Regions with the same pixel value belong to the same class.
## 📏 Evaluation Metric
- Metric Used: Intersection over Union (IoU)
- IoU is calculated on the validation set during training.
- Final test IoU is computed by the challenge organizers using hidden ground truth.
- Higher IoU values indicate better segmentation quality.
## ✅ Expected Results Summary
- Checkpoints saved successfully after training
- Training logs available for performance analysis
- Segmentation masks generated for all test images
- Outputs reproducible using the provided steps
## 🏆 Interpretation Guidelines
- Well-segmented regions align closely with scene objects.
- Sharp boundaries indicate good class separation.
- Misclassified regions may indicate class imbalance or visual similarity.
- All outputs can be reproduced by following the steps outlined in this README.
## 📌 Results Summary

The model successfully generates pixel-wise segmentation masks for all test images using the best checkpoint selected based on validation IoU.

