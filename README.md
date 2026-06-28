# 🍉 MelonBrix
**Deep Learning–based Multimodal Watermelon Sweetness Estimation**

MelonBrix is a deep learning project for **non-destructive estimation of watermelon sugar content (Brix value)** and **flesh redness**, using a combination of **image data and acoustic signals**.

The system leverages **computer vision**, **audio signal processing**, and **multimodal fusion networks** to provide accurate and objective quality assessment for agricultural applications.

---

## ✨ Features


- Continuous Sugar (Brix) Regression: Accurately predict Brix levels using multimodal inputs.

- Ordinal Redness Classification: Automated CIELab-based grading for internal flesh quality (Grades 1 to 5). Evaluated using Exact and Off-by-One (±1) Accuracy.

- Advanced Multimodal Fusion: Implementations of baseline Feature Fusion, Gated Multimodal Units (GMU), and Cross-Modal Attention.

- State-of-the-art Backbones: Includes ResNet, EfficientNetV2, ConvNeXt, and Swin Transformers.

- Scientific Evaluation Pipeline: Zero-leakage deterministic train/val splits, sample-level (per-watermelon) prediction aggregation, and generation of publication-ready metrics (MAE, RMSE, $R^2$, Exact Accuracy, Accuracy ±1, F1-Score).
- End-to-end pipeline: dataset → training → validation → inference  

---

## 📦 Environment Setup

### Create Conda Environment
```Bash
conda remove --name watermelon_lab --all -y
conda clean --all -y

conda env create -f environment.yml
conda activate watermelon_lab
```

Verification (GPU Check)
```Bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"
```

📁 Project Structure
```Bash
Watermelon_Project/
│
├── configs/               # YAML experiment configurations (e.g., sugar_cross.yaml)
├── data/                  # Dataset logic & PyTorch Dataset class
├── models/                # Deep Learning architectures (Fusion, Gated, Swin, CrossAtt, ConvNeXt)
├── src/                   # Core execution scripts
│   ├── train.py                 # Main training loop with early stopping
│   ├── compare_models.py        # Compares all Sugar models on shared Val set
│   ├── compare_redness_models.py# Compares all Redness models on shared Val set
│   ├── evaluate_per_melon.py    # Generates final Brix scatter plots (Sample-Level)
│   ├── evaluate_redness.py      # Generates Redness confusion matrices (Sample-Level)
│   ├── test.py                  # Blind inference on unlabelled data
│   ├── valid.py                 # Legacy combined validation script
│   ├── error_analysis.py        # Finds extreme prediction outliers
│   └── visualize_Grad_CAM.py    # Explainable AI visualizer
├── utils/                 # Helper utilities
│   ├── data_organizer.py        # Parses raw folders
│   ├── redness_grader.py        # Automated CIELab redness labeling
│   ├── visualize_inputs.py      # Audio spectrogram & Skin visualizations
│   └── visualize_distribution.py# Data distribution plotting
├── checkpoints/           # Saved model weights (.pth files)
└── outputs/               # Saved figures, CSV tables, and XAI heatmaps

```
---
## Dataset Preparation

Organize Raw Dataset
```Bash
python -m utils.data_organizer \
  --input "/mnt/d/Watermelon_DataSet" \
  --output "./data/processed_dataset"
```

#### Automatic Redness Labeling

Now run this code on the dataset prepared in the previous step (data/processed_dataset) to rename the Flesh directories and assign the corresponding labels.

```Bash
python -m utils.redness_grader --dataset "./data/processed_dataset"
```
Dataset Sanity Check
```Bash
python test_dataset.py
```
---
## 🚀 Training
Train specific models using config files, for example:

Sugar (Brix) Models
```Bash
# Train Gated Network for Sugar Prediction
python src/train.py --config configs/sugar_gated.yaml
# or
python src/train.py --config configs/sugar_fusion.yaml
```

Redness Classification Models:
```Bash
python src/train.py --config configs/red_gated.yaml
# Train Fusion Network for Redness Classification
python src/train.py --config configs/red_fusion.yaml
```

🔹 Each training run creates a separate experiment folder inside checkpoints/, ensuring results never overwrite each other.

---
## 🧪 Validation (Labeled Data)
```Bash
python src/valid.py \
  --data_dir "./data/processed_dataset" \
  --sugar_ckpt "./checkpoints/SugarNet_MobileNetV3/best.pth" \
  --red_ckpt "./checkpoints/RednessNet_ResNet18/best.pth" \
  --output "./outputs/validation_report.csv"
```

Outputs include:

  - MAE (Sugar regression)

  - Accuracy (Redness classification)

  - Integer Error statistics

---

## 🧠 Multimodal Combined Validation
```Bash
python src/valid.py \
  --data_dir "./data/processed_dataset" \
  --sugar_ckpt "./checkpoints/Sugar_Gated_EffV2/best.pth" \
  --sugar_arch "gated" \
  --red_ckpt "./checkpoints/Redness_Fusion_ResNet/best.pth" \
  --red_arch "fusion" \
  --output "./outputs/validation_report_gated.csv"
```

---
## 🕶 Blind Test (Unlabeled Data)
```Bash
python src/test.py \
  --data_dir "./data/processed_dataset" \
  --sugar_ckpt "./checkpoints/Sugar_Gated_EffV2/best.pth" \
  # --sugar_arch "gated" \
  --red_ckpt "./checkpoints/Redness_Fusion_ResNet/best.pth" \
  # --red_arch "fusion"
  --output "./outputs/blind_test_results.csv"
```

Use this when no labels (Brix / Flesh) are available.

---

## 📊 Visualization Tools
#### Input Visualization (Waveform + Mel-Spectrogram)
```Bash
python -m utils.visualize_inputs --id 1
```
'id' in this command is related to watermelon name (id). 


Top: audio waveform

Bottom: Mel-spectrogram

Bright colors = higher energy (healthy signal)

#### Dataset Distribution
```Bash
python -m utils.visualize_distribution --output "watermelon_stats_final.png"
```
---
## 🔬 Scientific Evaluation & Analysis
Once the models are trained, use these scripts to extract scientific metrics.

#### 1. Generate Comparison Table (All Models)
  - Creates a comprehensive table comparing MAE, RMSE, and R-Squared across all trained architectures.
  
  ```Bash
  python src/compare_models.py
  ```
  - For Redness classification (Outputs Accuracy, Macro F1-Score):
  ```
  python src/compare_redness_models.py
  ```
####  2. Per-Melon Evaluation & Plotting

Evaluates a single chosen model and generates high-quality visualizations.

Brix Scatter Plot (Targeting Ideal Y=X Fit):
```
python src/evaluate_per_melon.py \
  --config configs/sugar_cross.yaml \
  --ckpt checkpoints/Sugar_CrossAtt/best.pth
```
#### 3. Evaluate a Single Model (Scatter Plot Generation)
```Bash
python src/evaluate.py \
  --config configs/sugar_fusion.yaml \
  --ckpt checkpoints/Sugar_Fusion_ResNet/best.pth
```
#### 4. Explainable AI (Grad-CAM)
Visualize where the model is looking to make its prediction.
```Bash

python src/visualize_Grad_CAM.py \
  --config configs/sugar_fusion.yaml \
  --ckpt checkpoints/Sugar_Fusion_ResNet/best.pth \
  --image_path data/processed_dataset/watermelon01_3kg_12Brix/Skin/IMG_1.jpg \
  --audio_path data/processed_dataset/watermelon01_3kg_12Brix/011.m4a
```
#### 4. Error Analysis
Find outliers and understand where the model struggles.
```Bash

python src/error_analysis.py \
  --config configs/sugar_fusion.yaml \
  --ckpt checkpoints/Sugar_Fusion_ResNet/best.pth
```
---
## 🎯 Conclusion

This project demonstrates a complete deep learning engineering pipeline, from dataset preparation to multimodal inference, using modern architectures and clean experiment management. By moving from traditional feature extraction to advanced multimodal transformers and explainable AI, this project bridges the gap between agricultural research and modern artificial intelligence.


### 📜 License

MIT License

### 👤 Author

RezaEs79