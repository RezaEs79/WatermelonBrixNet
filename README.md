# 🍉 MelonBrix
**Deep Learning–based Multimodal Watermelon Sweetness Estimation**

MelonBrix is a deep learning project for **non-destructive estimation of watermelon sugar content (Brix value)** and **flesh redness**, using a combination of **image data and acoustic signals**.

The system leverages **computer vision**, **audio signal processing**, and **multimodal fusion networks** to provide accurate and objective quality assessment for agricultural applications.

---

## ✨ Features
- Sugar (Brix) regression using image + audio data  
- Redness classification of watermelon flesh  
- Multimodal fusion (Gated / Feature Fusion)  
- Support for CNNs, EfficientNetV2, MobileNet, ResNet, Swin Transformer  
- Modular experiment-based training (no checkpoint conflicts)  
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
├── configs/            # Experiment configurations (YAML)
├── data/               # Dataset & preprocessing
├── models/             # Model architectures
├── utils/              # Helper utilities (metrics, plots, logging)
├── checkpoints/        # Saved models (per experiment)
├── outputs/            # Inference results & reports
├── train.py            # Training loop
├── inference.py        # Final inference
└── valid.py            # Evaluation & validation
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

Redness Models
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
## 🎯 Conclusion

This project demonstrates a complete deep learning engineering pipeline, from dataset preparation to multimodal inference, using modern architectures and clean experiment management.


### 📜 License

MIT License

### 👤 Author

RezaEs79

