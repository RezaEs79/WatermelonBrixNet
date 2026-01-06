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
```bash
conda remove --name watermelon_lab --all -y
conda clean --all -y

conda create -n watermelon_lab python=3.10 -y
conda activate watermelon_lab
```

Install PyTorch (CUDA 12.1)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Install Dependencies
```
pip install pandas numpy librosa matplotlib scikit-learn tqdm pyyaml pillow
pip install opencv-python
```

Verification (GPU Check)
```
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"
```

📁 Project Structure
```
Watermelon_Project/
│
├── configs/            # Experiment configurations (YAML)
├── data/               # Dataset & preprocessing
├── models/             # Model architectures
├── utils/              # Helper utilities (metrics, plots, logging)
├── checkpoints/        # Saved models (per experiment)
├── outputs/            # Inference results & reports
├── main.py             # Entry point
├── train.py            # Training loop
├── inference.py        # Final inference
└── valid.py            # Evaluation & validation
```
---
## Dataset Preparation

Organize Raw Dataset
```
python -m utils.data_organizer \
  --input "/mnt/d/Watermelon_DataSet" \
  --output "./data/processed_dataset"
```

Automatic Redness Labeling
```
python -m utils.redness_grader --dataset "./data/processed_dataset"
```
Dataset Sanity Check
```
python test_dataset.py
```
---


🔊 Audio Processing Requirements
```
conda install -c conda-forge ffmpeg -y
```
---

## 🚀 Training
Sugar (Brix) Models
```
python src/train.py --config configs/sugar_gated.yaml
python src/train.py --config configs/sugar_fusion.yaml
```

Redness Models
```
python src/train.py --config configs/red_gated.yaml
python src/train.py --config configs/red_fusion.yaml
```

🔹 Each training run creates a separate experiment folder inside checkpoints/, ensuring results never overwrite each other.


## 🧪 Validation (Labeled Data)
```
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
```
python src/valid.py \
  --data_dir "./data/processed_dataset" \
  --sugar_ckpt "./checkpoints/Sugar_Gated_EffV2/best.pth" \
  --sugar_arch "gated" \
  --red_ckpt "./checkpoints/Redness_Fusion_ResNet/best.pth" \
  --red_arch "fusion" \
  --output "./outputs/validation_report_gated.csv"
```
---
## 🍉 Final Inference (Harvest Time!)

``` Bash
python inference.py \
  --sugar_ckpt "./checkpoints/SugarNet_MobileNetV3/best.pth" \
  --red_ckpt "./checkpoints/RednessNet_ResNet18/best.pth" \
  --output "./outputs/final_report.csv"
```

This produces a CSV file containing final predictions, ready to be attached to reports or publications.

---

## 🕶 Blind Test (Unlabeled Data)
```
python src/test.py \
  --data_dir "./data/processed_dataset" \
  --sugar_ckpt "./checkpoints/SugarNet_MobileNetV3/best.pth" \
  --red_ckpt "./checkpoints/RednessNet_ResNet18/best.pth" \
  --output "./outputs/blind_test_results.csv"
```

Use this when no labels (Brix / Flesh) are available.

---

## 📊 Visualization Tools
Input Visualization (Waveform + Mel-Spectrogram)
```
python -m utils.visualize_inputs --id 1
```

Top: audio waveform

Bottom: Mel-spectrogram

Bright colors = higher energy (healthy signal)

Dataset Distribution
```
python -m utils.visualize_distribution --output "watermelon_stats_final.png"
```

---
## 🎯 Conclusion

This project demonstrates a complete deep learning engineering pipeline, from dataset preparation to multimodal inference, using modern architectures and clean experiment management.

If all steps run successfully:
🎉 Congratulations! You have built a full-scale deep learning system from scratch.

### 📜 License

MIT License

### 👤 Author

RezaEs79