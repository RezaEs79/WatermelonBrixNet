<<<<<<< HEAD
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

Note: The dataset split mechanism dynamically enforces zero data-leakage across all trained models.

Full Dataset Training (100% Data for Production / Final Deployment):

To train the best-performing architectures on the entire dataset without holding out a validation split:
```Bash
python src/train_full.py --config configs/sugar_fusion.yaml --epochs 25 --save_name best_full.pth
python src/train_full.py --config configs/red_cross.yaml --epochs 25 --save_name best_full.pth
```
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
=======
# WatermelonBrixNet



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

* [Create](https://docs.gitlab.com/user/project/repository/web_editor/#create-a-file) or [upload](https://docs.gitlab.com/user/project/repository/web_editor/#upload-a-file) files
* [Add files using the command line](https://docs.gitlab.com/topics/git/add_files/#add-files-to-a-git-repository) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://git.iut.ac.ir/40202634/watermelonbrixnet.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

* [Set up project integrations](https://git.iut.ac.ir/40202634/watermelonbrixnet/-/settings/integrations)

## Collaborate with your team

* [Invite team members and collaborators](https://docs.gitlab.com/user/project/members/)
* [Create a new merge request](https://docs.gitlab.com/user/project/merge_requests/creating_merge_requests/)
* [Automatically close issues from merge requests](https://docs.gitlab.com/user/project/issues/managing_issues/#closing-issues-automatically)
* [Enable merge request approvals](https://docs.gitlab.com/user/project/merge_requests/approvals/)
* [Set auto-merge](https://docs.gitlab.com/user/project/merge_requests/auto_merge/)

## Test and Deploy

Use the built-in continuous integration in GitLab.

* [Get started with GitLab CI/CD](https://docs.gitlab.com/ci/quick_start/)
* [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/user/application_security/sast/)
* [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/topics/autodevops/requirements/)
* [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/user/clusters/agent/)
* [Set up protected environments](https://docs.gitlab.com/ci/environments/protected_environments/)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
>>>>>>> 3cb635d2b525e28af2f936296d33f4fbd197ce07
