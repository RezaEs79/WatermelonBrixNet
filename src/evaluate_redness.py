import sys
import os

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)
# ----------------

import argparse
import yaml
import warnings
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings('ignore')

from models import build_model
from data.dataset import create_metadata, get_validation_split, WatermelonDataset

def main():
    parser = argparse.ArgumentParser(description="Evaluate Sample-Level Redness Classification")
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to trained model .pth file')
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation', help='Directory to save plots')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = config['model']['name'].upper()

    print("Preparing Validation Data for Classification...")
    # Use centralized dataset functions for zero data leakage
    df_full = create_metadata(config['dataset']['root_dir'], excluded_ids=config['dataset'].get('excluded_ids', []))
    unique_ids = df_full['id'].unique()
    
    _, val_ids, split_method = get_validation_split(unique_ids, config['dataset'])
    
    print(f"📌 Split Method: {split_method}")
    print(f"👉 Validation IDs: {sorted(val_ids)}")
    
    val_df = df_full[df_full['id'].isin(val_ids)].copy()
    
    # Use identical Dataset class as training (ensures exact same preprocessing)
    val_ds = WatermelonDataset(val_df, config, weights_map=None, augment=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config['dataset']['batch_size'], shuffle=False)

    print(f"Loading {model_name} Classification Model...")
    model = build_model(config['model']['name'], num_classes=config['model']['num_classes'])
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    y_pred_all = []
    y_true_all = []

    print("Running Inference on Validation Set...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            audio = batch['audio'].to(device)
            skin = batch['skin'].to(device)
            mass = batch['mass'].to(device)
            targets = batch['redness'].to(device)
            
            outputs = model(audio, skin, mass)
            _, predicted = torch.max(outputs, 1)
            
            y_pred_all.extend(predicted.cpu().numpy())
            y_true_all.extend(targets.cpu().numpy())

    # Map predictions back to the dataframe to group by watermelon
    val_df['predicted_class'] = y_pred_all
    val_df['true_class'] = y_true_all
    
    # Aggregate (Majority Vote) per watermelon
    y_true_agg = []
    y_pred_agg = []
    melon_ids = []
    
    for wm_id, group in val_df.groupby('id'):
        # Majority vote (Mode) for classification
        pred_mode = group['predicted_class'].mode().iloc[0]
        true_class = group['true_class'].iloc[0]
        
        y_pred_agg.append(pred_mode)
        y_true_agg.append(true_class)
        melon_ids.append(wm_id)

    # Calculate final sample-level metrics
    acc = accuracy_score(y_true_agg, y_pred_agg)
    
    # Calculate Off-by-One Accuracy for Ordinal Data
    off_by_one_acc = np.mean(np.abs(np.array(y_true_agg) - np.array(y_pred_agg)) <= 1)
    
    conf_matrix = confusion_matrix(y_true_agg, y_pred_agg)
    class_report = classification_report(y_true_agg, y_pred_agg, zero_division=0)
    
    print("\n" + "="*50)
    print(f"🏆 SAMPLE-LEVEL CLASSIFICATION RESULTS: {model_name} 🏆")
    print("="*50)
    print(f"Total Unique Watermelons Evaluated : {len(y_true_agg)}")
    print(f"Aggregated Exact Accuracy          : {acc*100:.2f}%")
    print(f"Aggregated Accuracy (±1 Grade)     : {off_by_one_acc*100:.2f}%")
    print("\nClassification Report (Aggregated):\n", class_report)
    print("="*50)

    print("\n📝 Exact Predicted Classes per Watermelon:")
    print("-" * 50)
    for m_id, t_cls, p_cls in zip(melon_ids, y_true_agg, y_pred_agg):
        # Adding 1 to display human-readable classes (1 to 5) instead of internal PyTorch (0 to 4)
        print(f"Watermelon ID: {m_id:02d} | Actual Class: {int(t_cls)+1} | Predicted Class: {int(p_cls)+1}")
    print("-" * 50)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    
    present_classes = sorted(list(set(y_true_agg) | set(y_pred_agg)))
    class_names = [f"Class {int(c) + 1}" for c in present_classes]
    
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'{model_name}: Aggregated Redness Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Class (Ground Truth)', fontsize=12)
    plt.xlabel('Predicted Class (Majority Vote)', fontsize=12)
    plt.tight_layout()
    
    plot_path = os.path.join(args.output_dir, f"{model_name.lower()}_aggregated_redness_cm.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📈 Final Paper-ready Confusion Matrix saved to {plot_path}")

if __name__ == "__main__":
    main()
'''
    python src/evaluate_redness.py \
  --config configs/red_fusion.yaml \
  --ckpt checkpoints/Redness_Fusion_ResNet/best.pth
'''