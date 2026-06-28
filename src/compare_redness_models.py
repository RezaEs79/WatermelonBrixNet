import sys
import os

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)
# ----------------

import yaml
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

import warnings
warnings.filterwarnings('ignore')

from models import build_model
from data.dataset import create_metadata, get_validation_split, WatermelonDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---------------------------------------------------------
    # List of Redness Classification Models
    # ---------------------------------------------------------
    models_to_test = [
        {
            "name": "FusionNet",
            "config": "configs/red_fusion.yaml",
            "ckpt": "checkpoints/Redness_Fusion_ResNet/best.pth" 
        },
        {
            "name": "GatedNet",
            "config": "configs/red_gated.yaml",
            "ckpt": "checkpoints/Redness_Gated_EffV2/best.pth" # Check exact path in your system
        },
        {
            "name": "SwinNet",
            "config": "configs/red_swin.yaml",
            "ckpt": "checkpoints/Redness_Swin_Tiny/best.pth" # Check exact path in your system
        },
        {
            "name": "CrossAtt",
            "config": "configs/red_cross.yaml",
            "ckpt": "checkpoints/Redness_CrossAtt/best.pth"
        },
        {
            "name": "ConvNeXt",
            "config": "configs/red_convnext.yaml",
            "ckpt": "checkpoints/Redness_ConvNeXt/best.pth" # Check exact path in your system
        }
    ]

    print("Preparing Shared Validation Data for Redness Classification...")
    
    # Load base config just for dataset metadata
    with open(models_to_test[0]['config'], 'r') as f:
        base_config = yaml.safe_load(f)
        
    df_full = create_metadata(base_config['dataset']['root_dir'], excluded_ids=base_config['dataset'].get('excluded_ids', []))
    unique_ids = df_full['id'].unique()
    
    # Centralized split ensuring zero data leakage
    _, val_ids, split_method = get_validation_split(unique_ids, base_config['dataset'])
    
    print(f"📌 Split Method: {split_method}")
    print(f"👉 Validation IDs: {sorted(val_ids)}")

    val_df = df_full[df_full['id'].isin(val_ids)].copy()
    
    results = []

    for m_info in models_to_test:
        print(f"\n" + "="*50)
        print(f"Evaluating Classification Model: {m_info['name']}")
        print("="*50)
        
        if not os.path.exists(m_info['ckpt']):
            print(f"⚠️ Checkpoint not found for {m_info['name']}. Skipping...")
            continue
            
        with open(m_info['config'], 'r') as f:
            config = yaml.safe_load(f)
            
        # Ensure exact same preprocessing as training
        val_ds = WatermelonDataset(val_df, config, weights_map=None, augment=False)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config['dataset']['batch_size'], shuffle=False)

        # Build and Load Model
        model = build_model(config['model']['name'], num_classes=config['model']['num_classes'])
        model.load_state_dict(torch.load(m_info['ckpt'], map_location=device, weights_only=True))
        model.to(device)
        model.eval()

        y_pred_all = []
        y_true_all = []
        melon_ids_all = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Inference"):
                audio = batch['audio'].to(device)
                skin = batch['skin'].to(device)
                mass = batch['mass'].to(device)
                targets = batch['redness'].to(device)
                
                outputs = model(audio, skin, mass)
                _, predicted = torch.max(outputs, 1)
                
                y_pred_all.extend(predicted.cpu().numpy())
                y_true_all.extend(targets.cpu().numpy())
                melon_ids_all.extend(batch['id'].cpu().numpy())

        # Aggregate (Majority Vote) per watermelon
        temp_df = pd.DataFrame({
            'id': melon_ids_all,
            'pred': y_pred_all,
            'true': y_true_all
        })
        
        y_true_agg = []
        y_pred_agg = []
        
        for wm_id, group in temp_df.groupby('id'):
            pred_mode = group['pred'].mode().iloc[0]
            true_class = group['true'].iloc[0]
            y_pred_agg.append(pred_mode)
            y_true_agg.append(true_class)

        # Calculate Classification Metrics
        acc = accuracy_score(y_true_agg, y_pred_agg)
        macro_f1 = f1_score(y_true_agg, y_pred_agg, average='macro', zero_division=0)
        
        # Calculate Off-by-One Accuracy for Ordinal Data
        off_by_one_acc = np.mean(np.abs(np.array(y_true_agg) - np.array(y_pred_agg)) <= 1)

        results.append({
            'Model Architecture': m_info['name'],
            'Exact Acc (%)': round(acc * 100, 2),
            'Acc (±1) (%)': round(off_by_one_acc * 100, 2),
            'Macro F1-Score': round(macro_f1, 3)
        })

    # Display and Save Final Table
    if results:
        results_df = pd.DataFrame(results)
        
        # Sort by best Exact Accuracy (highest is better)
        results_df = results_df.sort_values(by='Exact Acc (%)', ascending=False).reset_index(drop=True)
        
        print("\n" + "✨"*25)
        print("🏆 FINAL CLASSIFICATION COMPARISON (REDNESS) 🏆")
        print("✨"*25)
        print(results_df.to_string(index=False))
        
        out_dir = './outputs/evaluation'
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, 'redness_comparison_table.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\n📁 Table saved to: {csv_path}")
    else:
        print("No models were evaluated. Check checkpoint paths.")

if __name__ == "__main__":
    main()