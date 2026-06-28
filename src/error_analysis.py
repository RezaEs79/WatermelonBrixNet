import os
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import build_model
from evaluate import create_multimodal_metadata, MultimodalDataset # Reusing the functions

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Find Worst Predictions (Error Analysis)")
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to trained model .pth file')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Prepare Validation Data
    print("Preparing Validation Data for Error Analysis...")
    df_full = create_multimodal_metadata(config['dataset']['root_dir'])
    unique_ids = df_full['id'].unique()
    _, val_ids = train_test_split(unique_ids, test_size=config['dataset']['val_split'], random_state=42)
    
    # Keep validation dataframe to track original IDs
    val_df = df_full[df_full['id'].isin(val_ids)].copy()
    
    # Reset index to match DataLoader iteration order
    val_df.reset_index(drop=True, inplace=True)
    
    img_size_cfg = config['audio'].get('img_size', [128, 128])
    val_ds = MultimodalDataset(val_df, img_size=tuple(img_size_cfg))
    
    # Note: batch_size=1 makes tracking specific samples much easier and bug-free
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

    # 3. Load Model
    print("Loading Model...")
    model = build_model(config['model']['name'], num_classes=config['model']['num_classes'])
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 4. Run Inference and track errors
    results = []
    
    print("Running Inference...")
    with torch.no_grad():
        for idx, (audio, skin, mass, targets) in enumerate(tqdm(val_loader)):
            audio, skin, mass = audio.to(device), skin.to(device), mass.to(device)
            output = model(audio, skin, mass)
            
            pred_val = output.item()
            true_val = targets.item()
            error = abs(pred_val - true_val)
            
            # Retrieve original metadata from dataframe
            row_info = val_df.iloc[idx]
            
            results.append({
                'Watermelon_ID': row_info['id'],
                'Folder': os.path.basename(row_info['folder_path']),
                'Actual_Brix': true_val,
                'Predicted_Brix': round(pred_val, 2),
                'Absolute_Error': round(error, 2)
            })

    # 5. Convert to DataFrame and sort by worst errors
    res_df = pd.DataFrame(results)
    
    # We might have multiple taps (audio segments) for the same watermelon
    # So we group by Watermelon ID and average the predictions/errors
    grouped_res = res_df.groupby(['Watermelon_ID', 'Folder', 'Actual_Brix']).agg(
        Mean_Predicted_Brix=('Predicted_Brix', 'mean'),
        Mean_Absolute_Error=('Absolute_Error', 'mean')
    ).reset_index()
    
    # Sort descending to bring worst predictions to the top
    worst_predictions = grouped_res.sort_values(by='Mean_Absolute_Error', ascending=False)
    
    print("\n" + "="*60)
    print("🚨 WORST PREDICTIONS (ERROR ANALYSIS) 🚨")
    print("="*60)
    print(worst_predictions.head(5).to_string(index=False))
    print("="*60)
    print("📌 Tip: Check the raw audio and image files for the IDs with the highest errors.")

if __name__ == "__main__":
    main()