import sys
import os

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)
# ----------------

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import librosa
import cv2
import warnings

warnings.filterwarnings('ignore')
from data.dataset import create_metadata

def extract_features(df):
    features = []
    
    print(f"\n🔍 Detected Columns in Metadata: {df.columns.tolist()}")
    
    # Group by watermelon ID to calculate aggregated per-watermelon stats
    for wm_id, group in tqdm(df.groupby('id'), desc="Extracting Modality Features"):
        brix = group['brix'].iloc[0] if 'brix' in df.columns else 0
        redness = group['redness'].iloc[0] if 'redness' in df.columns else 0
        mass = group['mass'].iloc[0] if 'mass' in df.columns else 0
        
        skin_sats = []
        skin_vals = []
        audio_bws = []
        audio_rolloffs = []
        
        # Iterate over each row (tap) in the watermelon group
        for _, row in group.iterrows():
            folder = row.get('folder_path', '')
            
            # 1. --- PROCESS AUDIO ---
            audio_file = row.get('audio_file')
            if pd.notna(audio_file):
                # Build absolute path
                if os.path.exists(str(audio_file)):
                    audio_path = str(audio_file)
                else:
                    audio_path = os.path.join(folder, str(audio_file))
                    
                if os.path.exists(audio_path):
                    try:
                        y, sr = librosa.load(audio_path, sr=None)
                        if len(y) > 0:
                            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                            impact_sample = librosa.frames_to_samples(np.argmax(onset_env))
                            
                            # Extract 0.3 seconds AFTER the impact
                            decay_y = y[impact_sample : impact_sample + int(0.3 * sr)]
                            
                            if len(decay_y) > int(0.05 * sr):
                                bw = librosa.feature.spectral_bandwidth(y=decay_y, sr=sr)
                                rolloff = librosa.feature.spectral_rolloff(y=decay_y, sr=sr, roll_percent=0.85)
                                audio_bws.append(np.mean(bw))
                                audio_rolloffs.append(np.mean(rolloff))
                    except Exception:
                        pass
            
            # 2. --- PROCESS SKIN IMAGES ---
            skin_file = row.get('skin_file')
            if pd.notna(skin_file):
                # Build absolute path for Skin
                if os.path.exists(str(skin_file)):
                    skin_path = str(skin_file)
                else:
                    skin_path = os.path.join(folder, str(skin_file))
                    if not os.path.exists(skin_path):
                        skin_path = os.path.join(folder, 'Skin', str(skin_file))
                        
                if os.path.exists(skin_path):
                    try:
                        img = cv2.imread(skin_path)
                        if img is not None:
                            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                            skin_sats.append(np.mean(hsv[:,:,1])) # Saturation
                            skin_vals.append(np.mean(hsv[:,:,2])) # Brightness
                    except Exception:
                        pass

        # Calculate Means for this Watermelon
        mean_sat = np.mean(skin_sats) if skin_sats else 0
        mean_val = np.mean(skin_vals) if skin_vals else 0
        mean_bw = np.mean(audio_bws) if audio_bws else 0
        mean_rolloff = np.mean(audio_rolloffs) if audio_rolloffs else 0
        
        features.append({
            'Target: Brix': brix,
            'Target: Redness': redness,
            'Input: Mass (kg)': mass,
            'Input: Skin (Saturation)': mean_sat,
            'Input: Skin (Brightness)': mean_val,
            'Input: Audio (Bandwidth)': mean_bw,      
            'Input: Audio (Rolloff)': mean_rolloff    
        })
        
    return pd.DataFrame(features)

def main():
    config_path = "configs/sugar_fusion.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Could not find {config_path}.")
        return
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    root_dir = config['dataset']['root_dir']
    print(f"Loading dataset from: {root_dir}")
    
    df_full = create_metadata(root_dir, excluded_ids=config['dataset'].get('excluded_ids', []))
    df_features = extract_features(df_full)
    
    print("\nComputing Spearman Correlation Matrix...")
    corr_matrix = df_features.corr(method='spearman')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdYlBu_r', vmin=-1, vmax=1, 
                linewidths=0.5, cbar_kws={"shrink": .8}, square=True)
    
    plt.title('Spearman Correlation between Multimodal Inputs and Quality Targets', 
              fontsize=14, pad=15, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    out_dir = './outputs/evaluation'
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, 'correlation_heatmap_advanced.png')
    
    plt.savefig(plot_path, dpi=300)
    print(f"\n✅ Advanced Statistical Correlation Heatmap saved to: {plot_path}")

if __name__ == "__main__":
    main()