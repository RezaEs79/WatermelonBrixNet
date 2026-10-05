import sys
import os

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)
# ----------------

import yaml
import librosa
import numpy as np
import pandas as pd
from data.dataset import create_metadata
import warnings
warnings.filterwarnings('ignore')

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
    
    # ------------------ DYNAMIC COLUMN DETECTION ------------------
    audio_cols = [c for c in df_full.columns if any(k in c.lower() for k in ['audio', 'sound', 'tap', 'wav', 'm4a', 'mic'])]
    print(f"\n🔍 Detected Audio Columns: {audio_cols}")
    if not audio_cols:
        print("❌ CRITICAL ERROR: NO AUDIO COLUMNS FOUND IN DATAFRAME!")
        return
    # --------------------------------------------------------------

    # فقط 3 هندوانه اول را بررسی می‌کنیم تا لاگ‌ها شلوغ نشود
    test_ids = df_full['id'].unique()[:3]
    
    print("\n" + "="*60)
    print("🛠️ STARTING AUDIO EXTRACTION DEBUG 🛠️")
    print("="*60)
    
    for wm_id in test_ids:
        group = df_full[df_full['id'] == wm_id]
        print(f"\n🍉 Watermelon ID: {wm_id} | Rows in Group: {len(group)}")
        
        audio_paths = group[audio_cols].values.flatten()
        valid_paths = [p for p in pd.Series(audio_paths).dropna().unique() if isinstance(p, str)]
        
        print(f"   ➤ Found {len(valid_paths)} valid audio paths to process.")
        
        if len(valid_paths) == 0:
            print("   ❌ ERROR: Paths are empty or not strings!")
            continue
            
        # بررسی فقط یک فایل صوتی از این هندوانه
        test_file = valid_paths[0]
        print(f"   ➤ Processing File: {test_file}")
        
        if not os.path.exists(test_file):
            print(f"   ❌ ERROR: FILE DOES NOT EXIST ON DISK: {test_file}")
            continue
            
        try:
            # 1. Load Audio
            y, sr = librosa.load(test_file, sr=None)
            print(f"   ✅ Loaded successfully! Shape: {y.shape}, Sample Rate: {sr}")
            
            if len(y) == 0:
                print("   ❌ ERROR: Audio file is empty (length 0).")
                continue
                
            # 2. Find Impact
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            impact_idx = np.argmax(onset_env)
            impact_sample = librosa.frames_to_samples(impact_idx)
            impact_time = impact_sample / sr
            print(f"   ✅ Impact detected at: Sample {impact_sample} ({impact_time:.3f} seconds)")
            
            # 3. Extract Decay
            decay_length = int(0.3 * sr)
            decay_y = y[impact_sample : impact_sample + decay_length]
            print(f"   ✅ Extracted Decay Segment length: {len(decay_y)} samples")
            
            if len(decay_y) < int(0.05 * sr):
                print(f"   ❌ ERROR: Decay segment is too short! ({len(decay_y)} samples). This will be skipped in main code.")
                continue
                
            # 4. Calculate Features
            bw = librosa.feature.spectral_bandwidth(y=decay_y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=decay_y, sr=sr, roll_percent=0.85)
            
            print(f"   🌟 CALCULATED FEATURES:")
            print(f"      - Mean Bandwidth: {np.mean(bw):.2f}")
            print(f"      - Mean Rolloff:   {np.mean(rolloff):.2f}")
            
        except Exception as e:
            print(f"   ❌ EXCEPTION DURING PROCESSING: {e}")

if __name__ == "__main__":
    main()