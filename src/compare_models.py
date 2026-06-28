import os
import sys
# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)
# ----------------
import re
import glob
import warnings
import yaml
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torchaudio.transforms as AT
import torchvision.transforms as VT
from PIL import Image
from data.dataset import create_metadata, get_validation_split
warnings.filterwarnings('ignore')
from models import build_model

# ==========================================
# 1. Dataset Functions (Standalone for safety)
# ==========================================
EXCLUDED_IDS = [16, 20, 23, 24, 25, 31]
SAMPLE_RATE_M4A = 96000
SAMPLE_RATE_CSV = 20000
DURATION = 1.0

def load_audio_csv(file_path):
    try:
        df = pd.read_csv(file_path, header=None)
        col_idx = 1 if df.shape[1] > 1 else 0
        return pd.to_numeric(df.iloc[:, col_idx], errors='coerce').dropna().values
    except: return None

def load_m4a_mono(file_path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(file_path, sr=None, mono=False)
        if y.ndim > 1: y = np.mean(y, axis=0)
        return y, sr
    except: return None, None

def segment_tap_centered(signal, sample_rate, window_duration=1.0):
    if signal is None or len(signal) == 0: return []
    peaks, _ = find_peaks(np.abs(signal), height=np.max(np.abs(signal))*0.2, distance=int(0.4*sample_rate))
    segments = []
    for p in peaks[:3]:
        win = int(window_duration * sample_rate)
        pre_peak = int(0.1 * win)
        start = max(0, p - pre_peak)
        end = min(len(signal), start + win)
        seg = signal[start:end]
        if len(seg) < win: seg = np.pad(seg, (0, win - len(seg)), 'constant')
        segments.append(torch.from_numpy(seg).float())
    return segments

def create_multimodal_metadata(root_dir):
    data_records = []
    if not os.path.exists(root_dir): return pd.DataFrame()
    folders = sorted([f for f in os.listdir(root_dir) if f.startswith('watermelon')])
    for folder in folders:
        match = re.search(r'watermelon(\d+)', folder)
        if not match: continue
        wm_id = int(match.group(1))
        if wm_id in EXCLUDED_IDS: continue
        parts = folder.split('_')
        if len(parts) < 3: continue
        try:
            mass = float(parts[1].replace('kg', ''))
            brix = float(parts[2].replace('Brix', ''))
        except ValueError: continue
        
        full_path = os.path.join(root_dir, folder)
        audio_files = glob.glob(os.path.join(full_path, '*.m4a')) + glob.glob(os.path.join(full_path, '*.csv'))
        
        skin_path = os.path.join(full_path, 'Skin')
        skin_images = glob.glob(os.path.join(skin_path, '*.*')) if os.path.exists(skin_path) else []
        skin_images = [f for f in skin_images if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        skin_file = skin_images[0] if skin_images else None
        
        for f_path in audio_files:
            fname = os.path.basename(f_path)
            ftype = 'M4A' if fname.endswith('.m4a') else 'CSV'
            for i in range(3):
                data_records.append({'id': wm_id, 'mass': mass, 'brix': brix, 'folder_path': full_path, 'audio_file': fname, 'skin_file': skin_file, 'source': ftype, 'tap_idx': i})
    return pd.DataFrame(data_records)

class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, img_size=(128, 128)):
        self.df = dataframe
        self.img_size = img_size
        self.mel_transform = AT.MelSpectrogram(sample_rate=SAMPLE_RATE_M4A, n_fft=4096, hop_length=1024, n_mels=128)
        self.db_transform = AT.AmplitudeToDB()
        self.audio_resize = VT.Resize(img_size, antialias=True)
        self.img_transform = VT.Compose([VT.Resize((224, 224)), VT.ToTensor(), VT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(row['folder_path'], row['audio_file'])
        
        if row['source'] == 'M4A':
            sig, sr = load_m4a_mono(file_path)
            if sig is None: sig = np.zeros(SAMPLE_RATE_M4A)
        else:
            sig = load_audio_csv(file_path)
            sr = SAMPLE_RATE_CSV
            if sig is None: sig = np.zeros(sr)

        taps = segment_tap_centered(sig, sr, DURATION)
        tap_idx = int(row['tap_idx'])
        tensor = taps[tap_idx] if taps and tap_idx < len(taps) else (taps[0] if taps else torch.zeros(int(DURATION*sr)))
        
        if row['source'] == 'CSV': 
            if not isinstance(tensor, torch.Tensor): tensor = torch.from_numpy(tensor).float()
            tensor = AT.Resample(SAMPLE_RATE_CSV, SAMPLE_RATE_M4A)(tensor)
            
        target_len = int(DURATION * SAMPLE_RATE_M4A)
        if len(tensor) < target_len: tensor = torch.nn.functional.pad(tensor, (0, target_len - len(tensor)))
        else: tensor = tensor[:target_len]

        spec = self.db_transform(self.mel_transform(tensor))
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
        spec = self.audio_resize(spec.unsqueeze(0)).repeat(3, 1, 1)

        skin_path = row['skin_file']
        if skin_path and os.path.exists(skin_path):
            try: skin_tensor = self.img_transform(Image.open(skin_path).convert('RGB'))
            except: skin_tensor = torch.zeros(3, 224, 224)
        else: skin_tensor = torch.zeros(3, 224, 224)

        mass_tensor = torch.tensor([float(row['mass']) / 10.0], dtype=torch.float32)
        target = float(row['brix'])

        return spec, skin_tensor, mass_tensor, torch.tensor(target, dtype=torch.float32)

# ==========================================
# 2. Main Comparison Logic
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---------------------------------------------------------
    # IMPORTANT: Update these paths if your checkpoint folders 
    # have slightly different names!
    # ---------------------------------------------------------
    models_to_test = [
        {
            "name": "FusionNet",
            "config": "configs/sugar_fusion.yaml",
            "ckpt": "checkpoints/Sugar_Fusion_ResNet/best.pth" # Adjust path if needed
        },
        {
            "name": "GatedNet",
            "config": "configs/sugar_gated.yaml",
            "ckpt": "checkpoints/Sugar_Gated_EffV2/best.pth" # Adjust path if needed
        },
        {
            "name": "SwinNet",
            "config": "configs/sugar_swin.yaml",
            "ckpt": "checkpoints/Sugar_Swin_Tiny/best.pth" # Adjust path if needed
        },
        {
            "name": "CrossAtt",
            "config": "configs/sugar_cross.yaml",
            "ckpt": "checkpoints/Sugar_CrossAtt/best.pth" # Adjust path if needed
        },
        {
            "name": "ConvNeXt",
            "config": "configs/sugar_convnext.yaml",
            "ckpt": "checkpoints/Sugar_ConvNeXt/best.pth" # Adjust path if needed
        }
    ]

    # Shared Dataset Setup (Assuming all use the same root_dir)
    print("Preparing Shared Validation Data...")
    
    # We load the config of the first model just to get the dataset path
    with open(models_to_test[0]['config'], 'r') as f:
        base_config = yaml.safe_load(f)
        
    df_full = create_multimodal_metadata(base_config['dataset']['root_dir'])
    unique_ids = df_full['id'].unique()
    
    _, val_ids, split_method = get_validation_split(unique_ids, base_config['dataset'])

    print(f"📌 Split Method: {split_method}")
    print(f"👉 Validation IDs: {sorted(val_ids)}")

    val_df = df_full[df_full['id'].isin(val_ids)]
    
    results = []

    for m_info in models_to_test:
        print(f"\n" + "="*50)
        print(f"Evaluating Model: {m_info['name']}")
        print("="*50)
        
        if not os.path.exists(m_info['ckpt']):
            print(f"⚠️ Checkpoint not found for {m_info['name']}. Skipping...")
            continue
            
        with open(m_info['config'], 'r') as f:
            config = yaml.safe_load(f)
            
        # Handle specific image sizes (128 for ResNet, 224 for Swin/ConvNeXt)
        img_size_cfg = config['audio'].get('img_size', [128, 128])
        val_ds = MultimodalDataset(val_df, img_size=tuple(img_size_cfg))
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config['dataset']['batch_size'], shuffle=False)

        # Build and Load Model
        model = build_model(config['model']['name'], num_classes=config['model']['num_classes'])
        model.load_state_dict(torch.load(m_info['ckpt'], map_location=device, weights_only=True))
        model.to(device)
        model.eval()

        y_true = []
        y_pred = []

        with torch.no_grad():
            for audio, skin, mass, targets in tqdm(val_loader, desc=f"Inference"):
                audio, skin, mass = audio.to(device), skin.to(device), mass.to(device)
                
                # Handling the ConvNeXt dimension fix inside evaluation if needed
                # (Assuming the models/__init__.py correctly imports the updated ConvNextNet)
                outputs = model(audio, skin, mass)
                
                y_pred.extend(outputs.cpu().numpy().flatten())
                y_true.extend(targets.cpu().numpy().flatten())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Calculate Metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        int_acc = np.mean(np.round(y_true) == np.round(y_pred)) * 100

        results.append({
            'Model Architecture': m_info['name'],
            'MAE (Brix)': round(mae, 3),
            'RMSE (Brix)': round(rmse, 3),
            'R-Squared (R2)': round(r2, 3),
            'Accuracy (%)': round(int_acc, 1)
        })

    # Display and Save Final Table
    if results:
        results_df = pd.DataFrame(results)
        
        # Sort by best RMSE (lowest is better)
        results_df = results_df.sort_values(by='RMSE (Brix)', ascending=True).reset_index(drop=True)
        
        print("\n" + "✨"*25)
        print("🏆 FINAL MODEL COMPARISON (VALIDATION SET) 🏆")
        print("✨"*25)
        print(results_df.to_string(index=False))
        
        out_dir = './outputs/evaluation'
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, 'model_comparison_table.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\n📁 Table saved to: {csv_path}")
    else:
        print("No models were evaluated. Check checkpoint paths.")

if __name__ == "__main__":
    main()