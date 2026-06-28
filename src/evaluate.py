import os
import re
import glob
import warnings
import argparse
import yaml
import torch
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torchaudio.transforms as AT
import torchvision.transforms as VT

warnings.filterwarnings('ignore')

# Add project root to Python path for custom module imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import build_model

# ==========================================
# 1. Data preparation and preprocessing utilities
# ==========================================
EXCLUDED_IDS = [16, 20, 23, 24, 25, 31]
SAMPLE_RATE_M4A = 96000
SAMPLE_RATE_CSV = 20000
DURATION = 1.0

# Load vibration signal from CSV file
def load_audio_csv(file_path):
    try:
        df = pd.read_csv(file_path, header=None)
        col_idx = 1 if df.shape[1] > 1 else 0
        return pd.to_numeric(df.iloc[:, col_idx], errors='coerce').dropna().values
    except: return None

# Load M4A audio file and convert to mono
def load_m4a_mono(file_path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(file_path, sr=None, mono=False)
        if y.ndim > 1: y = np.mean(y, axis=0)
        return y, sr
    except: return None, None

# Extract tap-centered signal segments based on peak detection
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

# Build metadata table linking audio, image, mass, and target Brix values
def create_multimodal_metadata(root_dir):
    """
    Construct metadata records linking audio samples,
    skin images, mass measurements, and target Brix values.
    """
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

# Dataset class for multimodal learning (audio + skin image + mass)
class MultimodalDataset(torch.utils.data.Dataset):
    """
    Multimodal dataset for watermelon sweetness prediction.

    Inputs:
        - Audio tapping signal (Mel spectrogram)
        - Skin image
        - Watermelon mass

    Target:
        - Brix value (sweetness level)
    """
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
# 2. Main evaluation pipeline
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(description="Scientific Evaluation for Paper")
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to trained model .pth file')
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation', help='Directory to save plots')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = config['model']['name'].upper()

    print("Preparing Validation Data...")
    df_full = create_multimodal_metadata(config['dataset']['root_dir'])
    if df_full.empty:
        print("❌ Dataset is empty. Check your root_dir path in the config file.")
        return

    # 1. Check if user specified exact validation IDs in the YAML config
    if 'fixed_val_ids' in config['dataset']:
        val_ids = config['dataset']['fixed_val_ids']
        print(f"🔒 Using fixed validation IDs from config: {val_ids}")
    else:
        # Fallback to random splitting if not specified
        unique_ids = df_full['id'].unique()
        _, val_ids = train_test_split(unique_ids, test_size=config['dataset']['val_split'], random_state=42)
        print(f"🎲 Using random validation IDs: {val_ids}")
        
    val_df = df_full[df_full['id'].isin(val_ids)]
    
    # Retrieve spectrogram resolution from config (required for architectures such as Swin Transformer and ConvNeXt)
    img_size_cfg = config['audio'].get('img_size', [128, 128])
    
    val_ds = MultimodalDataset(val_df, img_size=tuple(img_size_cfg))
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config['dataset']['batch_size'], shuffle=False)

    print(f"Loading {model_name} Model...")
    model = build_model(config['model']['name'], num_classes=config['model']['num_classes'])
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    print("Running Inference on Validation Set...")
    with torch.no_grad():
        for audio, skin, mass, targets in tqdm(val_loader):
            audio, skin, mass = audio.to(device), skin.to(device), mass.to(device)
            outputs = model(audio, skin, mass)
            
            y_pred.extend(outputs.cpu().numpy().flatten())
            y_true.extend(targets.cpu().numpy().flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    int_acc = np.mean(np.round(y_true) == np.round(y_pred)) * 100

    print("\n" + "="*40)
    print(f"📊 SCIENTIFIC RESULTS: {model_name}")
    print("="*40)
    print(f"Samples Evaluated : {len(y_true)}")
    print(f"MAE  (Mean Abs Err): {mae:.3f} Brix")
    print(f"RMSE (Root MSE)    : {rmse:.3f} Brix")
    print(f"R²   (R-Squared)   : {r2:.3f}")
    print(f"Integer Accuracy   : {int_acc:.1f}%")
    print("="*40)

    # Store ground-truth and predicted values
    with open(os.path.join(args.output_dir, "metrics_summary.txt"), "a") as f:
        f.write(f"{model_name} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f} | Acc: {int_acc:.1f}%\n")

    # Run inference on the validation set
    plt.figure(figsize=(8, 8))
    sns.set_theme(style="whitegrid")
    
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7, color='dodgerblue', edgecolor='k', s=80)
    
    min_val = min(min(y_true), min(y_pred)) - 0.5
    max_val = max(max(y_true), max(y_pred)) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal Fit (Y=X)')

    plt.title(f'{model_name}: Predicted vs Actual Brix', fontsize=16, fontweight='bold')
    plt.xlabel('Actual Brix (Ground Truth)', fontsize=14)
    plt.ylabel('Predicted Brix (Model Output)', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    textstr = '\n'.join((
        f'RMSE = {rmse:.2f}',
        f'$R^2$ = {r2:.2f}'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.85, textstr, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=props)

    plot_path = os.path.join(args.output_dir, f"{model_name.lower()}_scatter.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Scatter plot saved to {plot_path}")

if __name__ == "__main__":
    main()