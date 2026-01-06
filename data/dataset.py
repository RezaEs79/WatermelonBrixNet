import os
import glob
import re
import numpy as np
import pandas as pd
import librosa
from PIL import Image
from scipy.signal import find_peaks

import torch
from torch.utils.data import Dataset
import torchaudio.transforms as AT
import torchvision.transforms as VT

# ==========================================
# 1. Helper Functions (Audio Processing)
# ==========================================

def load_audio(file_path, target_sr=96000):
    """
    Loads audio file (M4A or CSV) and converts to mono signal.
    """
    try:
        if file_path.lower().endswith('.m4a'):
            y, sr = librosa.load(file_path, sr=None, mono=True)
            return y, sr
        elif file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path, header=None)
            col_idx = 1 if df.shape[1] > 1 else 0
            raw_signal = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').dropna().values
            return raw_signal, 20000 
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def segment_tap_centered(signal, sample_rate, window_duration=1.0):
    """
    Detects impacts (taps) and extracts a fixed-length window.
    """
    if signal is None or len(signal) == 0: return []
    peaks, _ = find_peaks(np.abs(signal), height=np.max(np.abs(signal))*0.2, distance=int(0.4*sample_rate))
    segments = []
    for p in peaks[:3]:
        window_samples = int(window_duration * sample_rate)
        pre_peak = int(0.1 * window_samples)
        start = max(0, p - pre_peak)
        end = min(len(signal), start + window_samples)
        seg = signal[start:end]
        if len(seg) < window_samples:
            pad_amount = window_samples - len(seg)
            seg = np.pad(seg, (0, pad_amount), 'constant')
        segments.append(torch.from_numpy(seg).float())
    return segments

# --- NEW: Reusable Transform Function ---
def get_audio_transforms(sample_rate, n_fft=4096, hop_length=1024, n_mels=128):
    """
    Returns the standard transforms used for preprocessing audio across the project.
    """
    mel_transform = AT.MelSpectrogram(
        sample_rate=sample_rate, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=n_mels
    )
    db_transform = AT.AmplitudeToDB()
    return mel_transform, db_transform

# ==========================================
# 2. Metadata Builder
# ==========================================
# (این تابع create_metadata بدون تغییر باقی می‌ماند، برای کوتاهی اینجا تکرار نکردم)
def create_metadata(root_dir, excluded_ids=None):
    # ... (همان کد قبلی) ...
    # برای اینکه کد کامل اجرا شود، فرض می‌کنیم کد قبلی اینجاست.
    # اگر کل فایل را جایگزین می‌کنید، حتما کد create_metadata را از مرحله قبل اینجا بگذارید.
    if excluded_ids is None: excluded_ids = []
    data_records = []
    folders = sorted([f for f in os.listdir(root_dir) if f.startswith('watermelon')])
    for folder in folders:
        match = re.search(r'watermelon(\d+)', folder)
        if not match: continue
        wm_id = int(match.group(1))
        if wm_id in excluded_ids: continue
        full_path = os.path.join(root_dir, folder)
        try:
            parts = folder.split('_')
            mass = float(parts[1].replace('kg', ''))
            brix = float(parts[2].replace('Brix', ''))
        except: continue
        redness_grade = -1
        flesh_dirs = glob.glob(os.path.join(full_path, 'Flesh*'))
        if flesh_dirs:
            try: redness_grade = int(os.path.basename(flesh_dirs[0]).split('_')[2])
            except: pass
        if redness_grade == -1: continue
        skin_path = os.path.join(full_path, 'Skin')
        skin_img_file = None
        if os.path.exists(skin_path):
            imgs = [f for f in glob.glob(os.path.join(skin_path, '*.*')) if f.lower().endswith(('.jpg','.png','.jpeg'))]
            if imgs: skin_img_file = imgs[0]
        audio_files = glob.glob(os.path.join(full_path, '*.m4a')) + glob.glob(os.path.join(full_path, '*.csv'))
        for f_path in audio_files:
            fname = os.path.basename(f_path)
            ftype = 'M4A' if fname.endswith('.m4a') else 'CSV'
            for i in range(3):
                data_records.append({'id': wm_id, 'mass': mass, 'brix': brix, 'redness': redness_grade, 'folder_path': full_path, 'audio_file': fname, 'skin_file': skin_img_file, 'source': ftype, 'tap_idx': i})
    return pd.DataFrame(data_records)

# ==========================================
# 3. Main Dataset Class
# ==========================================
class WatermelonDataset(Dataset):
    def __init__(self, dataframe, config, weights_map=None, augment=False):
        self.df = dataframe
        self.config = config
        self.weights_map = weights_map
        self.augment = augment
        
        self.target_sr = config['audio']['sample_rate']
        self.duration = config['audio']['duration']
        # self.audio_size = (128, 128)
        # --- CHANGE: Read size from config, default to (128, 128) if not present ---
        img_size_cfg = config['audio'].get('img_size', [128, 128])
        self.audio_size = tuple(img_size_cfg)
        
        # --- REUSE: Using the shared function ---
        self.mel_transform, self.db_transform = get_audio_transforms(self.target_sr)
        
        self.audio_resize = VT.Resize(self.audio_size, antialias=True)
        self.freq_mask = AT.FrequencyMasking(30)
        self.time_mask = AT.TimeMasking(60)
        self.resampler_csv = AT.Resample(20000, self.target_sr)

        self.img_size = (224, 224)
        if augment:
            self.img_transform = VT.Compose([VT.Resize((256, 256)), VT.RandomCrop(self.img_size), VT.RandomHorizontalFlip(), VT.ColorJitter(brightness=0.1), VT.ToTensor(), VT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            self.img_transform = VT.Compose([VT.Resize(self.img_size), VT.ToTensor(), VT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(row['folder_path'], row['audio_file'])
        sig, sr = load_audio(file_path)
        if sig is None: sig, sr = np.zeros(self.target_sr), self.target_sr
            
        taps = segment_tap_centered(sig, sr, self.duration)
        tap_idx = int(row['tap_idx'])
        tensor = taps[tap_idx] if taps and tap_idx < len(taps) else (taps[0] if taps else torch.zeros(int(self.duration * sr)))
            
        if row['source'] == 'CSV': tensor = self.resampler_csv(tensor)
        target_len = int(self.duration * self.target_sr)
        if len(tensor) < target_len: tensor = torch.nn.functional.pad(tensor, (0, target_len - len(tensor)))
        else: tensor = tensor[:target_len]
            
        spec = self.db_transform(self.mel_transform(tensor))
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
        if self.augment:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)
        spec = self.audio_resize(spec.unsqueeze(0)).repeat(3, 1, 1)

        skin_path = row['skin_file']
        if skin_path and os.path.exists(skin_path):
            try: skin_tensor = self.img_transform(Image.open(skin_path).convert('RGB'))
            except: skin_tensor = torch.zeros(3, *self.img_size)
        else: skin_tensor = torch.zeros(3, *self.img_size)

        mass_val = float(row['mass'])
        mass_tensor = torch.tensor([mass_val / 10.0], dtype=torch.float32)
        target_brix = float(row['brix'])
        target_redness = int(row['redness']) - 1
        weight = 1.0
        if self.weights_map:
            weight = self.weights_map.get(int(round(target_brix)), 1.0)
            
        return {'audio': spec, 'skin': skin_tensor, 'mass': mass_tensor, 'brix': torch.tensor(target_brix, dtype=torch.float32), 'redness': torch.tensor(target_redness, dtype=torch.long), 'weight': torch.tensor(weight, dtype=torch.float32), 'id': row['id']}