# import sys
# import os

# # --- PATH FIX ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# if project_root not in sys.path: sys.path.append(project_root)
# # ----------------

# import glob
# import re
# import argparse
# import numpy as np
# import pandas as pd
# import librosa
# from PIL import Image
# from scipy.signal import find_peaks
# from tqdm import tqdm
# import warnings
# import torch
# import torchaudio.transforms as AT
# import torchvision.transforms as VT
# from models import build_model # <--- FIXED IMPORT

# warnings.filterwarnings('ignore')
# warnings.filterwarnings("ignore", message="PySoundFile failed")
# warnings.filterwarnings("ignore", category=FutureWarning)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SAMPLE_RATE = 96000
# DURATION = 1.0
# AUDIO_SIZE = (128, 128)
# IMG_SIZE = (224, 224)

# # (Helper functions same as valid.py)
# def load_audio(file_path):
#     try:
#         if file_path.lower().endswith('.m4a'): return librosa.load(file_path, sr=None, mono=True)
#         elif file_path.lower().endswith('.csv'):
#             df = pd.read_csv(file_path, header=None)
#             col_idx = 1 if df.shape[1] > 1 else 0
#             raw = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').dropna().values
#             return raw, 20000
#     except: return None, None

# def segment_taps(signal, sr):
#     if signal is None or len(signal) == 0: return []
#     peaks, _ = find_peaks(np.abs(signal), height=np.max(np.abs(signal))*0.2, distance=int(0.4*sr))
#     segments = []
#     for p in peaks[:3]:
#         win = int(DURATION * sr)
#         start = max(0, p - int(0.1 * win))
#         end = min(len(signal), start + win)
#         seg = signal[start:end]
#         if len(seg) < win: seg = np.pad(seg, (0, win - len(seg)), 'constant')
#         segments.append(torch.from_numpy(seg).float())
#     return segments

# mel_transform = AT.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=4096, hop_length=1024, n_mels=128)
# db_transform = AT.AmplitudeToDB()
# audio_resize = VT.Resize(AUDIO_SIZE, antialias=True)
# resampler_csv = AT.Resample(20000, SAMPLE_RATE)
# img_transform = VT.Compose([VT.Resize(IMG_SIZE), VT.ToTensor(), VT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# def prepare_batch(segments, source_sr, skin_tensor, mass_tensor):
#     audio_tensors = []
#     for seg in segments:
#         if source_sr == 20000: seg = resampler_csv(seg)
#         target_len = int(DURATION * SAMPLE_RATE)
#         if len(seg) < target_len: seg = torch.nn.functional.pad(seg, (0, target_len - len(seg)))
#         else: seg = seg[:target_len]
#         spec = db_transform(mel_transform(seg))
#         spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
#         spec = audio_resize(spec.unsqueeze(0)).repeat(3, 1, 1)
#         audio_tensors.append(spec)
#     if not audio_tensors: return None, None, None
#     audio_batch = torch.stack(audio_tensors).to(DEVICE)
#     n = audio_batch.size(0)
#     skin_batch = skin_tensor.repeat(n, 1, 1, 1).to(DEVICE)
#     mass_batch = mass_tensor.repeat(n, 1).to(DEVICE)
#     return audio_batch, skin_batch, mass_batch

# def run_test(data_dir, sugar_ckpt, red_ckpt, output_csv, sugar_arch, red_arch):
#     print(f"Loading Models on {DEVICE}...")
#     sugar_model = build_model(sugar_arch, num_classes=1).to(DEVICE)
#     sugar_model.load_state_dict(torch.load(sugar_ckpt, map_location=DEVICE))
#     sugar_model.eval()
    
#     red_model = build_model(red_arch, num_classes=5).to(DEVICE)
#     red_model.load_state_dict(torch.load(red_ckpt, map_location=DEVICE))
#     red_model.eval()
    
#     folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
#     print(f"\nTesting on {len(folders)} folders...")
    
#     results = []
#     for folder in tqdm(folders):
#         full_path = os.path.join(data_dir, folder)
#         mass_kg = 0.0
#         try:
#             parts = folder.split('_')
#             for p in parts:
#                 if 'kg' in p: mass_kg = float(p.replace('kg', ''))
#         except: pass
        
#         mass_tensor = torch.tensor([[mass_kg / 10.0]], dtype=torch.float32)
#         skin_path = os.path.join(full_path, 'Skin')
#         skin_tensor = torch.zeros(3, *IMG_SIZE)
#         if os.path.exists(skin_path):
#             imgs = glob.glob(os.path.join(skin_path, '*.*'))
#             if imgs:
#                 try: skin_tensor = img_transform(Image.open(imgs[0]).convert('RGB'))
#                 except: pass
        
#         files = glob.glob(os.path.join(full_path, '*.m4a')) + glob.glob(os.path.join(full_path, '*.csv'))
#         brix_preds, red_probs = [], []
        
#         with torch.no_grad():
#             for f_path in files:
#                 sig, sr = load_audio(f_path)
#                 segments = segment_taps(sig, sr)
#                 aud, skn, mss = prepare_batch(segments, sr, skin_tensor, mass_tensor)
#                 if aud is not None:
#                     p_brix = sugar_model(aud, skn, mss)
#                     brix_preds.append(p_brix.mean().item())
#                     p_red = red_model(aud, skn, mss)
#                     red_probs.append(torch.softmax(p_red, dim=1).mean(dim=0).cpu().numpy())
        
#         if brix_preds:
#             final_brix = np.mean(brix_preds)
#             final_redness = np.argmax(np.mean(red_probs, axis=0)) + 1
#             results.append({'Folder Name': folder, 'Predicted Brix': round(final_brix, 2), 'Predicted Redness': final_redness})
            
#     pd.DataFrame(results).to_csv(output_csv, index=False)
#     print(f"\n✅ Results saved to: {output_csv}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_dir", type=str, required=True)
#     parser.add_argument("--sugar_ckpt", type=str, required=True)
#     parser.add_argument("--red_ckpt", type=str, required=True)
#     parser.add_argument("--sugar_arch", type=str, default="gated", help="Arch: gated/fusion")
#     parser.add_argument("--red_arch", type=str, default="gated", help="Arch: gated/fusion")
#     parser.add_argument("--output", type=str, default="./outputs/blind_test_results_new.csv")
#     args = parser.parse_args()
    
#     run_test(args.data_dir, args.sugar_ckpt, args.red_ckpt, args.output, args.sugar_arch, args.red_arch)


import sys
import os

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
# ----------------

import glob
import re
import argparse
import numpy as np
import pandas as pd
import librosa
from PIL import Image
from scipy.signal import find_peaks
from tqdm import tqdm
import warnings
import torch
import torchaudio.transforms as AT
import torchvision.transforms as VT
from models import build_model

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 96000
DURATION = 1.0
AUDIO_SIZE = (128, 128)
IMG_SIZE = (224, 224)

def load_audio(file_path):
    """Loads audio signal from either .m4a microphone files or .csv accelerometer/logger files."""
    try:
        if file_path.lower().endswith('.m4a'):
            return librosa.load(file_path, sr=None, mono=True)
        elif file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path, header=None)
            col_idx = 1 if df.shape[1] > 1 else 0
            raw = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').dropna().values
            return raw, 20000
    except Exception:
        return None, None

def segment_taps(signal, sr):
    """Detects up to 3 acoustic tap peaks and extracts fixed-duration windows around each tap."""
    if signal is None or len(signal) == 0:
        return []
    peaks, _ = find_peaks(np.abs(signal), height=np.max(np.abs(signal)) * 0.2, distance=int(0.4 * sr))
    segments = []
    for p in peaks[:3]:
        win = int(DURATION * sr)
        start = max(0, p - int(0.1 * win))
        end = min(len(signal), start + win)
        seg = signal[start:end]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)), 'constant')
        segments.append(torch.from_numpy(seg).float())
    return segments

mel_transform = AT.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=4096, hop_length=1024, n_mels=128)
db_transform = AT.AmplitudeToDB()
audio_resize = VT.Resize(AUDIO_SIZE, antialias=True)
resampler_csv = AT.Resample(20000, SAMPLE_RATE)
img_transform = VT.Compose([
    VT.Resize(IMG_SIZE),
    VT.ToTensor(),
    VT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def prepare_batch(segments, source_sr, skin_tensor, mass_tensor):
    """Converts segmented audio taps, skin image, and mass into normalized model-ready tensors."""
    audio_tensors = []
    for seg in segments:
        if source_sr == 20000:
            seg = resampler_csv(seg)
        target_len = int(DURATION * SAMPLE_RATE)
        if len(seg) < target_len:
            seg = torch.nn.functional.pad(seg, (0, target_len - len(seg)))
        else:
            seg = seg[:target_len]
        spec = db_transform(mel_transform(seg))
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
        spec = audio_resize(spec.unsqueeze(0)).repeat(3, 1, 1)
        audio_tensors.append(spec)
    if not audio_tensors:
        return None, None, None
    audio_batch = torch.stack(audio_tensors).to(DEVICE)
    n = audio_batch.size(0)
    skin_batch = skin_tensor.repeat(n, 1, 1, 1).to(DEVICE)
    mass_batch = mass_tensor.repeat(n, 1).to(DEVICE)
    return audio_batch, skin_batch, mass_batch

def run_test(data_dir, sugar_ckpt, red_ckpt, output_csv, sugar_arch, red_arch):
    """Runs multimodal inference across all folders and saves continuous and integer Brix predictions."""
    print(f"Loading Models on {DEVICE}...")
    sugar_model = build_model(sugar_arch, num_classes=1).to(DEVICE)
    sugar_model.load_state_dict(torch.load(sugar_ckpt, map_location=DEVICE))
    sugar_model.eval()
    
    red_model = build_model(red_arch, num_classes=5).to(DEVICE)
    red_model.load_state_dict(torch.load(red_ckpt, map_location=DEVICE))
    red_model.eval()
    
    folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    print(f"\nTesting on {len(folders)} folders...")
    
    results = []
    for folder in tqdm(folders):
        full_path = os.path.join(data_dir, folder)
        mass_kg = 0.0
        try:
            parts = folder.split('_')
            for p in parts:
                if 'kg' in p:
                    mass_kg = float(p.replace('kg', ''))
        except Exception:
            pass
        
        mass_tensor = torch.tensor([[mass_kg / 10.0]], dtype=torch.float32)
        skin_path = os.path.join(full_path, 'Skin')
        skin_tensor = torch.zeros(3, *IMG_SIZE)
        if os.path.exists(skin_path):
            imgs = glob.glob(os.path.join(skin_path, '*.*'))
            if imgs:
                try:
                    skin_tensor = img_transform(Image.open(imgs[0]).convert('RGB'))
                except Exception:
                    pass
        
        files = glob.glob(os.path.join(full_path, '*.m4a')) + glob.glob(os.path.join(full_path, '*.csv'))
        brix_preds, red_probs = [], []
        
        with torch.no_grad():
            for f_path in files:
                sig, sr = load_audio(f_path)
                segments = segment_taps(sig, sr)
                aud, skn, mss = prepare_batch(segments, sr, skin_tensor, mass_tensor)
                if aud is not None:
                    p_brix = sugar_model(aud, skn, mss)
                    brix_preds.append(p_brix.mean().item())
                    p_red = red_model(aud, skn, mss)
                    red_probs.append(torch.softmax(p_red, dim=1).mean(dim=0).cpu().numpy())
        
        if brix_preds:
            final_brix = float(np.mean(brix_preds))
            rounded_brix = int(np.rint(final_brix))
            final_redness = int(np.argmax(np.mean(red_probs, axis=0)) + 1)
            results.append({
                'Folder Name': folder,
                'Predicted Brix': round(final_brix, 2),
                'Predicted Brix (Int)': rounded_brix,
                'Predicted Redness': final_redness
            })
            
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--sugar_ckpt", type=str, required=True)
    parser.add_argument("--red_ckpt", type=str, required=True)
    parser.add_argument("--sugar_arch", type=str, default="gated", help="Arch: gated/fusion")
    parser.add_argument("--red_arch", type=str, default="gated", help="Arch: gated/fusion")
    parser.add_argument("--output", type=str, default="./outputs/blind_test_results_new.csv")
    args = parser.parse_args()
    
    run_test(args.data_dir, args.sugar_ckpt, args.red_ckpt, args.output, args.sugar_arch, args.red_arch)