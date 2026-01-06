import sys
import os

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)
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

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", category=FutureWarning)

from models import build_model # <--- FIXED IMPORT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 96000
DURATION = 1.0
AUDIO_SIZE = (128, 128)
IMG_SIZE = (224, 224)
EXCLUDED_IDS = [16, 20, 23, 24, 25, 31]

# (Helper functions load_audio, segment_taps, prepare_batch remain the same - keep them!)
def load_audio(file_path):
    try:
        if file_path.lower().endswith('.m4a'):
            y, sr = librosa.load(file_path, sr=None, mono=True)
            return y, sr
        elif file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path, header=None)
            col_idx = 1 if df.shape[1] > 1 else 0
            raw = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').dropna().values
            return raw, 20000
    except: return None, None

def segment_taps(signal, sr):
    if signal is None or len(signal) == 0: return []
    peaks, _ = find_peaks(np.abs(signal), height=np.max(np.abs(signal))*0.2, distance=int(0.4*sr))
    segments = []
    for p in peaks[:3]:
        win = int(DURATION * sr)
        start = max(0, p - int(0.1 * win))
        end = min(len(signal), start + win)
        seg = signal[start:end]
        if len(seg) < win: seg = np.pad(seg, (0, win - len(seg)), 'constant')
        segments.append(torch.from_numpy(seg).float())
    return segments

mel_transform = AT.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=4096, hop_length=1024, n_mels=128)
db_transform = AT.AmplitudeToDB()
audio_resize = VT.Resize(AUDIO_SIZE, antialias=True)
resampler_csv = AT.Resample(20000, SAMPLE_RATE)
img_transform = VT.Compose([VT.Resize(IMG_SIZE), VT.ToTensor(), VT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def prepare_batch(segments, source_sr, skin_tensor, mass_tensor):
    audio_tensors = []
    for seg in segments:
        if source_sr == 20000: seg = resampler_csv(seg)
        target_len = int(DURATION * SAMPLE_RATE)
        if len(seg) < target_len: seg = torch.nn.functional.pad(seg, (0, target_len - len(seg)))
        else: seg = seg[:target_len]
        spec = db_transform(mel_transform(seg))
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
        spec = audio_resize(spec.unsqueeze(0)).repeat(3, 1, 1)
        audio_tensors.append(spec)
    if not audio_tensors: return None, None, None
    audio_batch = torch.stack(audio_tensors).to(DEVICE)
    n = audio_batch.size(0)
    skin_batch = skin_tensor.repeat(n, 1, 1, 1).to(DEVICE)
    mass_batch = mass_tensor.repeat(n, 1).to(DEVICE)
    return audio_batch, skin_batch, mass_batch

def run_validation(data_dir, sugar_ckpt, red_ckpt, output_csv, sugar_arch, red_arch):
    print(f"Loading Models on {DEVICE}...")
    
    # --- FIXED: Use build_model ---
    sugar_model = build_model(sugar_arch, num_classes=1).to(DEVICE)
    sugar_model.load_state_dict(torch.load(sugar_ckpt, map_location=DEVICE))
    sugar_model.eval()
    
    red_model = build_model(red_arch, num_classes=5).to(DEVICE)
    red_model.load_state_dict(torch.load(red_ckpt, map_location=DEVICE))
    red_model.eval()
    
    folders = sorted([f for f in os.listdir(data_dir) if f.startswith('watermelon')])
    print(f"\nValidating on {len(folders)} folders...")
    
    results = []
    total_int_error_abs = 0
    exact_matches = 0
    total_samples = 0
    
    for folder in tqdm(folders):
        match = re.search(r'watermelon(\d+)', folder)
        if not match: continue
        wm_id = int(match.group(1))
        if wm_id in EXCLUDED_IDS: continue
        
        try:
            parts = folder.split('_')
            mass_kg = float(parts[1].replace('kg', ''))
            real_brix = float(parts[2].replace('Brix', ''))
        except: continue 
        
        full_path = os.path.join(data_dir, folder)
        real_redness = -1
        flesh_dirs = glob.glob(os.path.join(full_path, 'Flesh*'))
        if flesh_dirs:
            try: real_redness = int(os.path.basename(flesh_dirs[0]).split('_')[2])
            except: pass
            
        mass_tensor = torch.tensor([[mass_kg / 10.0]], dtype=torch.float32)
        skin_path = os.path.join(full_path, 'Skin')
        skin_tensor = torch.zeros(3, *IMG_SIZE)
        if os.path.exists(skin_path):
            imgs = glob.glob(os.path.join(skin_path, '*.*'))
            if imgs:
                try: skin_tensor = img_transform(Image.open(imgs[0]).convert('RGB'))
                except: pass
        
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
            final_brix = np.mean(brix_preds)
            pred_int_brix = int(round(final_brix))
            real_int_brix = int(real_brix)
            int_error = real_int_brix - pred_int_brix
            total_int_error_abs += abs(int_error)
            if int_error == 0: exact_matches += 1
            total_samples += 1
            
            final_redness = np.argmax(np.mean(red_probs, axis=0)) + 1
            
            results.append({
                'ID': wm_id,
                'Real Brix': real_brix, 'Pred Brix': round(final_brix, 2), 'Pred Int': pred_int_brix,
                'Int Error': int_error, 'Real Redness': real_redness, 'Pred Redness': final_redness
            })
            
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print("\n" + "="*40)
    print("VALIDATION RESULTS")
    print("="*40)
    if total_samples > 0:
        print(f"Exact Integer Matches:   {exact_matches} ({exact_matches/total_samples*100:.1f}%)")
        print(f"Total Abs Int Error:     {total_int_error_abs}")
        print(f"Mean Abs Int Error:      {total_int_error_abs/total_samples:.2f}")
    else:
        print("No samples validated.")
    print(f"Results saved to:        {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/processed_dataset")
    parser.add_argument("--sugar_ckpt", type=str, required=True)
    parser.add_argument("--red_ckpt", type=str, required=True)
    parser.add_argument("--sugar_arch", type=str, default="gated", help="Arch name: gated/fusion")
    parser.add_argument("--red_arch", type=str, default="gated", help="Arch name: gated/fusion")
    parser.add_argument("--output", type=str, default="./outputs/validation_report.csv")
    args = parser.parse_args()
    
    run_validation(args.data_dir, args.sugar_ckpt, args.red_ckpt, args.output, args.sugar_arch, args.red_arch)