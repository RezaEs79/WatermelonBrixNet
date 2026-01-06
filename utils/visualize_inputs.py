import argparse
import os
import glob
import re
import numpy as np
import warnings

# --- 1. Suppress Specific Warnings ---
warnings.filterwarnings('ignore')
# هشدار مربوط به ناتوانی در لود اولیه و تلاش دوم
warnings.filterwarnings("ignore", message="PySoundFile failed")
# هشدارهای مربوط به نسخه‌های آینده کتابخانه
warnings.filterwarnings("ignore", category=FutureWarning)

# --- CRITICAL FOR WSL: Use non-interactive backend ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import librosa
import torch
import torchaudio.transforms as AT

# *** REUSE: Importing logic from dataset module ***
from data.dataset import load_audio, segment_tap_centered, get_audio_transforms

# Config
SAMPLE_RATE = 96000
DURATION = 1.0 

def find_watermelon_folder(root_dir, wm_id):
    """Find folder path for a given watermelon ID."""
    folders = sorted([f for f in os.listdir(root_dir) if f.startswith('watermelon')])
    for folder in folders:
        match = re.search(r'watermelon(\d+)', folder)
        if match and int(match.group(1)) == int(wm_id):
            return os.path.join(root_dir, folder)
    return None

def visualize_matplotlib_save(data_dir, wm_id, output_path):
    # 1. Setup Logic
    folder_path = find_watermelon_folder(data_dir, wm_id)
    if not folder_path:
        print(f"❌ Error: Watermelon ID {wm_id} not found.")
        return

    folder_name = os.path.basename(folder_path)
    print(f"🔍 Analyzing: {folder_name}")
    
    files = glob.glob(os.path.join(folder_path, '*.m4a')) + glob.glob(os.path.join(folder_path, '*.csv'))
    if not files:
        print("❌ No audio files found.")
        return

    # Get Shared Transforms
    mel_transform, db_transform = get_audio_transforms(SAMPLE_RATE)

    # 2. Process Data
    all_data = []
    for f_path in files:
        fname = os.path.basename(f_path)
        
        # Load audio (Warnings are now suppressed)
        sig, sr = load_audio(f_path, target_sr=SAMPLE_RATE)
        
        if sig is None: continue
            
        taps = segment_tap_centered(sig, sr, window_duration=DURATION)
        
        if taps:
            tap_data = []
            for t in taps:
                # Spectrogram
                spec = db_transform(mel_transform(t))
                spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
                
                tap_data.append({
                    'waveform': t.numpy(),
                    'spectrogram': spec.squeeze().numpy()
                })
            
            all_data.append({'filename': fname, 'taps': tap_data})
            print(f"   ✅ Processed: {fname} ({len(taps)} taps)")

    if not all_data: return

    # 3. Plotting with Matplotlib
    num_files = len(all_data)
    max_taps = 3
    
    # Create figure
    fig, axes = plt.subplots(num_files * 2, max_taps, figsize=(15, 3.5 * num_files))
    fig.suptitle(f"Input Analysis: {folder_name}\n(Odd Rows: Waveform | Even Rows: Spectrogram)", fontsize=16, y=0.99)

    # Handle single file case where axes might not be 2D array
    if num_files * 2 == 1: axes = np.array([axes])
    if len(axes.shape) == 1: axes = axes.reshape(-1, max_taps)

    for i, file_info in enumerate(all_data):
        row_wave = i * 2
        row_spec = (i * 2) + 1
        
        fname = file_info['filename']
        taps = file_info['taps']

        # Label the file on the left
        axes[row_wave, 0].set_ylabel(f"{fname}\nAmplitude", fontsize=9, fontweight='bold')
        axes[row_spec, 0].set_ylabel("Mel Freq", fontsize=9)

        for j in range(max_taps):
            ax_w = axes[row_wave, j]
            ax_s = axes[row_spec, j]

            if j < len(taps):
                # Plot Waveform
                ax_w.plot(taps[j]['waveform'], color='#1f77b4', linewidth=0.8)
                ax_w.set_title(f"Tap {j+1}", fontsize=10)
                ax_w.grid(True, alpha=0.3)
                ax_w.set_xticklabels([]) # Clean look
                
                # Plot Spectrogram
                im = ax_s.imshow(taps[j]['spectrogram'], aspect='auto', origin='lower', cmap='magma')
                if i == num_files - 1: ax_s.set_xlabel("Time")
            else:
                # Hide empty plots
                ax_w.axis('off')
                ax_s.axis('off')

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    
    # Save instead of show
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    
    print(f"\n✅ Visualization saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize inputs and save to image.")
    parser.add_argument("--id", type=int, required=True, help="Watermelon ID")
    parser.add_argument("--data_dir", type=str, default="./data/processed_dataset")
    parser.add_argument("--output", type=str, default=None, help="Custom output filename (optional)")
    args = parser.parse_args()
    
    # تعیین نام فایل خروجی به صورت پویا اگر کاربر نامی نداده باشد
    if args.output is None:
        final_output = f"input_analysis_WM{args.id}.png"
    else:
        final_output = args.output
    
    visualize_matplotlib_save(args.data_dir, args.id, final_output)