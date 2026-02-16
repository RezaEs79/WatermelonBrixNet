import sys
import os

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)
# ----------------

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from data.dataset import create_metadata

# Set the visualization style
sns.set_style("whitegrid")

def add_labels(ax, max_height):
    """
    Add value labels on top of bars up to a specified maximum height.
    """
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            # Small, proportional offset for label placement
            offset = max_height * 0.02 
            ax.text(p.get_x() + p.get_width() / 2., height + offset, 
                    int(height), ha="center", va="bottom", fontweight='bold', fontsize=10)

def plot_stats(data_dir, output_path):
    print(f"Loading metadata from: {data_dir}")
    
    EXCLUDED_IDS = [16, 20, 23, 24, 25, 31]
    df_full = create_metadata(data_dir, excluded_ids=EXCLUDED_IDS)
    
    if df_full.empty:
        print("❌ No data found!")
        return

    # Remove duplicate samples
    df_unique = df_full.drop_duplicates(subset=['id']).copy()
    
    # Round sugar content (Brix) to the nearest integer
    df_unique['brix'] = df_unique['brix'].round().astype(int)

    total_wms = len(df_unique)
    print(f"✅ Total Unique Watermelons: {total_wms}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Watermelon Dataset Statistics (Total: {total_wms})', fontsize=16)

    # ==========================================
    # Plot 1: Sugar (Brix)
    # ==========================================
    # Use hue=x and legend=False to apply a color gradient without warnings
    sns.countplot(data=df_unique, x='brix', ax=axes[0], palette='viridis', hue='brix', legend=False)
    
    mean_brix = df_unique['brix'].mean()
    
    axes[0].set_title('Sugar Distribution (Brix)', fontsize=13)
    axes[0].set_xlabel('Brix Value')
    axes[0].set_ylabel('Count')

    # Compute the maximum bar height to adjust the plot limits
    max_h_brix = 0
    for p in axes[0].patches:
        if p.get_height() > max_h_brix: max_h_brix = p.get_height()
    
    # Add 15% headroom above the tallest bar
    axes[0].set_ylim(0, max_h_brix * 1.15) 
    add_labels(axes[0], max_h_brix)

    # --- Display the mean value in the corner of the plot ---
    # (0.95, 0.95) corresponds to the top-right corner in axes coordinates
    axes[0].text(0.95, 0.95, f'Mean Brix: {mean_brix:.2f}', 
                 transform=axes[0].transAxes, 
                 ha='right', va='top', 
                 fontsize=12, fontweight='bold', color='darkblue',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    # ==========================================
    # Plot 2: Redness Class
    # ==========================================
    sns.countplot(data=df_unique, x='redness', ax=axes[1], palette='Reds', hue='redness', legend=False)
    
    axes[1].set_title('Redness Class Distribution (1-5)', fontsize=13)
    axes[1].set_xlabel('Redness Grade')
    axes[1].set_ylabel('Count')

    # Compute bar height 
    max_h_red = 0
    for p in axes[1].patches:
        if p.get_height() > max_h_red: max_h_red = p.get_height()
        
    axes[1].set_ylim(0, max_h_red * 1.15)
    add_labels(axes[1], max_h_red)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\n📊 Stats Summary:")
    print(f"   Brix Mean: {mean_brix:.2f}")
    print(f"✅ Plot saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/processed_dataset")
    parser.add_argument("--output", type=str, default="watermelon_stats_final.png")
    args = parser.parse_args()
    
    plot_stats(args.data_dir, args.output)