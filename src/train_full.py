import sys
import os

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
# ----------------

import argparse
import yaml
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

from models import build_model
from data.dataset import create_metadata, WatermelonDataset

def compute_class_weights(df, task_type):
    """
    Computes inverse frequency weights per sample to handle class imbalance
    when training on the entire dataset.
    """
    weights_map = {}
    unique_melons = df.drop_duplicates(subset=['id'])
    
    if task_type == 'classification':
        counts = unique_melons['redness'].value_counts().to_dict()
        total = len(unique_melons)
        for _, row in unique_melons.iterrows():
            cls = row['redness']
            weights_map[row['id']] = total / (len(counts) * counts.get(cls, 1))
    else:
        for _, row in unique_melons.iterrows():
            weights_map[row['id']] = 1.0
            
    return weights_map

def main():
    parser = argparse.ArgumentParser(description="Train Model on 100% of Dataset (No Validation Split)")
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml file')
    parser.add_argument('--epochs', type=int, default=None, help='Fixed number of epochs (overrides config to prevent overfitting)')
    parser.add_argument('--save_name', type=str, default='best_full.pth', help='Filename for the saved checkpoint')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model']['name']
    num_classes = config['model'].get('num_classes', 1)
    task_type = 'classification' if num_classes > 1 else 'regression'

    # Determine number of epochs (vital when training without early stopping)
    train_cfg = config.get('train', config.get('training', {}))
    epochs = args.epochs if args.epochs is not None else train_cfg.get('epochs', 15)
    lr = float(train_cfg.get('lr', train_cfg.get('learning_rate', 1e-4)))
    weight_decay = float(train_cfg.get('weight_decay', 1e-4))
    batch_size = config['dataset'].get('batch_size', 16)

    print("=" * 60)
    print(f"🍉 FULL DATASET TRAINING MODE (100% DATA) | Task: {task_type.upper()}")
    print("=" * 60)

    df_full = create_metadata(
        config['dataset']['root_dir'],
        excluded_ids=config['dataset'].get('excluded_ids', [])
    )
    unique_ids = sorted(df_full['id'].unique())
    print(f"✅ Loaded {len(unique_ids)} unique watermelons ({len(df_full)} total tap samples).")
    print(f"👉 All Watermelon IDs included in training: {unique_ids}")

    weights_map = compute_class_weights(df_full, task_type)
    train_ds = WatermelonDataset(df_full, config, weights_map=weights_map, augment=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['dataset'].get('num_workers', 2),
        pin_memory=True
    )

    print(f"\n🔨 Building {model_name.upper()} (num_classes={num_classes})...")
    model = build_model(model_name, num_classes=num_classes)
    model.to(device)

    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.SmoothL1Loss()  # Huber Loss for robust Brix regression

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    exp_name = config.get('exp_name', f"{task_type.capitalize()}_{model_name}_Full")
    save_dir = os.path.join("checkpoints", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, args.save_name)

    best_train_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]", leave=False)
        for batch in pbar:
            audio = batch['audio'].to(device)
            skin = batch['skin'].to(device)
            mass = batch['mass'].to(device)

            optimizer.zero_grad()
            outputs = model(audio, skin, mass)

            if task_type == 'classification':
                targets = batch['redness'].to(device).long()
                loss = criterion(outputs, targets)
            else:
                target_key = 'brix' if 'brix' in batch else 'sugar'
                targets = batch[target_key].to(device).float()
                loss = criterion(outputs.squeeze(-1), targets.squeeze(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_sz = audio.size(0)
            running_loss += loss.item() * batch_sz
            total_samples += batch_sz
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        epoch_loss = running_loss / max(total_samples, 1)
        print(f"📈 Epoch [{epoch:02d}/{epochs:02d}] - Full-Data Train Loss: {epoch_loss:.4f}")

        # Save checkpoint at final epoch (or improved training stability)
        if epoch_loss <= best_train_loss or epoch == epochs:
            best_train_loss = min(best_train_loss, epoch_loss)
            torch.save(model.state_dict(), save_path)

    print("\n" + "=" * 60)
    print(f"🎉 Training Complete on 100% of Dataset!")
    print(f"💾 Final production weights saved to: {save_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()