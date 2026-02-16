import sys
import os

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)
# ----------------

import argparse
import yaml
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
from datetime import datetime

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", category=FutureWarning)

from data.dataset import WatermelonDataset, create_metadata
# Import factory function
from models import build_model 

def setup_logger(save_dir):
    log_file = os.path.join(save_dir, 'train_log.txt')
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

def load_config(config_path):
    with open(config_path, 'r') as f: return yaml.safe_load(f)

def calculate_weights(df, task_type):
    if task_type == 'regression': vals = df['brix'].round().astype(int)
    else: vals = df['redness'].astype(int)
    counts = vals.value_counts()
    weights = {k: 1.0 / (v + 1e-6) for k, v in counts.items()}
    mean_w = np.mean(list(weights.values()))
    return {k: (v / mean_w)**0.8 for k, v in weights.items()}

def train(config_path):
    config = load_config(config_path)
    base_save_dir = config['training']['save_dir']
    experiment_dir = os.path.join(base_save_dir, config['experiment_name'])
    os.makedirs(experiment_dir, exist_ok=True)
    
    logger = setup_logger(experiment_dir)
    start_time = datetime.now()
    logger.info(f"🚀 Start Time: {start_time}")
    logger.info(f"Experiment: {config['experiment_name']}")

    DEVICE = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

    logger.info("[1/4] Preparing Dataset...")
    root_dir = config['dataset']['root_dir']
    df_full = create_metadata(root_dir, excluded_ids=config['dataset']['excluded_ids'])
    
    use_weighted = config['training'].get('use_weighted_loss', True)
    if use_weighted:
        weights_map = calculate_weights(df_full, config['model']['type'])
        logger.info("⚖️ Class Balancing: ENABLED")
    else:
        weights_map = None
        logger.info("⚖️ Class Balancing: DISABLED")
    
    # Split
    # ---------------------------------------------------------
    # Modified section for manual selection of validation data
    # ---------------------------------------------------------
    unique_ids = df_full['id'].unique()
    
    # 1. Check whether a fixed validation list is provided in the config
    fixed_val_ids = config['dataset'].get('fixed_val_ids', None)
    
    if fixed_val_ids and len(fixed_val_ids) > 0:
        # Manual mode
        val_ids = np.array(fixed_val_ids)
        # Everything not in validation goes to training
        train_ids = np.array([uid for uid in unique_ids if uid not in val_ids])
        logger.info(f"📌 Using FIXED Validation Set (From Config)")
    else:
        # Automatic mode (Random Split)
        val_size = int(len(unique_ids) * config['dataset']['val_split'])
        train_size = len(unique_ids) - val_size
        np.random.seed(41)
        np.random.shuffle(unique_ids)
        train_ids, val_ids = unique_ids[:train_size], unique_ids[train_size:]
        logger.info(f"🎲 Using RANDOM Validation Set (Split: {config['dataset']['val_split']})")
    
    # Display dataset statistics
    logger.info(f"Train IDs: {len(train_ids)} | Val IDs: {len(val_ids)}")
    clean_val_ids = sorted([int(x) for x in val_ids])
    logger.info(f"👉 Validation IDs: {clean_val_ids}")
    # ---------------------------------------------------------    
    train_df = df_full[df_full['id'].isin(train_ids)].copy()
    val_df = df_full[df_full['id'].isin(val_ids)].copy()
    
    train_ds = WatermelonDataset(train_df, config, weights_map=weights_map, augment=True)
    val_ds = WatermelonDataset(val_df, config, weights_map=None, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=config['dataset']['batch_size'], shuffle=True, num_workers=config['dataset']['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=config['dataset']['batch_size'], shuffle=False, num_workers=config['dataset']['num_workers'])

    logger.info("[2/4] Initializing Model...")
    task_type = config['model']['type']
    model_name = config['model']['name']
    
    if task_type == 'regression':
        model = build_model(model_name, num_classes=1).to(DEVICE)
        criterion = nn.MSELoss(reduction='none')
        best_metric = float('inf')
        metric_name = "MAE"
    else:
        model = build_model(model_name, num_classes=5).to(DEVICE)
        criterion = nn.CrossEntropyLoss(reduction='none')
        best_metric = 0.0
        metric_name = "Accuracy"
        
    logger.info(f"   Architecture: {model_name}")
    logger.info(f"   Task: {task_type}")

    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    logger.info("[3/4] Starting Training Loop...")
    num_epochs = config['training']['epochs']
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_metric = 0.0, 0.0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{num_epochs}", leave=False)
        
        for batch in loop:
            audio, skin, mass = batch['audio'].to(DEVICE), batch['skin'].to(DEVICE), batch['mass'].to(DEVICE)
            weights = batch['weight'].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(audio, skin, mass)
            
            if task_type == 'regression':
                target = batch['brix'].to(DEVICE).unsqueeze(1)
                loss = ((outputs - target) ** 2 * weights.unsqueeze(1)).mean()
                train_metric += torch.abs(torch.round(outputs) - target).sum().item()
            else:
                target = batch['redness'].to(DEVICE)
                loss = (criterion(outputs, target) * weights).mean()
                train_metric += (torch.max(outputs, 1)[1] == target).sum().item()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        model.eval()
        val_loss, val_metric = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                audio, skin, mass = batch['audio'].to(DEVICE), batch['skin'].to(DEVICE), batch['mass'].to(DEVICE)
                outputs = model(audio, skin, mass)
                if task_type == 'regression':
                    target = batch['brix'].to(DEVICE).unsqueeze(1)
                    val_loss += nn.MSELoss()(outputs, target).item()
                    val_metric += torch.abs(torch.round(outputs) - target).sum().item()
                else:
                    target = batch['redness'].to(DEVICE)
                    val_loss += nn.CrossEntropyLoss()(outputs, target).item()
                    val_metric += (torch.max(outputs, 1)[1] == target).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if task_type == 'regression':
            avg_train_m = train_metric / len(train_ds)
            avg_val_m = val_metric / len(val_ds)
            is_best = avg_val_m < best_metric
            metric_str = f"Train MAE: {avg_train_m:.2f} | Val MAE: {avg_val_m:.2f}"
        else:
            avg_train_m = train_metric / len(train_ds)
            avg_val_m = val_metric / len(val_ds)
            is_best = avg_val_m > best_metric
            metric_str = f"Train Acc: {avg_train_m*100:.1f}% | Val Acc: {avg_val_m*100:.1f}%"

        scheduler.step(avg_val_loss)
        logger.info(f"Ep {epoch+1:02d}: Loss {avg_train_loss:.4f}->{avg_val_loss:.4f} | {metric_str}")
        
        torch.save(model.state_dict(), os.path.join(experiment_dir, "last.pth"))
        if is_best:
            best_metric = avg_val_m
            torch.save(model.state_dict(), os.path.join(experiment_dir, "best.pth"))
            logger.info(f"   🏆 Best Saved! ({best_metric:.2f})")

    end_time = datetime.now()
    logger.info(f"\n✅ Training Complete. Best {metric_name}: {best_metric:.2f}")
    logger.info(f"Total Duration: {end_time - start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)