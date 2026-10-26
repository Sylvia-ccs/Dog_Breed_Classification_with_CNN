import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta

from config import *


def set_seed():
    """Fix all random seeds for reproducible experiments."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = CUDNN_BENCH


# ==================================================
# Dataset
# ==================================================
class DogBreedDataset(Dataset):
    """Custom dataset for dog breed classification task."""
    def __init__(self, img_dir, labels_path, transform=None, breed_to_idx=None):
        self.img_dir = img_dir
        self.labels_df = pd.read_csv(labels_path).reset_index(drop=True)
        self.transform = transform

        if breed_to_idx is not None:
            self.breed_to_idx = breed_to_idx
        else:
            self.breed_to_idx = {
                breed: idx
                for idx, breed in enumerate(sorted(self.labels_df["breed"].unique()))
            }
        self.num_classes = len(self.breed_to_idx)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_id = self.labels_df.iloc[idx]["id"]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        label = self.breed_to_idx[self.labels_df.iloc[idx]["breed"]]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_model(num_classes):
    """Initialize EfficientNetV2-S with random weights (train from scratch)."""
    return models.efficientnet_v2_s(weights=None, num_classes=num_classes).to(DEVICE)


# ==================================================
# Core Training / Validation (with AMP optimization)
# ==================================================
def train_epoch(model, loader, criterion, optimizer, scaler=None):
    """Run one training epoch with optional automatic mixed precision."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = scaler is not None

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE, non_blocking=PIN_MEMORY)
        labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

    return total_loss / total, correct / total


def val_epoch(model, loader, criterion):
    """Run one evaluation epoch (full FP32 precision)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE, non_blocking=PIN_MEMORY)
            labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

    return total_loss / total, correct / total


def run_training(model, train_loader, val_loader, num_epochs, lr, weight_decay, use_cosine):
    """Full training pipeline with SGD and optional cosine annealing."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=MOMENTUM, weight_decay=weight_decay
    )
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
        if use_cosine else None
    )
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = val_epoch(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if scheduler:
            scheduler.step()
            print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")

    return history


# ==================================================
# Mixup Functions
# ==================================================
def mixup_data(x, y, alpha):
    """Generate mixed samples and labels via linear interpolation."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_train_epoch(model, loader, criterion, optimizer, alpha, scaler=None):
    """Train one epoch with mixup augmentation and optional AMP."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = scaler is not None

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE, non_blocking=PIN_MEMORY)
        labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)
        mixed_imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(mixed_imgs)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += lam * pred.eq(y_a).sum().item() + (1 - lam) * pred.eq(y_b).sum().item()

    return total_loss / total, correct / total


def run_mixup_training(model, train_loader, val_loader, num_epochs, lr, weight_decay, alpha):
    """Full training pipeline with mixup and cosine annealing."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=MOMENTUM, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = mixup_train_epoch(
            model, train_loader, criterion, optimizer, alpha, scaler
        )
        val_loss, val_acc = val_epoch(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        scheduler.step()

    return history


# ==================================================
# Plotting & CSV Export
# ==================================================
def plot_curves(history, title, save_path):
    """Plot paired loss and accuracy curves for a single experiment."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Classification Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_wd_comparison(histories, save_path):
    """Plot overlayed curves to compare weight decay effects."""
    epochs = range(1, FULL_EXP_EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#1f77b4', '#ff7f0e']

    ax1 = axes[0]
    for idx, (wd, hist) in enumerate(histories.items()):
        ax1.plot(epochs, hist["train_loss"], color=colors[idx], linestyle='-', label=f'Train (λ={wd})')
        ax1.plot(epochs, hist["val_loss"], color=colors[idx], linestyle='--', label=f'Val (λ={wd})')
    ax1.set_title("Loss Comparison")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    for idx, (wd, hist) in enumerate(histories.items()):
        ax2.plot(epochs, hist["train_acc"], color=colors[idx], linestyle='-', label=f'Train (λ={wd})')
        ax2.plot(epochs, hist["val_acc"], color=colors[idx], linestyle='--', label=f'Val (λ={wd})')
    ax2.set_title("Accuracy Comparison")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def history_to_dataframe(history, group_col_name, group_value):
    """Convert training history to long-format DataFrame for CSV export."""
    df = pd.DataFrame(history)
    df.insert(0, "epoch", range(1, len(df) + 1))
    df.insert(1, group_col_name, group_value)
    return df


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
