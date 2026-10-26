import os
import torch

# ===================== Dataset Paths =====================
TRAIN_IMG_DIR = "data/train/images"
VAL_IMG_DIR   = "data/val/images"
TEST_IMG_DIR  = "data/test/images"
TRAIN_LABELS  = "data/train_labels.csv"
VAL_LABELS    = "data/val_labels.csv"
TEST_LABELS   = "data/test_labels.csv"

# ===================== Assignment Hyperparameters =====================
IMG_SIZE    = 100
BATCH_SIZE  = 128
MOMENTUM    = 0.9
PADDING     = 12
FLIP_PROB   = 0.5
SEED        = 42

LR_CANDIDATES = [0.5, 0.05, 0.001]
WD_CANDIDATES = [1e-4, 5e-4]
MIXUP_ALPHA   = 0.2

LR_EXP_EPOCHS   = 15
FULL_EXP_EPOCHS = 300

# ===================== Performance Optimizations =====================
NUM_WORKERS = 4
PIN_MEMORY  = True
CUDNN_BENCH = True

# ===================== Output =====================
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ===================== Device =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
