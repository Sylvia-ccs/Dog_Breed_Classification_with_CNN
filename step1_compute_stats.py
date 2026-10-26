import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import *

set_seed()
print(f"Running on device: {DEVICE}")
print("\n" + "="*70)
print("Step 1: Compute Training Set Channel Statistics")
print("="*70)

# Only resize + ToTensor — no augmentation, no normalization
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

train_set = DogBreedDataset(TRAIN_IMG_DIR, TRAIN_LABELS, transform=preprocess)
loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)

channel_sum = torch.zeros(3)
channel_sq_sum = torch.zeros(3)
total_pixels = 0

print("Processing training images to compute mean and std...")
for imgs, _ in loader:
    batch_pixels = imgs.size(0) * imgs.size(2) * imgs.size(3)
    channel_sum += imgs.sum(dim=[0, 2, 3])
    channel_sq_sum += (imgs ** 2).sum(dim=[0, 2, 3])
    total_pixels += batch_pixels

train_mean = channel_sum / total_pixels
train_std = torch.sqrt(channel_sq_sum / total_pixels - train_mean ** 2)
train_mean = train_mean.numpy()
train_std = train_std.numpy()

# Save as CSV
stats_df = pd.DataFrame({
    "channel": ["R", "G", "B"],
    "mean": train_mean,
    "std": train_std
})
stats_df.to_csv(os.path.join(RESULT_DIR, "train_channel_stats.csv"), index=False)

print("\n=== Training Set Channel Statistics ===")
print(f"Mean (R, G, B): {train_mean.round(4)}")
print(f"Std  (R, G, B): {train_std.round(4)}")
print(f"\nResults saved to {RESULT_DIR}/train_channel_stats.csv")
