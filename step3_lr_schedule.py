import pandas as pd
from torchvision import transforms
from utils import *

set_seed()
print(f"Running on device: {DEVICE}")
print("\n" + "="*70)
print("Step 3: Learning Rate Schedule Comparison (300 epochs)")
print("="*70)

# Load previous results
stats_df = pd.read_csv(os.path.join(RESULT_DIR, "train_channel_stats.csv"))
train_mean = stats_df["mean"].values
train_std  = stats_df["std"].values
best_lr = load_json(os.path.join(RESULT_DIR, "best_lr.json"))["best_lr"]
print(f"Using best initial LR: {best_lr}")

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Pad(PADDING, fill=0, padding_mode="constant"),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=FLIP_PROB),
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean, std=train_std)
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean, std=train_std)
])

# Build datasets
train_dataset = DogBreedDataset(TRAIN_IMG_DIR, TRAIN_LABELS, transform=train_transform)
breed_to_idx = train_dataset.breed_to_idx
num_classes = train_dataset.num_classes
val_dataset = DogBreedDataset(VAL_IMG_DIR, VAL_LABELS, transform=val_transform, breed_to_idx=breed_to_idx)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)

schedule_histories = {}

# --- Experiment 1: Constant LR ---
print("\n--- Experiment 1: Constant Learning Rate ---")
set_seed()
model_const = get_model(num_classes)
hist_const = run_training(
    model_const, train_loader, val_loader,
    num_epochs=FULL_EXP_EPOCHS,
    lr=best_lr, weight_decay=0, use_cosine=False
)
schedule_histories["constant"] = hist_const

const_df = history_to_dataframe(hist_const, "schedule", "constant")
const_df.to_csv(os.path.join(RESULT_DIR, "constant_lr_results.csv"), index=False)
plot_curves(hist_const, "Constant Learning Rate",
            os.path.join(RESULT_DIR, "constant_lr.png"))

# --- Experiment 2: Cosine Annealing ---
print("\n--- Experiment 2: Cosine Annealing Learning Rate ---")
set_seed()
model_cos = get_model(num_classes)
hist_cos = run_training(
    model_cos, train_loader, val_loader,
    num_epochs=FULL_EXP_EPOCHS,
    lr=best_lr, weight_decay=0, use_cosine=True
)
schedule_histories["cosine"] = hist_cos

cos_df = history_to_dataframe(hist_cos, "schedule", "cosine")
cos_df.to_csv(os.path.join(RESULT_DIR, "cosine_lr_results.csv"), index=False)
plot_curves(hist_cos, "Cosine Annealing Learning Rate",
            os.path.join(RESULT_DIR, "cosine_lr.png"))

# Summary
print("\n" + "-"*70)
print("Final Results: Learning Rate Schedule Comparison")
print("-"*70)
print(f"{'Schedule':<15} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
for name, hist in schedule_histories.items():
    print(f"{name:<15} {hist['train_loss'][-1]:<12.4f} {hist['train_acc'][-1]:<12.4f} {hist['val_loss'][-1]:<12.4f} {hist['val_acc'][-1]:<12.4f}")
