import pandas as pd
from torchvision import transforms
from utils import *

set_seed()
print(f"Running on device: {DEVICE}")
print("\n" + "="*70)
print("Step 4: Weight Decay Regularization Experiment (300 epochs)")
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

wd_histories = {}

for wd in WD_CANDIDATES:
    print(f"\n--- Running experiment: weight decay λ = {wd} ---")
    set_seed()
    model = get_model(num_classes)

    history = run_training(
        model, train_loader, val_loader,
        num_epochs=FULL_EXP_EPOCHS,
        lr=best_lr, weight_decay=wd, use_cosine=True
    )
    wd_histories[wd] = history

    # Save standalone CSV
    exp_df = history_to_dataframe(history, "weight_decay", wd)
    csv_path = os.path.join(RESULT_DIR, f"weight_decay_{wd}_results.csv")
    exp_df.to_csv(csv_path, index=False)

    # Save individual plot
    plot_curves(history, f"Weight Decay λ = {wd}",
                os.path.join(RESULT_DIR, f"wd_{wd}.png"))

# Overlaid comparison plot
plot_wd_comparison(wd_histories, os.path.join(RESULT_DIR, "wd_comparison.png"))

# Select best weight decay
best_wd = max(wd_histories.keys(), key=lambda x: wd_histories[x]["val_acc"][-1])
save_json({"best_wd": best_wd}, os.path.join(RESULT_DIR, "best_wd.json"))

# Summary
print("\n" + "-"*70)
print("Final Results: Weight Decay Experiment")
print("-"*70)
print(f"{'Weight Decay λ':<18} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
for wd, hist in wd_histories.items():
    print(f"{wd:<18} {hist['train_loss'][-1]:<12.4f} {hist['train_acc'][-1]:<12.4f} {hist['val_loss'][-1]:<12.4f} {hist['val_acc'][-1]:<12.4f}")
print(f"\nBest weight decay coefficient: λ = {best_wd}")
