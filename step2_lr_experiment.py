import pandas as pd
from torchvision import transforms
from utils import *

set_seed()
print(f"Running on device: {DEVICE}")
print("\n" + "="*70)
print("Step 2: Initial Learning Rate Experiment (15 epochs)")
print("="*70)

# Load precomputed statistics
stats_df = pd.read_csv(os.path.join(RESULT_DIR, "train_channel_stats.csv"))
train_mean = stats_df["mean"].values
train_std  = stats_df["std"].values

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

# Build datasets and dataloaders
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

# Run all LR experiments
lr_histories = {}
for lr in LR_CANDIDATES:
    print(f"\n--- Running experiment: learning rate = {lr} ---")
    set_seed()
    model = get_model(num_classes)

    history = run_training(
        model, train_loader, val_loader,
        num_epochs=LR_EXP_EPOCHS,
        lr=lr, weight_decay=0, use_cosine=False
    )
    lr_histories[lr] = history

    # Save standalone CSV
    exp_df = history_to_dataframe(history, "learning_rate", lr)
    csv_path = os.path.join(RESULT_DIR, f"lr_{lr}_results.csv")
    exp_df.to_csv(csv_path, index=False)

    # Save plot
    plot_curves(history, f"Learning Rate = {lr}",
                os.path.join(RESULT_DIR, f"lr_{lr}.png"))

# Select best LR by validation accuracy
best_lr = max(lr_histories.keys(), key=lambda x: lr_histories[x]["val_acc"][-1])
save_json({"best_lr": best_lr}, os.path.join(RESULT_DIR, "best_lr.json"))

# Summary
print("\n" + "-"*70)
print("Final Results: Initial Learning Rate Experiment")
print("-"*70)
print(f"{'LR':<10} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
for lr, hist in lr_histories.items():
    print(f"{lr:<10} {hist['train_loss'][-1]:<12.4f} {hist['train_acc'][-1]:<12.4f} {hist['val_loss'][-1]:<12.4f} {hist['val_acc'][-1]:<12.4f}")
print(f"\nBest initial learning rate: {best_lr}")
