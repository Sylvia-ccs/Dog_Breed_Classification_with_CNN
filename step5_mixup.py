import pandas as pd
from torchvision import transforms
from utils import *

set_seed()
print(f"Running on device: {DEVICE}")
print("\n" + "="*70)
print("Step 5: Mixup Augmentation + Final Test Set Evaluation")
print("="*70)

# Load optimal hyperparameters
stats_df = pd.read_csv(os.path.join(RESULT_DIR, "train_channel_stats.csv"))
train_mean = stats_df["mean"].values
train_std  = stats_df["std"].values
best_lr = load_json(os.path.join(RESULT_DIR, "best_lr.json"))["best_lr"]
best_wd = load_json(os.path.join(RESULT_DIR, "best_wd.json"))["best_wd"]
print(f"Optimal config: lr = {best_lr}, weight_decay = {best_wd}")

# 1. Plot Beta distribution PDF
print("\nGenerating Beta distribution PDF plot...")
x = np.linspace(0, 1, 1000)
pdf = beta.pdf(x, MIXUP_ALPHA, MIXUP_ALPHA)
plt.figure(figsize=(6, 4))
plt.plot(x, pdf, "b-", linewidth=2)
plt.title(f"Beta Distribution PDF (α = β = {MIXUP_ALPHA})")
plt.xlabel("λ (mixup coefficient)")
plt.ylabel("Probability Density")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "beta_pdf.png"), dpi=300, bbox_inches="tight")
plt.close()

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Pad(PADDING, fill=0, padding_mode="constant"),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=FLIP_PROB),
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean, std=train_std)
])
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean, std=train_std)
])

# Build all datasets
train_dataset = DogBreedDataset(TRAIN_IMG_DIR, TRAIN_LABELS, transform=train_transform)
breed_to_idx = train_dataset.breed_to_idx
num_classes = train_dataset.num_classes

val_dataset  = DogBreedDataset(VAL_IMG_DIR, VAL_LABELS, transform=eval_transform, breed_to_idx=breed_to_idx)
test_dataset = DogBreedDataset(TEST_IMG_DIR, TEST_LABELS, transform=eval_transform, breed_to_idx=breed_to_idx)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)

# 2. Train with Mixup
print("\n--- Training with Mixup augmentation (300 epochs) ---")
set_seed()
model_mixup = get_model(num_classes)
hist_mixup = run_mixup_training(
    model_mixup, train_loader, val_loader,
    num_epochs=FULL_EXP_EPOCHS,
    lr=best_lr, weight_decay=best_wd, alpha=MIXUP_ALPHA
)

# Save mixup results
mixup_df = history_to_dataframe(hist_mixup, "experiment", "mixup_alpha_0.2")
mixup_df.to_csv(os.path.join(RESULT_DIR, "mixup_training_results.csv"), index=False)
plot_curves(hist_mixup, "Mixup Augmentation (α = 0.2)",
            os.path.join(RESULT_DIR, "mixup.png"))

# 3. Evaluate on hold-out test set
print("\n--- Evaluating final model on hold-out test set ---")
criterion = nn.CrossEntropyLoss()
test_loss, test_acc = val_epoch(model_mixup, test_loader, criterion)

test_result_df = pd.DataFrame({
    "metric": ["test_loss", "test_accuracy"],
    "value": [test_loss, test_acc]
})
test_result_df.to_csv(os.path.join(RESULT_DIR, "final_test_result.csv"), index=False)

# Final summary
print("\n" + "="*70)
print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
print("="*70)
print(f"Final Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"\nAll results saved to: {RESULT_DIR}/")
