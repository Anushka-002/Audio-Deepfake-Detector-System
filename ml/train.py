import sys
sys.path.insert(0, "ml")   # so imports inside ml/ work
import os
os.environ["TORCH_AUDIO_BACKEND"] = "soundfile"
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
from dataset  import get_dataloaders
from model    import get_model

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = "data/raw"
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR    = Path("ml/checkpoints")
SAVE_DIR.mkdir(exist_ok=True)

print(f"Training on: {DEVICE}")

# ── Data ──────────────────────────────────────────────────────────────────────
train_loader, val_loader, _ = get_dataloaders(DATA_DIR, BATCH_SIZE)

# ── Model ─────────────────────────────────────────────────────────────────────
model = get_model(DEVICE)

# ── Loss and Optimizer ────────────────────────────────────────────────────────
# CrossEntropyLoss:
#   - Works with raw logits (no softmax needed before)
#   - Combines log_softmax + negative log likelihood
#   - Perfect for binary/multi-class classification
criterion = nn.CrossEntropyLoss()

# Adam: adaptive learning rate optimizer — best default for deep learning
optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# LR scheduler: gradually reduces learning rate — helps model converge
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):

    # ── TRAIN phase ───────────────────────────────────────────────────────────
    model.train()   # enables Dropout + BatchNorm training behavior
    train_loss, train_correct, train_total = 0.0, 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [TRAIN]", leave=False)
    for features, labels in loop:
        features = features.to(DEVICE)   # move data to GPU/CPU
        labels   = labels.to(DEVICE)

        # Forward pass: model makes predictions
        logits = model(features)          # (batch, 2)

        # Compute loss: how wrong are the predictions?
        loss = criterion(logits, labels)

        # Backward pass: compute gradients
        optimizer.zero_grad()   # clear gradients from last step
        loss.backward()         # compute new gradients
        optimizer.step()        # update weights

        # Track metrics
        preds = logits.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total   += labels.size(0)
        train_loss    += loss.item()

        loop.set_postfix(loss=f"{loss.item():.4f}")

    train_acc  = train_correct / train_total
    train_loss = train_loss / len(train_loader)

    # ── VALIDATION phase ──────────────────────────────────────────────────────
    model.eval()    # disables Dropout, uses running stats for BatchNorm
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():   # no gradients needed during evaluation
        for features, labels in val_loader:
            features = features.to(DEVICE)
            labels   = labels.to(DEVICE)
            logits   = model(features)
            loss     = criterion(logits, labels)

            preds       = logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)
            val_loss    += loss.item()

    val_acc  = val_correct / val_total
    val_loss = val_loss / len(val_loader)

    scheduler.step()

    # ── Print epoch summary ───────────────────────────────────────────────────
    print(
        f"Epoch {epoch:02d}/{EPOCHS}  |  "
        f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f}  |  "
        f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f}"
    )

    # ── Save best model ───────────────────────────────────────────────────────
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = SAVE_DIR / "best_model.pt"
        torch.save({
            "epoch":      epoch,
            "model_state": model.state_dict(),
            "val_acc":    val_acc,
            "val_loss":   val_loss,
        }, save_path)
        print(f"  ✓ Saved best model  (val_acc={val_acc:.4f})")

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
print(f"Model saved to: ml/checkpoints/best_model.pt")