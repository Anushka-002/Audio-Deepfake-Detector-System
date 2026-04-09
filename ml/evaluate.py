# ml/evaluate.py
import sys
sys.path.insert(0, "ml")

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import numpy as np

from dataset import get_dataloaders
from model   import get_model

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "ml/checkpoints/best_model.pt"
DATA_DIR   = "data/raw"

print(f"Loading model from {CHECKPOINT}")
_, _, test_loader = get_dataloaders(DATA_DIR, batch_size=32)
model = get_model(DEVICE)

ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Model from epoch {ckpt['epoch']}  val_acc={ckpt['val_acc']:.4f}")

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(DEVICE)
        logits   = model(features)
        probs    = F.softmax(logits, dim=1)[:, 1]  # prob of FAKE

        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

print("\n── Test Results ─────────────────────────────────")
print(f"Accuracy  : {accuracy_score(all_labels, all_preds):.4f}")
print(f"F1 Score  : {f1_score(all_labels, all_preds):.4f}")
print(f"ROC-AUC   : {roc_auc_score(all_labels, all_probs):.4f}")
print("\nConfusion Matrix (rows=actual, cols=predicted):")
print("         Pred:REAL  Pred:FAKE")
cm = confusion_matrix(all_labels, all_preds)
print(f"Act:REAL    {cm[0][0]:6d}    {cm[0][1]:6d}")
print(f"Act:FAKE    {cm[1][0]:6d}    {cm[1][1]:6d}")
print("\nFull Report:")
print(classification_report(all_labels, all_preds, target_names=["REAL", "FAKE"]))