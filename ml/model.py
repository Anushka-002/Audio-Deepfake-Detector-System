# ml/model.py
"""
CNN model for audio deepfake detection.

Input:  (batch, 1, 128, 251)  ← mel spectrogram
Output: (batch, 2)            ← logits for [REAL, FAKE]

Think of it like image classification — we treat the
spectrogram exactly like a grayscale photo.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU → MaxPool — the basic building block."""

    def __init__(self, in_ch: int, out_ch: int, pool: tuple = (2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x):
        return self.block(x)


class DeepfakeCNN(nn.Module):
    """
    4-layer CNN for deepfake detection.

    Architecture:
        Input (1, 128, 251)
          ↓ ConvBlock(1→32)    MaxPool(2,2) → (32, 64, 125)
          ↓ ConvBlock(32→64)   MaxPool(2,2) → (64, 32, 62)
          ↓ ConvBlock(64→128)  MaxPool(2,2) → (128, 16, 31)
          ↓ ConvBlock(128→256) MaxPool(2,2) → (256, 8, 15)
          ↓ AdaptiveAvgPool   → (256, 1, 1)
          ↓ Flatten           → (256,)
          ↓ FC + Dropout      → (128,)
          ↓ FC                → (2,)   ← [real_score, fake_score]
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
        )

        # AdaptiveAvgPool: no matter what size comes in, output is (256,1,1)
        # This makes the model robust to slightly different input sizes
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x   # raw logits — loss function applies softmax internally


def get_model(device: str = "cpu") -> DeepfakeCNN:
    model = DeepfakeCNN()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: DeepfakeCNN  |  Parameters: {total_params:,}")
    return model