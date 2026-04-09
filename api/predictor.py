import torch
import numpy as np
import torchaudio
import torchaudio.transforms as T
import io
import time
from pathlib import Path
from typing import Dict, Any
import sys

# Ensure the app can see the ml folder
sys.path.append("..") 

# Import the specific names from YOUR files
from ml.model import DeepfakeDetectorCNN
from ml.preprocessing import preprocess_audio
from ml.extractors import AudioFeatureExtractor 

# ── CONFIG (Must match your configs/config.yaml) ──────────────
SAMPLE_RATE  = 16000
N_SAMPLES    = 64000   # Changed to 4 seconds to match Phase 6
N_FFT        = 1024
HOP_LENGTH   = 256
N_MELS       = 128     # Changed to 128 to match your CNN input
TOP_DB       = 80

class DeepfakePredictor:
    def __init__(self, checkpoint_path: str = None, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[Predictor] Loading model on {self.device}...")
        
        # 1. Initialize the CNN we built
        self.model = DeepfakeDetectorCNN()
        
        # 2. Load weights if they exist (Phase 9)
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            # Handle if ckpt is a state_dict or a full dict
            state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            self.threshold = ckpt.get("best_threshold", 0.5)
        else:
            self.threshold = 0.5
            print("[Predictor] WARNING: No checkpoint found. Using random weights.")

        self.model.to(self.device)
        self.model.eval()
        
        # 3. Build transforms
        self._build_transforms()
        print(f"[Predictor] Ready ✅")

    def _build_transforms(self):
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS
        ).to(self.device)
        self.db_transform = T.AmplitudeToDB(stype="power", top_db=TOP_DB).to(self.device)

    def _preprocess(self, audio_bytes: bytes) -> torch.Tensor:
        buffer = io.BytesIO(audio_bytes)
        waveform, sr = torchaudio.load(buffer)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.to(self.device)

        # Resample
        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE).to(self.device)
            waveform = resampler(waveform)

        # Normalize
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        # Fix length (Pad/Trim)
        if waveform.shape[1] > N_SAMPLES:
            waveform = waveform[:, :N_SAMPLES]
        else:
            pad_len = N_SAMPLES - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        # Mel spectrogram
        mel = self.mel_transform(waveform)
        mel_db = self.db_transform(mel)

        # Normalize features (Crucial for CNN)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

        return mel_db.unsqueeze(0)  # Shape: (1, 1, 128, 251)

    @torch.no_grad()
    def predict(self, audio_bytes: bytes) -> Dict[str, Any]:
        t0 = time.time()
        mel = self._preprocess(audio_bytes)

        # The CNN returns a single probability value
        prob = self.model(mel).item()
        
        is_fake = prob >= self.threshold
        inference_ms = (time.time() - t0) * 1000

        return {
            "label": "fake" if is_fake else "real",
            "fake_prob": round(prob, 4),
            "confidence": round(prob * 100 if is_fake else (1 - prob) * 100, 2),
            "inference_ms": round(inference_ms, 2)
        }