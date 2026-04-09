# ml/inference.py
import sys
sys.path.insert(0, "ml")

import torch
import torch.nn.functional as F
from pathlib import Path

from preprocessing import preprocess_audio
from extractors    import AudioFeatureExtractor
from model         import get_model

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "ml/checkpoints/best_model.pt"

# Load once at module level (expensive — do it once, reuse forever)
_model     = None
_extractor = AudioFeatureExtractor()

def load_model():
    global _model
    if _model is None:
        _model = get_model(DEVICE)
        ckpt   = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
        _model.load_state_dict(ckpt["model_state"])
        _model.eval()
        print(f"Model loaded from checkpoint (epoch {ckpt['epoch']})")
    return _model


def predict(audio_path: str) -> dict:
    """
    Predict whether an audio file is REAL or FAKE.

    Returns:
        {
          "label":      "REAL" or "FAKE",
          "confidence": 0.0 to 1.0,
          "real_prob":  float,
          "fake_prob":  float,
        }
    """
    model = load_model()

    # Preprocess
    waveform = preprocess_audio(audio_path)
    features = _extractor.get_mel_spectrogram(waveform)
    features = _extractor.normalize(features)
    features = features.unsqueeze(0).to(DEVICE)   # add batch dim → (1,1,128,251)

    # Inference
    with torch.no_grad():
        logits = model(features)
        probs  = F.softmax(logits, dim=1)[0]

    real_prob = probs[0].item()
    fake_prob = probs[1].item()
    label     = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = max(real_prob, fake_prob)

    return {
        "label":      label,
        "confidence": round(confidence, 4),
        "real_prob":  round(real_prob, 4),
        "fake_prob":  round(fake_prob, 4),
    }


# Quick CLI test
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ml/inference.py path/to/audio.wav")
        sys.exit(1)
    result = predict(sys.argv[1])
    print(result)