import librosa
import torch
import numpy as np

# --- CONSTANTS ---
# Keeping these here ensures every file is exactly 4 seconds at 16kHz
SAMPLE_RATE = 16000
N_SAMPLES = 64000 

def preprocess_audio(file_path: str) -> torch.Tensor:
    """
    Full pipeline: load (via librosa) → mono → resample → normalize → pad/trim.
    This version avoids the Windows torchcodec DLL error entirely.
    """
    try:
        # 1. Load, Resample, and Mono in one step (The Librosa Magic)
        # This bypasses the broken C++ torchaudio drivers.
        y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        
        # 2. Convert to PyTorch Tensor
        waveform = torch.from_numpy(y).unsqueeze(0) 
        
        # 3. Normalize (Peak Normalization)
        # Ensures quiet and loud recordings are treated equally
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
            
        # 4. Pad or Trim (Center-based)
        # Ensures the model always gets exactly (1, 64000) samples
        n = waveform.shape[-1]
        if n > N_SAMPLES:
            # Trim from the center (better for speech)
            start = (n - N_SAMPLES) // 2
            waveform = waveform[:, start:start + N_SAMPLES]
        elif n < N_SAMPLES:
            # Pad with silence (zeros) at the end
            waveform = torch.nn.functional.pad(waveform, (0, N_SAMPLES - n))
            
        return waveform

    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        # Return a "Silent" tensor if the file is broken so training doesn't crash
        return torch.zeros(1, N_SAMPLES)