# ml/extractors.py
import torch
import torchaudio.transforms as T


class AudioFeatureExtractor:
    def __init__(
        self,
        sample_rate: int  = 16000,
        n_fft:       int  = 1024,
        hop_length:  int  = 256,
        n_mels:      int  = 128,
        n_mfcc:      int  = 40,
    ):
        self.mel_transform = T.MelSpectrogram(
            sample_rate = sample_rate,
            n_fft       = n_fft,
            hop_length  = hop_length,
            n_mels      = n_mels,
            f_min       = 0.0,
            f_max       = 8000.0,
            power       = 2.0,
        )
        self.to_db = T.AmplitudeToDB(stype="power", top_db=80.0)
        
        self.mfcc_transform = T.MFCC(
            sample_rate = sample_rate,
            n_mfcc      = n_mfcc,
            melkwargs   = {
                "n_fft":      n_fft,
                "hop_length": hop_length,
                "n_mels":     n_mels,
            },
        )

    def get_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Returns (1, 128, time_frames)"""
        mel = self.mel_transform(waveform)
        mel = self.to_db(mel)
        return mel

    def get_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """Returns (1, 40, time_frames)"""
        return self.mfcc_transform(waveform)

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        mean = features.mean()
        std  = features.std() + 1e-8
        return (features - mean) / std