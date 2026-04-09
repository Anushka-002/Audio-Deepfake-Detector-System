import os
os.environ["TORCH_AUDIO_BACKEND"] = "soundfile"
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple
import random
import sys
sys.path.insert(0, "ml")

from preprocessing import preprocess_audio
from extractors    import AudioFeatureExtractor


class WaveFakeDataset(Dataset):
    """Labels: 0 = REAL,  1 = FAKE"""

    def __init__(self, data_dir: str, split: str = "train",
                 val_split: float = 0.15, test_split: float = 0.10,
                 seed: int = 42):
        super().__init__()
        self.extractor = AudioFeatureExtractor()
        self.samples   = self._build_samples(data_dir)

        if len(self.samples) == 0:
            raise ValueError(f"No audio files found in {data_dir}")

        self.samples = self._split(self.samples, split, val_split, test_split, seed)

        real = sum(1 for _, l in self.samples if l == 0)
        fake = sum(1 for _, l in self.samples if l == 1)
        print(f"  [{split:5s}]  total={len(self.samples):,}  "
              f"real={real:,}  fake={fake:,}")

    def _build_samples(self, data_dir: str) -> List[Tuple[Path, int]]:
        root    = Path(data_dir)
        samples = []

        # ── REAL audio — LJSpeech ────────────────────────────────────────────
        real_dir = root / "LJSpeech-1.1" / "wavs"
        if real_dir.exists():
            wavs = sorted(real_dir.glob("*.wav"))
            print(f"  Real audio : {len(wavs):,} files  ({real_dir})")
            for f in wavs:
                samples.append((f, 0))
        else:
            print(f"  [!] Real audio NOT found at {real_dir}")
            print(f"      Run Step 4 to download LJSpeech")

        # ── FAKE audio — all folders you have ────────────────────────────────
        # These are the exact folder names visible in your screenshot
        fake_dir_names = [
            "ljspeech_melgan",
            "ljspeech_melgan_large",
            "ljspeech_multi_band_melgan",
            "ljspeech_parallel_wavegan",
            "ljspeech_full_band_melgan",
            "ljspeech_hifiGAN",
            "ljspeech_waveglow",
            "jsut_multi_band_melgan",
            "jsut_parallel_wavegan",
            "common_voices_prompts_from_conform",  # adjust if name is different
        ]

        # Also scan generated_audio subfolder if it exists
        generated = root / "generated_audio"
        search_roots = [root]
        if generated.exists():
            search_roots.append(generated)

        found_fakes = 0
        for search_root in search_roots:
            for name in fake_dir_names:
                d = search_root / name
                if d.exists():
                    wavs = sorted(d.glob("*.wav"))
                    if wavs:
                        print(f"  Fake audio : {len(wavs):,} files  ({d.name})")
                        for f in wavs:
                            samples.append((f, 1))
                        found_fakes += len(wavs)

        if found_fakes == 0:
            # Fallback: scan everything under generated_audio
            if generated.exists():
                for sub in generated.iterdir():
                    if sub.is_dir():
                        wavs = sorted(sub.glob("*.wav"))
                        if wavs:
                            print(f"  Fake audio : {len(wavs):,} files  ({sub.name})")
                            for f in wavs:
                                samples.append((f, 1))

        return samples

    def _split(self, samples, split, val_split, test_split, seed):
        rng = random.Random(seed)
        rng.shuffle(samples)
        n       = len(samples)
        n_test  = int(n * test_split)
        n_val   = int(n * val_split)
        n_train = n - n_test - n_val
        if split == "train": return samples[:n_train]
        if split == "val":   return samples[n_train : n_train + n_val]
        if split == "test":  return samples[n_train + n_val:]
        raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        try:
            waveform = preprocess_audio(str(filepath))
        except Exception as e:
            waveform = torch.zeros(1, 64000)

        features = self.extractor.get_mel_spectrogram(waveform)
        features = self.extractor.normalize(features)
        return features, torch.tensor(label, dtype=torch.long)


def get_dataloaders(data_dir: str, batch_size: int = 32):
    # 1. Initialize the three splits
    train_ds = WaveFakeDataset(data_dir, split="train")
    val_ds   = WaveFakeDataset(data_dir, split="val")
    test_ds  = WaveFakeDataset(data_dir, split="test")

    # 2. Create Loaders explicitly (No **kw to avoid 'multiple values' error)
    # We set num_workers=0 for Windows stability
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False
    )

    return train_loader, val_loader, test_loader