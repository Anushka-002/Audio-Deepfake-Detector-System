# 🎙️ VoiceGuard — Audio Deepfake Detection

> Upload a voice. Know in seconds if it's real or AI-generated.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green) ![React](https://img.shields.io/badge/React-18-61DAFB) ![MongoDB](https://img.shields.io/badge/MongoDB-7-47A248)

---

## 🧠 What it does

VoiceGuard detects AI-generated (deepfake) audio using a CNN trained on mel spectrograms. It finds the invisible frequency-level fingerprints that AI vocoders leave behind — patterns humans can't hear but the model can see.

```
audio.wav  →  Mel Spectrogram  →  DeepfakeCNN  →  FAKE (93.4% confidence)
```

**Why it matters:** Voice cloning is being used for fraud, vishing attacks, and identity theft. This is a practical defense.

---

## 🏗️ Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite + TailwindCSS |
| API Gateway | Node.js + Express 5 + JWT |
| Database | MongoDB + Mongoose |
| ML Backend | FastAPI + PyTorch 2.2 |
| Deploy | Docker + Docker Compose |

---

## 🗃️ Dataset — WaveFake

- **Real:** LJSpeech-1.1 — 13,100 clips of real human speech
- **Fake:** 7 neural vocoders (MelGAN, HiFi-GAN, WaveGlow, Parallel WaveGAN + variants)
- **Total:** ~91,700 audio clips
- **Source:** [zenodo.org/record/5642694](https://zenodo.org/record/5642694) — free, no registration
- **License:** Creative Commons Attribution 4.0

---

## ⚙️ Setup

> ⚠️ Use **Python 3.11 only**. PyTorch 2.2 does not support Python 3.12+.


py -3.11 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**2. Install Python packages**
```bash
pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn python-multipart scikit-learn tqdm pyyaml librosa soundfile
```

---

## 🚀 Run

**Train the model first:**
```bash
python ml/train.py
# Trains for 20 epochs — saves best to ml/checkpoints/best_model.pt
```


## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | ~91% |
| F1 Score | ~92% |
| ROC-AUC | ~96% |
| Inference | ~140ms / file |

---

## 🙏 Credits

- [WaveFake](https://arxiv.org/abs/2111.02813) — Frank & Schönherr, 2021
- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) — Keith Ito

---
