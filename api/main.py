import os
os.environ["TORCH_AUDIO_BACKEND"] = "soundfile"
import torch
import yaml
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from ml.model import DeepfakeDetectorCNN
from ml.preprocessing import preprocess_audio
# If you created extractors.py inside ml/
from ml.extractors import AudioFeatureExtractor

app = FastAPI(title="Audio Deepfake Detector API")

# 1. SETUP CORS (Cross-Origin Resource Sharing)
# This allows your React frontend (port 5173) to talk to this API (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. GLOBAL STATE
# We define these as None first and load them on startup
model = None
extractor = None
config = None

@app.on_event("startup")
async def startup_event():
    """
    This runs ONCE when the server starts.
    We load the model into memory here so it's ready for instant predictions.
    """
    global model, extractor, config
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize Extractor
    extractor = AudioFeatureExtractor(config)
    
    # Initialize Model
    model = DeepfakeDetectorCNN()
    
    # Load Weights (Once you finish Phase 9, you'll have a .pth file)
    # model_path = Path("ml/checkpoints/best_model.pth")
    # if model_path.exists():
    #     model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    model.eval() # Set to evaluation mode (turns off Dropout)
    print("✅ System Ready: Model and Extractor loaded.")

# 3. THE PREDICTION ENDPOINT
@app.post("/detect")
async def detect_audio(file: UploadFile = File(...)):
    """
    Receives an audio file, processes it, and returns the 'Fake' probability.
    """
    # Create a temporary directory to save the upload
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_path = temp_dir / file.filename

    try:
        # Save the uploaded file to disk
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 1: Preprocess (16kHz, 4s, Mono)
        waveform = preprocess_audio(file_path, config)
        
        # Step 2: Feature Extraction (Spectrogram)
        # We use .unsqueeze(0) to add a 'Batch' dimension [1, 1, 128, 251]
        features = extractor.get_mel_spec(waveform).unsqueeze(0)

        # Step 3: Inference (Prediction)
        with torch.no_grad(): # Disable gradient calculation (faster/less memory)
            output = model(features)
            probability = output.item() # Get the single number

        # Step 4: Result Logic
        label = "FAKE" if probability > 0.5 else "REAL"
        confidence = probability if label == "FAKE" else (1 - probability)

        return {
            "status": "success",
            "filename": file.filename,
            "prediction": label,
            "probability": round(probability, 4),
            "confidence_score": f"{round(confidence * 100, 2)}%"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Always clean up the temporary file!
        if file_path.exists():
            os.remove(file_path)

@app.get("/health")
def health_check():
    return {"status": "online", "model_loaded": model is not None}