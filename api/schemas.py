# api/schemas.py
from pydantic import BaseModel
from typing import List


class PredictionResponse(BaseModel):
    label:         str          # "real" or "fake"
    fake_prob:     float
    real_prob:     float
    confidence:    float
    threshold:     float
    inference_ms:  float
    attention:     List[float]

class HealthResponse(BaseModel):
    status:  str
    device:  str
    model:   str