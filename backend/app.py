from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from model.utils import preprocess_input, weighted_bce
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("model.keras", custom_objects={'loss': weighted_bce  })

class PredictionInput(BaseModel):
    industry: str
    country: str
    cybersecurity: str

@app.post("/predict")
def predict(data: PredictionInput):
    X = preprocess_input(data)
    X_array = np.array([X], dtype=np.float32)   # 2D input for model
    proba = model.predict(X_array)[0][0]
    label = int(proba > 0.5)
    return {"probability": float(proba), "attack_predicted": bool(label)}
