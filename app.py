from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Dict
import torch
import torch.nn as nn
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Model definition
class FastSentimentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Request model
class SentimentRequest(BaseModel):
    text: str

# Global variables to store model and vectorizer
model = None
vectorizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model and vectorizer
    global model, vectorizer
    
    try:
        # Load the vectorizer
        with open('trained_model/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Initialize and load the model
        input_size = len(vectorizer.get_feature_names_out())
        model = FastSentimentClassifier(input_size=input_size)
        model.load_state_dict(torch.load('trained_model/best_model.pt', map_location=device))
        model.to(device)
        model.eval()
        print("Model and vectorizer loaded successfully")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise RuntimeError("Failed to load model and vectorizer")
    
    yield  # Server is running and ready to handle requests
    
    # Cleanup (if needed)
    print("Shutting down and cleaning up")

# Initialize FastAPI with lifespan
app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)

@app.post("/predict", response_model=Dict[str, float])
async def predict_sentiment(request: SentimentRequest):
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Transform the input text
        features = vectorizer.transform([request.text])
        features_tensor = torch.FloatTensor(features.toarray()).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(features_tensor)
            probability = torch.sigmoid(output).item()
        
        return {
            "positive_probability": probability,
            "negative_probability": 1 - probability
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    if model and vectorizer:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}