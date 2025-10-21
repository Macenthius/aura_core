# src/ai/mastering_model.py
"""
Contains the core PyTorch models for AuraCore's AI decision-making.
- GenreClassifier: Predicts the genre of the audio.
- ParameterPredictor: Predicts DSP parameters based on audio features and genre.
- AuraCoreModel: The main model that encapsulates the entire AI pipeline.
"""

import torch
import torch.nn as nn
from .feature_extractor import AudioFeatureExtractor # Import from the same 'ai' directory

# --- Constants ---
NUM_GENRES = 4 # Example: Pop, Rock, Electronic, Hip-Hop
NUM_EQ_BANDS = 8
# Compressor (4 params) + Limiter (1 param)
NUM_DYNAMICS_PARAMS = 4 + 1 
TOTAL_PARAMS = NUM_EQ_BANDS + NUM_DYNAMICS_PARAMS

class GenreClassifier(nn.Module):
    """A simple MLP to classify genre from audio features."""
    def __init__(self, feature_dim=32, num_genres=NUM_GENRES):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_genres)
        )
    def forward(self, features):
        return self.network(features)

class ParameterPredictor(nn.Module):
    """Predicts DSP parameters from audio features and genre."""
    def __init__(self, feature_dim=32, num_genres=NUM_GENRES, out_dim=TOTAL_PARAMS):
        super().__init__()
        # Input is audio features + one-hot encoded genre vector
        self.network = nn.Sequential(
            nn.Linear(feature_dim + num_genres, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, features, genre_one_hot):
        # Concatenate the two inputs
        combined_input = torch.cat((features, genre_one_hot), dim=1)
        return self.network(combined_input)

class AuraCoreModel(nn.Module):
    """The complete end-to-end AI model for real-time mastering."""
    def __init__(self, feature_dim=32, num_genres=NUM_GENRES, num_params=TOTAL_PARAMS):
        super().__init__()
        self.feature_extractor = AudioFeatureExtractor(feature_dim=feature_dim)
        self.genre_classifier = GenreClassifier(feature_dim=feature_dim, num_genres=num_genres)
        self.parameter_predictor = ParameterPredictor(feature_dim=feature_dim, num_genres=num_genres, out_dim=num_params)

    def forward(self, audio_buffer):
        # 1. Extract features from raw audio
        features = self.feature_extractor(audio_buffer)
        
        # 2. Classify genre from features
        genre_logits = self.genre_classifier(features)
        # In a real scenario, we use softmax and argmax
        genre_pred = torch.argmax(genre_logits, dim=1)
        genre_one_hot = nn.functional.one_hot(genre_pred, num_classes=NUM_GENRES).float()
        
        # 3. Predict DSP parameters
        predicted_params = self.parameter_predictor(features, genre_one_hot)
        
        return predicted_params

# --- Test Block ---
if __name__ == '__main__':
    print("--- Testing AuraCore End-to-End AI Model ---")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Instantiate the main model
    model = AuraCoreModel().to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")

    # 2. Create a dummy audio buffer
    dummy_audio = torch.randn(1, 1024).to(DEVICE)

    # 3. Run a forward pass and check the output shape
    with torch.no_grad():
        final_params = model(dummy_audio)
        
    print(f"\nInput audio shape: {dummy_audio.shape}")
    print(f"Final predicted parameter vector shape: {final_params.shape}")

    if final_params.shape[1] == TOTAL_PARAMS:
        print(f"✅ SUCCESS: Output shape ({TOTAL_PARAMS} params) is correct.")
    else:
        print(f"⚠️ ERROR: Output shape is incorrect.")