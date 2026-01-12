"""
===========================================================================
AURACORE: THE HYBRID BRAIN (AI Model)
===========================================================================

WHAT IS THIS FILE?
This file contains the "Neural Network Architecture". 
It defines the structure of the AI's "Brain Cells" (Layers).

THE TWO HALVES OF THE BRAIN:
1. THE FEATURE EXTRACTOR (CNN): 
   - Uses "Convolutional" layers (CNN).
   - Like an ear, it listens to the raw audio waves and looks for patterns
     (rhythm, brightness, bass levels).
   - It turns 4,096 audio samples into 32 "Clues" (Features).

2. THE PREDICTION MODEL (MLP):
   - Uses "Linear" layers (MLP).
   - It takes those 32 clues and tries to solve the puzzle:
     "Which slider settings will make this specific audio sound professional?"
   - It outputs 23 numbers (the predicted settings).

LINGO FOR BEGINNERS:
- Conv1d: A layer that scans the audio for specific patterns (like a magnifying glass).
- ReLU: An activation function that helps the AI learn complex, non-linear patterns.
- BatchNorm: Keeps the numbers inside the brain in a healthy range so it doesn't get "confused".
- Linear: A standard brain connection that relates one clue to another.
===========================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridModel(nn.Module):
    """
    A single model that combines the 'Ear' (Feature Extractor) 
    and the 'Logic' (Prediction Model).
    """
    def __init__(self):
        super(HybridModel, self).__init__()
        
        # --- PART 1: THE EAR (Feature Extractor) ---
        self.feature_extractor = nn.Sequential(
            # First layer: Looking for broad patterns.
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            
            # Second layer: Looking for more detail.
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            # Third layer: High-level audio characteristics.
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            # Condense the data down.
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(256, 32) # Result: 32 Clues (Features).
        )
        
        # --- PART 2: THE LOGIC (Parameter Predictor) ---
        self.predictor = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 23) # Result: 23 Predicted Parameters.
        )

    def forward(self, x):
        """
        How data flows through the model:
        Raw Audio -> Ear -> Clues -> Logic -> Final Settings.
        """
        # 1. Extract audio features (The 32 clues).
        features = self.feature_extractor(x)
        
        # 2. Predict the 23 raw parameter values.
        raw_output = self.predictor(features)
        
        # 3. SCALE THE PARAMETERS (The most important part!)
        # The AI outputs random numbers. We need to turn them into valid 
        # audio settings (like -12dB or 5ms).
        # We use sigmoid() to keep numbers between 0 and 1.
        # We use tanh() to keep numbers between -1 and 1.
        
        # EQ Gains (8 bands): -15dB to +15dB.
        eq_gains = torch.tanh(raw_output[:, 0:8]) * 15.0
        
        # Compressor (5 params):
        c_thresh  = (torch.tanh(raw_output[:, 8:9]) * 30.0) - 30.0 # -60 to 0 dB
        c_ratio   = (torch.sigmoid(raw_output[:, 9:10]) * 19.0) + 1.0  # 1:1 to 20:1
        c_attack  = (torch.sigmoid(raw_output[:, 10:11]) * 99.0) + 1.0 # 1 to 100 ms
        c_release = (torch.sigmoid(raw_output[:, 11:12]) * 480.0) + 20.0 # 20 to 500 ms
        c_makeup  = torch.sigmoid(raw_output[:, 12:13]) * 12.0 # 0 to 12 dB
        
        # Limiter (2 params):
        l_ceiling = (torch.sigmoid(raw_output[:, 13:14]) * -0.9) - 0.1 # -1.0 to -0.1 dB
        l_release = (torch.sigmoid(raw_output[:, 14:15]) * 480.0) + 20.0 # 20 to 500 ms
        
        # Put it all in a nice dictionary for the AI Worker to use.
        return {
            'eq_gains': eq_gains.unsqueeze(1),
            'comp_params': torch.stack([c_thresh, c_ratio, c_attack, c_release, c_makeup], dim=2),
            'limit_params': torch.stack([l_ceiling, l_release], dim=2)
        }, features
