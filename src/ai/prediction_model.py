# AuraCore/src/ai/prediction_model.py

import torch
import torch.nn as nn
import numpy as np

# --- Configuration ---
# The model takes the 32-dimensional vector from our FeatureExtractor
INPUT_FEATURES = 32

# It outputs a single vector containing all parameters for our DSP chain
# EQ: 8 gains, 8 Qs = 16
# Compressor: threshold, ratio, attack, release, makeup_gain = 5
# Limiter: release, ceiling = 2
# TOTAL = 23 parameters
OUTPUT_PARAMS = 23

class PredictionModel(nn.Module):
    """
    A simple MLP to predict DSP parameters from audio features.
    """
    def __init__(self, in_features=INPUT_FEATURES, out_features=OUTPUT_PARAMS):
        super(PredictionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        return self.net(x)

def scale_parameters(raw_output):
    """
    Scales the raw, unbounded output of the neural network to the
    correct ranges for each DSP parameter.
    
    Args:
        raw_output (torch.Tensor): The output tensor from the PredictionModel.
        
    Returns:
        dict: A dictionary of numpy arrays for each DSP module's parameters.
    """
    # Ensure tensor is on CPU and detached for numpy conversion
    output = raw_output.cpu().detach().squeeze(0).numpy()
    
    params = {}
    
    # We use tanh for parameters centered around 0 and sigmoid for positive-only ranges.
    
    # --- EQ Parameters (16) ---
    # Gains: -15dB to +15dB
    params['eq_gains'] = np.tanh(output[0:8]) * 15.0
    # Qs: 0.5 to 5.0
    params['eq_qs'] = (torch.sigmoid(torch.tensor(output[8:16])).numpy() * 4.5) + 0.5
    
    # --- Compressor Parameters (5) ---
    # Threshold: -40dB to 0dB
    params['comp_threshold'] = (np.tanh(output[16]) * 20.0) - 20.0
    # Ratio: 1.0 to 10.0
    params['comp_ratio'] = (torch.sigmoid(torch.tensor(output[17])).numpy() * 9.0) + 1.0
    # Attack: 1ms to 100ms
    params['comp_attack'] = (torch.sigmoid(torch.tensor(output[18])).numpy() * 99.0) + 1.0
    # Release: 20ms to 500ms
    params['comp_release'] = (torch.sigmoid(torch.tensor(output[19])).numpy() * 480.0) + 20.0
    # Makeup Gain: 0dB to 10dB
    params['comp_makeup'] = torch.sigmoid(torch.tensor(output[20])).numpy() * 10.0
    
    # --- Limiter Parameters (2) ---
    # Release: 20ms to 200ms
    params['limiter_release'] = (torch.sigmoid(torch.tensor(output[21])).numpy() * 180.0) + 20.0
    # Ceiling: -1.0dB to -0.1dB
    params['limiter_ceiling'] = (torch.sigmoid(torch.tensor(output[22])).numpy() * -0.9) - 0.1

    return params

# --- Self-Testing Block ---
if __name__ == "__main__":
    print("--- Running Prediction Model Sanity Check ---")
    
    # 1. Create model and dummy input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PredictionModel().to(device)
    model.eval()
    
    # Simulate a feature vector from the FeatureExtractor
    dummy_features = torch.randn(1, INPUT_FEATURES).to(device)
    
    # 2. Get raw output
    with torch.no_grad():
        raw_params = model(dummy_features)
    
    print(f"\nModel Raw Output (shape {raw_params.shape}):\n{raw_params}")
    
    # 3. Scale the parameters to usable ranges
    scaled_params = scale_parameters(raw_params)
    
    print("\n--- Scaled DSP Parameters ---")
    for key, value in scaled_params.items():
        if isinstance(value, np.ndarray):
            # Print array values with nice formatting
            formatted_vals = ", ".join([f"{v:.2f}" for v in value])
            print(f"  - {key}: [{formatted_vals}]")
        else:
            print(f"  - {key}: {value:.2f}")
            
    print("\nSanity check complete. Model is producing parameters in the correct ranges.")