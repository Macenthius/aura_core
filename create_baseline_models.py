# AuraCore/create_baseline_models.py

import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from ai.feature_extractor import FeatureExtractor
from ai.prediction_model import PredictionModel

MODEL_PATH = "models/"
FINAL_FE_MODEL = os.path.join(MODEL_PATH, "feature_extractor.pth")
FINAL_PM_MODEL = os.path.join(MODEL_PATH, "prediction_model.pth")

def main():
    """
    Saves the initial, untrained state of the models as the baseline.
    """
    print("--- Creating Baseline Models ---")
    os.makedirs(MODEL_PATH, exist_ok=True)

    # 1. Instantiate the models
    feature_extractor = FeatureExtractor()
    prediction_model = PredictionModel()

    # 2. Immediately save their initial state dictionaries
    torch.save(feature_extractor.state_dict(), FINAL_FE_MODEL)
    torch.save(prediction_model.state_dict(), FINAL_PM_MODEL)
    
    print(f"Baseline models saved successfully:")
    print(f"- {FINAL_FE_MODEL}")
    print(f"- {FINAL_PM_MODEL}")

if __name__ == "__main__":
    main()