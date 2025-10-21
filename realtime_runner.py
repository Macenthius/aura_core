# AuraCore/realtime_runner.py

import sounddevice as sd
import numpy as np
import torch
import sys
import os
import time

# (All component imports remain the same)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from ai.feature_extractor import FeatureExtractor
from ai.prediction_model import PredictionModel, scale_parameters
from core.dsp import EightBandEQ, Compressor, LookaheadLimiter

# --- Configuration ---
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
CHANNELS = 1
MODEL_PATH = "models/"
FE_MODEL_FILE = os.path.join(MODEL_PATH, "feature_extractor.pth")
PM_MODEL_FILE = os.path.join(MODEL_PATH, "prediction_model.pth")

# --- Device Selection ---
# SET THESE VALUES AFTER RUNNING THE SCRIPT ONCE TO SEE YOUR DEVICE LIST
# Example: INPUT_DEVICE = 3, OUTPUT_DEVICE = 5
INPUT_DEVICE = None   # <-- CHANGE THIS
OUTPUT_DEVICE = None  # <-- CHANGE THIS

# (Global component initialization code remains the same)
print("--- AuraCore: Initializing Real-time Engine ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = FeatureExtractor().to(device)
prediction_model = PredictionModel().to(device)
# (Model loading logic remains the same)
try:
    if os.path.exists(FE_MODEL_FILE):
        feature_extractor.load_state_dict(torch.load(FE_MODEL_FILE, map_location=device))
        print("Loaded trained weights for Feature Extractor.")
    if os.path.exists(PM_MODEL_FILE):
        prediction_model.load_state_dict(torch.load(PM_MODEL_FILE, map_location=device))
        print("Loaded trained weights for Prediction Model.")
except Exception as e:
    print(f"Error loading model weights: {e}.")
feature_extractor.eval(); prediction_model.eval()
if device.type == 'cuda':
    feature_extractor.half(); prediction_model.half()
eq = EightBandEQ(sample_rate=SAMPLE_RATE)
comp = Compressor(sample_rate=SAMPLE_RATE)
limiter = LookaheadLimiter(sample_rate=SAMPLE_RATE)

# (The audio_callback function remains exactly the same)
def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_buffer = indata[:, 0].astype(np.float32)
    input_tensor = torch.from_numpy(audio_buffer).unsqueeze(0).unsqueeze(0).to(device)
    if device.type == 'cuda':
        input_tensor = input_tensor.half()
    with torch.no_grad():
        feature_vector = feature_extractor(input_tensor)
        raw_params = prediction_model(feature_vector)
    scaled_params = scale_parameters(raw_params)
    eq.set_parameters(scaled_params['eq_gains'], scaled_params['eq_qs'])
    comp.set_parameters(scaled_params['comp_threshold'], scaled_params['comp_ratio'],
                        scaled_params['comp_attack'], scaled_params['comp_release'],
                        scaled_params['comp_makeup'])
    limiter.set_parameters(scaled_params['limiter_release'], scaled_params['limiter_ceiling'])
    processed_buffer = limiter.process(comp.process(eq.process(audio_buffer)))
    outdata[:] = processed_buffer.reshape(-1, 1)

def main():
    """Sets up and runs the real-time audio stream."""
    if INPUT_DEVICE is None or OUTPUT_DEVICE is None:
        print("--- Please select your audio devices ---")
        print("Available Devices:")
        print(sd.query_devices())
        print("\nEdit `realtime_runner.py` and set the INPUT_DEVICE and OUTPUT_DEVICE variables")
        print("to the index numbers of the devices you want to use.")
        print("INPUT should be your virtual cable ('CABLE Output').")
        print("OUTPUT should be your main speakers or headphones.")
        return

    print("\n--- Starting Real-time Audio Stream ---")
    print(f"Input: {sd.query_devices()[INPUT_DEVICE]['name']}")
    print(f"Output: {sd.query_devices()[OUTPUT_DEVICE]['name']}")
    print("Press Ctrl+C to stop.")
    
    try:
        with sd.Stream(device=(INPUT_DEVICE, OUTPUT_DEVICE),
                       samplerate=SAMPLE_RATE,
                       blocksize=BUFFER_SIZE,
                       channels=CHANNELS,
                       dtype='float32',
                       callback=audio_callback):
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n--- Stream stopped by user ---")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()