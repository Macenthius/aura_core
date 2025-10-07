# AuraCore/auracore_interactive.py

import sounddevice as sd
import numpy as np
import torch
import sys
import os
import time
import threading
import copy
from collections import deque

# --- Component Imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from ai.feature_extractor import FeatureExtractor
from ai.prediction_model import PredictionModel, scale_parameters
from core.dsp import EightBandEQ, Compressor, LookaheadLimiter

# --- Configuration ---
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
CHANNELS = 1
INPUT_DEVICE = None   # <-- SET YOUR VIRTUAL CABLE DEVICE INDEX
OUTPUT_DEVICE = None  # <-- SET YOUR SPEAKER DEVICE INDEX
MODEL_PATH = "models/"
FE_MODEL_FILE = os.path.join(MODEL_PATH, "feature_extractor.pth")
PM_MODEL_FILE = os.path.join(MODEL_PATH, "prediction_model.pth")

# --- Shared State for Multi-threading ---
# This data is shared between the audio thread and the main (command) thread
shared_state = {
    "overrides": {},  # User's manual parameter changes
    "latest_feature_vector": torch.zeros(1, 32), # The last feature vector seen by the audio thread
    "running": True
}
state_lock = threading.Lock() # A lock to prevent race conditions when accessing shared_state

# --- Online Learning ---
# A small buffer of recent corrections to fine-tune the model
replay_buffer = deque(maxlen=64)
learning_rate_finetune = 1e-5

# --- Initialize All Components ---
print("--- AuraCore: Initializing Interactive Engine ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = FeatureExtractor().to(device).eval()
prediction_model = PredictionModel().to(device) # Keep in train() mode for fine-tuning
# Load pre-trained weights if they exist
if os.path.exists(FE_MODEL_FILE): feature_extractor.load_state_dict(torch.load(FE_MODEL_FILE, map_location=device))
if os.path.exists(PM_MODEL_FILE): prediction_model.load_state_dict(torch.load(PM_MODEL_FILE, map_location=device))
# Set up a separate optimizer for fine-tuning
optimizer = torch.optim.Adam(prediction_model.parameters(), lr=learning_rate_finetune)
loss_fn = torch.nn.MSELoss()
# DSP components
eq = EightBandEQ(sample_rate=SAMPLE_RATE)
comp = Compressor(sample_rate=SAMPLE_RATE)
limiter = LookaheadLimiter(sample_rate=SAMPLE_RATE)

def audio_thread_func():
    """The function that runs the audio stream in a separate thread."""
    
    def audio_callback(indata, outdata, frames, time, status):
        """The real-time audio processing callback."""
        global shared_state
        audio_buffer = indata[:, 0].astype(np.float32)

        # AI Processing
        input_tensor = torch.from_numpy(audio_buffer).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            feature_vector = feature_extractor(input_tensor)
            raw_params = prediction_model(feature_vector)
        
        # Safely update the shared feature vector
        with state_lock:
            shared_state["latest_feature_vector"] = feature_vector.detach().clone()
        
        # Scale AI output and apply user overrides
        scaled_params = scale_parameters(raw_params)
        with state_lock:
            for key, value in shared_state["overrides"].items():
                # This logic allows overriding specific params, e.g., 'comp_ratio'
                # The key would be a tuple like ('comp', 'ratio')
                module, param = key
                if module == 'eq': # EQ gains are arrays
                    scaled_params['eq_gains'][param] = value
                else: # Other params are scalars
                    scaled_params[f"{module}_{param}"] = value

        # Set DSP parameters
        eq.set_parameters(scaled_params['eq_gains'], scaled_params['eq_qs'])
        comp.set_parameters(scaled_params['comp_threshold'], scaled_params['comp_ratio'], scaled_params['comp_attack'], scaled_params['comp_release'], scaled_params['comp_makeup'])
        limiter.set_parameters(scaled_params['limiter_release'], scaled_params['limiter_ceiling'])

        # Process audio through DSP chain
        processed_buffer = limiter.process(comp.process(eq.process(audio_buffer)))
        outdata[:] = processed_buffer.reshape(-1, 1)

    # Start the stream
    try:
        with sd.Stream(device=(INPUT_DEVICE, OUTPUT_DEVICE), samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE, channels=CHANNELS, dtype='float32', callback=audio_callback):
            while shared_state["running"]:
                time.sleep(0.1)
    except Exception as e:
        print(f"\nAudio thread error: {e}")
        shared_state["running"] = False

def fine_tune_model():
    """Performs a few steps of training on the corrections in the replay buffer."""
    if len(replay_buffer) == 0:
        return
    
    # Get a random sample of corrections from the buffer
    sample_indices = np.random.randint(0, len(replay_buffer), min(8, len(replay_buffer)))
    batch = [replay_buffer[i] for i in sample_indices]
    
    features = torch.cat([item[0] for item in batch], dim=0).to(device)
    target_params = torch.cat([item[1] for item in batch], dim=0).to(device)
    
    # Fine-tuning step
    optimizer.zero_grad()
    predicted_params = prediction_model(features)
    loss = loss_fn(predicted_params, target_params)
    loss.backward()
    optimizer.step()
    print(f"Fine-tuned model with {len(batch)} examples. Loss: {loss.item():.6f}")

def main():
    """The main thread for handling user input and triggering learning."""
    if INPUT_DEVICE is None or OUTPUT_DEVICE is None:
        print("Please configure your INPUT_DEVICE and OUTPUT_DEVICE in the script."); return

    # Start the audio processing in a background thread
    audio_thread = threading.Thread(target=audio_thread_func)
    audio_thread.start()
    
    print("\n--- AuraCore Interactive Engine is LIVE ---")
    print("Type commands to tweak the sound and teach the AI.")
    print("Commands: eq <band> <gain>, comp <param> <value>, reset, exit")
    print("Example: 'eq 4 +2' or 'comp ratio 3.5'")
    
    last_ai_params = None
    while shared_state["running"]:
        try:
            cmd = input("> ").lower().strip()
            if not cmd: continue

            if cmd == 'exit':
                shared_state["running"] = False; break
            if cmd == 'reset':
                with state_lock: shared_state["overrides"].clear()
                print("Manual overrides cleared.")
                continue

            parts = cmd.split()
            # Command parsing logic here... (example for eq and comp)
            with state_lock:
                current_feature_vector = shared_state["latest_feature_vector"].clone()
                # Get the AI's current raw prediction for this feature vector
                with torch.no_grad():
                    ai_raw_params = prediction_model(current_feature_vector)
                
                # Create a target by modifying the AI's prediction with the user's tweak
                target_raw_params = ai_raw_params.clone()

                # --- This is a simplified parser ---
                if parts[0] == 'eq' and len(parts) == 3:
                    band, gain = int(parts[1]), float(parts[2])
                    shared_state["overrides"][('eq', band)] = gain
                    # We need to reverse the scaling to update the raw target
                    target_raw_params[0, band] = torch.atanh(torch.tensor(gain / 15.0))
                    print(f"Overriding EQ band {band} to {gain} dB.")
                elif parts[0] == 'comp' and len(parts) == 3:
                    param, val = parts[1], float(parts[2])
                    shared_state["overrides"][('comp', param)] = val
                    # This requires more complex inverse scaling for each parameter...
                    # For simplicity, we will focus on fine-tuning from EQ changes for now.
                    print(f"Overriding Compressor {param} to {val}.")
                else:
                    print("Unknown command.")
                    continue
                
                # Add the correction to our replay buffer for learning
                replay_buffer.append((current_feature_vector, target_raw_params))
            
            # Trigger a fine-tuning step
            fine_tune_model()

        except (KeyboardInterrupt, EOFError):
            shared_state["running"] = False
        except Exception as e:
            print(f"Error in command loop: {e}")

    audio_thread.join()
    print("--- Engine Shut Down ---")

if __name__ == "__main__":
    main()