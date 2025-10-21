# AuraCore/main.py

import torch
import numpy as np
import time
import sys
import os

# Add src to the Python path to allow for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# --- Component Imports ---
from ai.feature_extractor import FeatureExtractor, INPUT_SAMPLES
from ai.prediction_model import PredictionModel, scale_parameters
from core.dsp import EightBandEQ, Compressor, LookaheadLimiter

# --- System Configuration ---
SAMPLE_RATE = 44100
BUFFER_SIZE = INPUT_SAMPLES
SIMULATION_DURATION_S = 10 

def simulate_audio_stream(duration_s, sr, buffer_size):
    """A generator that yields audio buffers like a real-time stream."""
    print("\n--- Starting Real-time Audio Stream Simulation ---")
    total_samples = duration_s * sr
    num_buffers = total_samples // buffer_size
    
    for i in range(num_buffers):
        # Generate a sine wave that sweeps in frequency to test adaptability
        start_sample = i * buffer_size
        t = np.linspace(start_sample / sr, (start_sample + buffer_size) / sr, buffer_size, endpoint=False)
        freq = 220 + 440 * (i / num_buffers)
        buffer = 0.6 * np.sin(2. * np.pi * freq * t) # 0.6 amplitude
        yield buffer.astype(np.float32)

def main():
    """Main function to run the full end-to-end system verification."""
    print("--- AuraCore: Full System Pipeline Verification ---")

    # 1. Initialize Device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize AI Components
    print("Loading AI models...")
    feature_extractor = FeatureExtractor().to(device)
    prediction_model = PredictionModel().to(device)
    
    feature_extractor.eval()
    prediction_model.eval()

    if device.type == 'cuda':
        feature_extractor.half()
        prediction_model.half()
        print("AI models set to FP16 on GPU for maximum performance.")

    # 3. Initialize DSP Components
    print("Initializing DSP chain (EQ, Compressor, Limiter)...")
    eq = EightBandEQ(sample_rate=SAMPLE_RATE)
    comp = Compressor(sample_rate=SAMPLE_RATE)
    limiter = LookaheadLimiter(sample_rate=SAMPLE_RATE)

    # 4. Run Simulated Audio Processing Loop
    audio_stream = simulate_audio_stream(SIMULATION_DURATION_S, SAMPLE_RATE, BUFFER_SIZE)
    latencies = []
    
    for audio_buffer in audio_stream:
        # --- This is the complete real-time processing loop ---
        start_time = time.perf_counter()

        # a. Convert numpy buffer to torch tensor for AI processing
        input_tensor = torch.from_numpy(audio_buffer).unsqueeze(0).unsqueeze(0).to(device)
        if device.type == 'cuda':
            input_tensor = input_tensor.half()
        
        with torch.no_grad():
            # b. AI: Extract features from the audio
            feature_vector = feature_extractor(input_tensor)
            
            # c. AI: Predict DSP parameters from the features
            raw_params = prediction_model(feature_vector)

        # d. Scale the raw AI output to usable parameter ranges
        scaled_params = scale_parameters(raw_params)
        
        # e. Set the parameters on our DSP modules
        eq.set_parameters(scaled_params['eq_gains'], scaled_params['eq_qs'])
        comp.set_parameters(
            scaled_params['comp_threshold'], scaled_params['comp_ratio'],
            scaled_params['comp_attack'], scaled_params['comp_release'],
            scaled_params['comp_makeup']
        )
        limiter.set_parameters(scaled_params['limiter_release'], scaled_params['limiter_ceiling'])

        # f. Process the audio through the full DSP chain
        eq_buffer = eq.process(audio_buffer)
        comp_buffer = comp.process(eq_buffer)
        processed_buffer = limiter.process(comp_buffer)

        # g. Stop timer and record latency
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

    # 5. Report Final Performance
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    
    print("\n--- Simulation Complete: Performance Report ---")
    print(f"Average total latency (AI + Full DSP): {avg_latency:.4f} ms")
    print(f"Maximum total latency: {max_latency:.4f} ms")
    print(f"Real-time budget: < 50ms (Achieved: {'Yes!' if max_latency < 50 else 'No, optimization required.'})")
    print("\nFull end-to-end pipeline is functional.")

if __name__ == "__main__":
    main()