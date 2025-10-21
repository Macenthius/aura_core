# data/data_generator.py
"""
Generates training data by finding optimal DSP parameters to match
an input track's audio features to a reference track's features.
"""

import numpy as np
import soundfile as sf
from scipy.optimize import minimize
import sys
import os

# Add the project root to the Python path to allow imports from `src`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- FIX: ADD THIS IMPORT BACK ---
from src.core.dsp_engine import MasteringChain
# ----------------------------------
from src.core.audio_analyzer import analyze_track

# --- FIX: ADD THIS OBJECT CREATION BACK ---
CHAIN = MasteringChain()
# ------------------------------------------
SR = 44100

def objective_function(params, input_audio, target_features):
    """
    This is the function we want to minimize.
    It returns the "error" or "distance" between the processed audio
    and the target features.
    """
    eq_gains = params[:8]
    compressor_threshold = params[8]
    compressor_ratio = params[9]
    
    mastering_params = {
        'eq': {'gains': eq_gains},
        'compressor': {'threshold': compressor_threshold, 'ratio': compressor_ratio},
        'limiter': {'ceiling': -0.1},
        'stereo': {'width': 1.0}
    }

    processed_audio = CHAIN.process_buffer(input_audio, mastering_params)
    
    sf.write('temp_processed.wav', processed_audio, SR)
    current_features = analyze_track('temp_processed.wav')

    error = 0.0
    if current_features:
        error += ((current_features['loudness_lufs'] - target_features['loudness_lufs']) / 10.0) ** 2
        error += ((current_features['crest_factor_db'] - target_features['crest_factor_db']) / 10.0) ** 2
        error += ((current_features['spectral_centroid_hz'] - target_features['spectral_centroid_hz']) / 5000.0) ** 2
    else:
        return 1e6
    
    print(f"Trying params... Error: {error:.4f}")
    return error

if __name__ == '__main__':
    print("--- Starting Data Generation (Optimization Test) ---")

    input_file = 'data/audio_samples/white_noise.wav'
    reference_file = 'data/audio_samples/sine_440hz.wav'

    print(f"Input: {input_file}")
    print(f"Reference: {reference_file}")

    input_audio, _ = sf.read(input_file)
    target_features = analyze_track(reference_file)
    print(f"\nTarget Features: {target_features}")

    if input_audio.ndim == 1:
        print("Input audio is mono. Converting to stereo.")
        input_audio = np.column_stack([input_audio, input_audio])

    initial_params = np.array([0.0] * 8 + [-20.0, 4.0])
    bounds = [(-12.0, 12.0)] * 8 + [(-40.0, 0.0), (1.0, 20.0)]

    print("\nStarting optimization... (This may take a minute)")
    result = minimize(
        objective_function,
        initial_params,
        args=(input_audio, target_features),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 50}
    )
    
    if os.path.exists('temp_processed.wav'):
        os.remove('temp_processed.wav')

    print("\n--- Optimization Complete ---")
    if result.success:
        print("✅ Optimizer found a solution.")
        optimal_params = result.x
        print("\nOptimal Parameters Found:")
        print(f"  - EQ Gains (dB): {np.round(optimal_params[:8], 2)}")
        print(f"  - Compressor Threshold (dB): {optimal_params[8]:.2f}")
        print(f"  - Compressor Ratio: {optimal_params[9]:.2f}:1")
    else:
        print(f"⚠️ Optimizer did not converge: {result.message}")