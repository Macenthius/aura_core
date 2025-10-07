# audio_collector.py
# Generates synthetic audio samples for testing the AuraCore pipeline.

import numpy as np
import soundfile as sf
from pathlib import Path

# --- Configuration ---
SAMPLE_RATE = 44100  # Hz
DURATION = 5  # seconds
AMPLITUDE = 0.5  # Peak amplitude
OUTPUT_DIR = "data/audio_samples"

def generate_sine_wave(freq, duration, sr):
    """Generates a pure sine wave."""
    t = np.linspace(0., duration, int(sr * duration), endpoint=False)
    sine = AMPLITUDE * np.sin(2 * np.pi * freq * t)
    return sine

def generate_white_noise(duration, sr):
    """Generates white noise."""
    noise = AMPLITUDE * np.random.normal(0, 1, int(sr * duration))
    # Normalize to prevent clipping, though random normal is unlikely to clip much
    noise /= np.max(np.abs(noise))
    noise *= AMPLITUDE
    return noise

if __name__ == "__main__":
    print("--- AuraCore: Starting Audio Collector ---")

    # Ensure the output directory exists
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory exists: {output_path.resolve()}")

    # --- Generate and Save Files ---

    # 1. 440 Hz Sine Wave (A4 note)
    print("Generating 440Hz sine wave...")
    sine_440hz = generate_sine_wave(440, DURATION, SAMPLE_RATE)
    sine_filename = output_path / "sine_440hz.wav"
    sf.write(sine_filename, sine_440hz, SAMPLE_RATE)
    print(f"✅ Saved file: {sine_filename}")

    # 2. White Noise
    print("Generating white noise...")
    white_noise = generate_white_noise(DURATION, SAMPLE_RATE)
    noise_filename = output_path / "white_noise.wav"
    sf.write(noise_filename, white_noise, SAMPLE_RATE)
    print(f"✅ Saved file: {noise_filename}")
    
    # 3. 1kHz Sine Wave (common for testing)
    print("Generating 1kHz sine wave...")
    sine_1khz = generate_sine_wave(1000, DURATION, SAMPLE_RATE)
    sine_1khz_filename = output_path / "sine_1khz.wav"
    sf.write(sine_1khz_filename, sine_1khz, SAMPLE_RATE)
    print(f"✅ Saved file: {sine_1khz_filename}")

    print("\n--- Audio collection complete. ---")