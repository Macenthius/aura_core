# src/core/audio_analyzer.py
"""
Analyzes audio files to extract key mastering-related features.
- Loudness (LUFS)
- Brightness (Spectral Centroid)
- Dynamics (Crest Factor)
"""

import librosa
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

def analyze_track(file_path):
    """
    Analyzes a single audio file and returns a dictionary of features.
    """
    try:
        audio, sr = sf.read(file_path)
        # Ensure audio is mono for most feature analyses
        if audio.ndim > 1:
            audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio

    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None

    features = {}

    # 1. Loudness (LUFS) - using the full stereo/mono signal
    meter = pyln.Meter(sr) # create BS.1770 meter
    loudness = meter.integrated_loudness(audio)
    features['loudness_lufs'] = loudness

    # 2. Dynamics (Crest Factor)
    rms_value = np.mean(librosa.feature.rms(y=audio_mono))
    peak_value = np.max(np.abs(audio_mono))
    if rms_value > 0:
        crest_factor = 20 * np.log10(peak_value / rms_value)
        features['crest_factor_db'] = crest_factor
    else:
        features['crest_factor_db'] = 0.0

    # 3. Brightness (Spectral Centroid)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_mono, sr=sr))
    features['spectral_centroid_hz'] = spectral_centroid

    return features

if __name__ == '__main__':
    print("--- Testing Audio Analyzer ---")
    # We'll test this on one of the files we created on Day 2
    test_file = 'data/audio_samples/sine_440hz.wav'
    
    analysis_results = analyze_track(test_file)

    if analysis_results:
        print(f"Analysis for: {test_file}")
        for feature, value in analysis_results.items():
            print(f"- {feature.replace('_', ' ').title()}: {value:.2f}")

    print("\n--- Testing on a more complex signal ---")
    test_file_noise = 'data/audio_samples/white_noise.wav'
    analysis_results_noise = analyze_track(test_file_noise)

    if analysis_results_noise:
        print(f"Analysis for: {test_file_noise}")
        for feature, value in analysis_results_noise.items():
            print(f"- {feature.replace('_', ' ').title()}: {value:.2f}")