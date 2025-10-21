# src/core/dsp_engine.py
"""
Core DSP components for the AuraCore mastering chain.
Includes EQ, Compressor, Limiter, and Stereo Enhancement.
"""

import numpy as np
from scipy import signal
import math

class MasteringChain:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.eq_bands = [60, 150, 400, 1000, 2400, 5000, 10000, 15000]
        # State for the limiter's gain envelope
        self.limiter_gain = 1.0
        print("DSP Mastering Chain initialized.")

    def _create_filter(self, center_freq, gain_db, Q=1.0):
        """Creates a single second-order IIR (biquad) peaking filter."""
        b, a = signal.iirpeak(center_freq, Q, fs=self.sample_rate)
        return b, a

    def apply_eq(self, audio_buffer, eq_params):
        """Applies an 8-band parametric EQ."""
        gains_db = eq_params.get('gains', [0]*8)
        processed_buffer = np.copy(audio_buffer)
        
        for i, center_freq in enumerate(self.eq_bands):
            gain_db = gains_db[i]
            if gain_db == 0:
                continue
            
            gain_linear = 10 ** (gain_db / 20.0)
            b, a = self._create_filter(center_freq, gain_db)
            filtered_signal = signal.lfilter(b, a, processed_buffer, axis=0)
            
            if gain_linear > 1.0: # Boost
                processed_buffer += (filtered_signal - processed_buffer) * (gain_linear - 1.0)
            else: # Cut
                processed_buffer -= (filtered_signal - processed_buffer) * (1.0 - gain_linear)

        return processed_buffer

    def apply_compressor(self, audio_buffer, comp_params):
        """Applies a digital audio compressor."""
        threshold_db = comp_params.get('threshold', -12.0)
        ratio = comp_params.get('ratio', 4.0)
        attack_ms = comp_params.get('attack', 5.0)
        release_ms = comp_params.get('release', 100.0)
        makeup_gain_db = comp_params.get('makeup_gain', 0.0)

        threshold_linear = 10 ** (threshold_db / 20.0)
        makeup_gain_linear = 10 ** (makeup_gain_db / 20.0)
        attack_coeff = math.exp(-1.0 / (attack_ms * self.sample_rate / 1000.0))
        release_coeff = math.exp(-1.0 / (release_ms * self.sample_rate / 1000.0))

        gain = 1.0
        output_buffer = np.zeros_like(audio_buffer)
        for i in range(audio_buffer.shape[0]):
            input_sample_abs = np.max(np.abs(audio_buffer[i]))
            if input_sample_abs > gain:
                gain = (1.0 - attack_coeff) * input_sample_abs + attack_coeff * gain
            else:
                gain = (1.0 - release_coeff) * input_sample_abs + release_coeff * gain
            
            computed_gain = 1.0
            if gain > threshold_linear:
                computed_gain = threshold_linear + ((gain - threshold_linear) / ratio)
                computed_gain /= gain
            
            output_buffer[i] = audio_buffer[i] * computed_gain * makeup_gain_linear
        return output_buffer

    def apply_limiter(self, audio_buffer, limit_params):
        """Applies a lookahead brickwall limiter."""
        ceiling_db = limit_params.get('ceiling', -0.1)
        lookahead_ms = limit_params.get('lookahead', 1.5)
        release_ms = limit_params.get('release', 50.0)

        ceiling_linear = 10 ** (ceiling_db / 20.0)
        lookahead_samples = int(lookahead_ms * self.sample_rate / 1000.0)
        release_coeff = math.exp(-1.0 / (release_ms * self.sample_rate / 1000.0))

        padded_buffer = np.pad(audio_buffer, ((0, lookahead_samples), (0, 0)), 'constant')
        output_buffer = np.zeros_like(audio_buffer)

        for i in range(audio_buffer.shape[0]):
            future_window = padded_buffer[i : i + lookahead_samples]
            future_peak = np.max(np.abs(future_window)) if future_window.size > 0 else 0

            target_gain = 1.0
            if future_peak > ceiling_linear:
                target_gain = ceiling_linear / future_peak
            
            if target_gain < self.limiter_gain:
                self.limiter_gain = target_gain
            else:
                self.limiter_gain = (1.0 - release_coeff) * target_gain + release_coeff * self.limiter_gain

            output_buffer[i] = audio_buffer[i] * self.limiter_gain
        
        return output_buffer

    def apply_stereo_enhancement(self, audio_buffer, stereo_params):
        """Placeholder for stereo width control."""
        return audio_buffer

    def process_buffer(self, audio_buffer, mastering_params):
        """Processes a single audio buffer through the full mastering chain."""
        processed_buffer = np.array(audio_buffer, dtype=np.float64)

        processed_buffer = self.apply_eq(processed_buffer, mastering_params['eq'])
        processed_buffer = self.apply_compressor(processed_buffer, mastering_params['compressor'])
        processed_buffer = self.apply_limiter(processed_buffer, mastering_params['limiter'])
        processed_buffer = self.apply_stereo_enhancement(processed_buffer, mastering_params['stereo'])
        
        return processed_buffer

if __name__ == '__main__':
    # Test with a signal that clips, to verify the limiter's ceiling
    dummy_params = {
        'eq': {'gains': [0]*8},
        'compressor': {'threshold': 0.0, 'ratio': 1.0, 'makeup_gain': 0.0},
        'limiter': {'ceiling': -0.1, 'lookahead': 1.5, 'release': 50.0},
        'stereo': {'width': 1.2}
    }
    
    sr = 44100
    buffer_len = 1024
    t = np.linspace(0, buffer_len/sr, buffer_len, endpoint=False)
    clipping_signal = 1.5 * np.sin(2 * np.pi * 440 * t)
    dummy_audio = np.column_stack([clipping_signal, clipping_signal])

    chain = MasteringChain(sample_rate=sr)
    processed_audio = chain.process_buffer(dummy_audio, dummy_params)

    print("\n--- DSP Chain Test (with Limiter) ---")
    
    original_peak = np.max(np.abs(dummy_audio))
    processed_peak = np.max(np.abs(processed_audio))
    target_peak = 10**(dummy_params['limiter']['ceiling'] / 20.0)

    print(f"Original Peak Amplitude: {original_peak:.4f}")
    print(f"Processed Peak Amplitude: {processed_peak:.4f}")
    print(f"Target Ceiling Amplitude: {target_peak:.4f}")
    
    if np.isclose(processed_peak, target_peak, atol=1e-4):
        print("✅ SUCCESS: Limiter successfully capped the signal at the ceiling.")
    else:
        print("⚠️ ERROR: Limiter did not control the signal to the target ceiling.")