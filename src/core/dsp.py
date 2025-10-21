# AuraCore/src/core/dsp.py

import numpy as np
from scipy.signal import lfilter
import soundfile as sf

class EightBandEQ:
    """
    An 8-band parametric equalizer with real-time capabilities and parameter smoothing.
    It works by cascading 8 second-order IIR (biquad) filters.
    (Version 2: Uses direct coefficient calculation for compatibility)
    """
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        self.center_freqs = np.array([60, 150, 400, 1000, 2400, 6000, 10000, 15000], dtype=float)
        num_bands = len(self.center_freqs)
        
        self._target_gains_db = np.zeros(num_bands)
        self._target_qs = np.full(num_bands, 1.0)
        self._current_gains_db = np.zeros(num_bands)
        self._current_qs = np.full(num_bands, 1.0)
        
        self.filters = [self._create_filter(i) for i in range(num_bands)]
        
    def _calculate_peaking_coeffs(self, gain_db, q, fc, sr):
        """
        Calculates biquad peaking filter coefficients from parameters.
        Based on the formulas from the Audio EQ Cookbook by Robert Bristow-Johnson.
        """
        if q <= 0: q = 0.01
        if fc >= sr / 2: fc = sr / 2.0 - 1.0

        A = 10**(gain_db / 40.0)
        w0 = 2.0 * np.pi * fc / sr
        alpha = np.sin(w0) / (2.0 * q)
        
        b0, b1, b2 = 1.0 + alpha * A, -2.0 * np.cos(w0), 1.0 - alpha * A
        a0, a1, a2 = 1.0 + alpha / A, -2.0 * np.cos(w0), 1.0 - alpha / A
        
        b = np.array([b0, b1, b2]) / a0
        a = np.array([a0, a1, a2]) / a0
        
        return b, a

    def _create_filter(self, band_idx):
        """Creates a single biquad filter using the current parameters."""
        gain_db = self._current_gains_db[band_idx]
        q = self._current_qs[band_idx]
        fc = self.center_freqs[band_idx]
        
        b, a = self._calculate_peaking_coeffs(gain_db, q, fc, self.sr)
        
        zi = np.zeros(max(len(a), len(b)) - 1)
        return {'b': b, 'a': a, 'zi': zi}

    def set_parameters(self, gains_db, qs):
        """Public method for the AI to set the target EQ parameters."""
        self._target_gains_db = np.array(gains_db, dtype=float)
        self._target_qs = np.array(qs, dtype=float)

    def process(self, audio_buffer):
        """Processes a single buffer of audio, applying the EQ."""
        for i in range(len(self.center_freqs)):
            b, a = self._calculate_peaking_coeffs(self._target_gains_db[i], self._target_qs[i], self.center_freqs[i], self.sr)
            self.filters[i]['b'], self.filters[i]['a'] = b, a

        processed_buffer = np.copy(audio_buffer)
        
        for i in range(len(self.center_freqs)):
            b, a = self.filters[i]['b'], self.filters[i]['a']
            processed_buffer, zf = lfilter(b, a, processed_buffer, zi=self.filters[i]['zi'])
            self.filters[i]['zi'] = zf

        self._current_gains_db = self._target_gains_db
        self._current_qs = self._target_qs
        
        return processed_buffer

class Compressor:
    """
    A real-time audio compressor with configurable parameters.
    """
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        self._envelope = 0.0
        
        self.threshold_db, self.ratio = -12.0, 3.0
        self.attack_ms, self.release_ms = 5.0, 100.0
        self.makeup_gain_db = 0.0
        
        self._alpha_attack, self._alpha_release = 0.0, 0.0
        self._makeup_gain_linear = 1.0
        
        self._recalculate_coeffs()

    def _recalculate_coeffs(self):
        attack_samples = self.sr * (self.attack_ms / 1000.0)
        self._alpha_attack = np.exp(-1.0 / attack_samples) if attack_samples > 0 else 0.0
        
        release_samples = self.sr * (self.release_ms / 1000.0)
        self._alpha_release = np.exp(-1.0 / release_samples) if release_samples > 0 else 0.0
        
        self._makeup_gain_linear = 10**(self.makeup_gain_db / 20.0)

    def set_parameters(self, threshold_db, ratio, attack_ms, release_ms, makeup_gain_db=0.0):
        self.threshold_db, self.ratio = threshold_db, max(1.0, ratio)
        self.attack_ms, self.release_ms = max(0.1, attack_ms), max(1.0, release_ms)
        self.makeup_gain_db = makeup_gain_db
        self._recalculate_coeffs()

    def process(self, audio_buffer):
        output_buffer = np.zeros_like(audio_buffer)
        threshold_linear = 10**(self.threshold_db / 20.0)
        
        for i, sample in enumerate(audio_buffer):
            sample_abs = abs(sample)
            
            if sample_abs > self._envelope:
                self._envelope = self._alpha_attack * self._envelope + (1 - self._alpha_attack) * sample_abs
            else:
                self._envelope = self._alpha_release * self._envelope + (1 - self._alpha_release) * sample_abs

            gain_reduction_db = 0.0
            if self._envelope > threshold_linear:
                envelope_db = 20 * np.log10(self._envelope)
                overshoot_db = envelope_db - self.threshold_db
                gain_reduction_db = overshoot_db * (1.0 - 1.0 / self.ratio)

            gain_reduction_linear = 10**(-gain_reduction_db / 20.0)
            total_gain = gain_reduction_linear * self._makeup_gain_linear
            output_buffer[i] = sample * total_gain
            
        return output_buffer

# (Replace the old LookaheadLimiter class with this one)
class LookaheadLimiter:
    """
    A real-time, lookahead brickwall limiter.
    (Version 2: Corrected buffer handling logic)
    """
    def __init__(self, sample_rate=44100, lookahead_ms=1.5, release_ms=50.0, ceiling_db=-0.1):
        self.sr = sample_rate
        self.release_ms = release_ms
        self.ceiling_db = ceiling_db
        
        # Lookahead buffer size
        self._lookahead_samples = int(self.sr * lookahead_ms / 1000.0)
        self._delay_buffer = np.zeros(self._lookahead_samples)
        
        self._envelope = 0.0
        self._gain = 1.0 # Current gain, starts at 1.0 (no reduction)
        self._recalculate_coeffs()

    def _recalculate_coeffs(self):
        """Calculates internal coefficients for gain smoothing."""
        release_samples = self.sr * (self.release_ms / 1000.0)
        self._alpha_release = np.exp(-1.0 / release_samples) if release_samples > 0 else 0.0
        self.ceiling_linear = 10**(self.ceiling_db / 20.0)

    def set_parameters(self, release_ms, ceiling_db):
        """Public method for the AI to set limiter parameters."""
        self.release_ms = release_ms
        self.ceiling_db = ceiling_db
        self._recalculate_coeffs()

    def process(self, audio_buffer):
        """Processes a buffer of audio, applying limiting with correct lookahead."""
        # Create a combined signal: the previous delay buffer + the new input buffer
        combined_signal = np.concatenate((self._delay_buffer, audio_buffer))
        
        # The signal we will actually apply gain to and output
        signal_to_output = combined_signal[:len(audio_buffer)]
        
        # Update the delay buffer for the *next* processing block
        self._delay_buffer = combined_signal[len(audio_buffer):]
        
        output_buffer = np.zeros_like(audio_buffer)

        # Loop through the length of the *input buffer*
        for i in range(len(audio_buffer)):
            # The sample we analyze is `lookahead_samples` in the future
            analysis_sample = combined_signal[i + self._lookahead_samples]
            
            # Level detection on the "future" sample
            sample_abs = abs(analysis_sample)
            if sample_abs > self._envelope:
                self._envelope = sample_abs # Instant attack
            else:
                self._envelope = self._alpha_release * self._envelope # Smoothed release
            
            # Gain computation
            target_gain = 1.0
            if self._envelope > self.ceiling_linear:
                target_gain = self.ceiling_linear / self._envelope
            
            # Smooth the actual gain being applied
            if target_gain < self._gain:
                self._gain = target_gain # Instant attack on gain reduction
            else:
                self._gain = self._alpha_release * self._gain + (1 - self._alpha_release) * target_gain

            # Apply the computed gain to the corresponding sample from the output signal
            output_buffer[i] = signal_to_output[i] * self._gain
            
        return output_buffer

def test_eq():
    print("--- Running EQ Self-Test ---")
    SR, DURATION, BUFFER_SIZE = 44100, 5, 1024
    eq = EightBandEQ(sample_rate=SR)
    noise = (np.random.rand(SR * DURATION) * 2 - 1).astype(np.float32)
    output_audio = np.zeros_like(noise)
    mid_boost_gains, qs = np.array([0,0,0,12,12,0,0,0]), np.full(8, 1.5)
    scooped_gains = np.array([6,3,-6,-9,-6,3,6,9])
    num_buffers = len(noise) // BUFFER_SIZE
    for i in range(num_buffers):
        if i == num_buffers // 2: eq.set_parameters(scooped_gains, qs)
        if i == 0: eq.set_parameters(mid_boost_gains, qs)
        start, end = i * BUFFER_SIZE, (i + 1) * BUFFER_SIZE
        output_audio[start:end] = eq.process(noise[start:end])
    sf.write("eq_test_output.wav", output_audio, SR)
    print("EQ test complete. Output saved to 'eq_test_output.wav'.")

def test_compressor():
    print("\n--- Running Compressor Self-Test ---")
    SR = 44100
    len_noise, len_tone = int(SR * 0.05), int(SR * 0.4)
    snare_noise = (np.random.rand(len_noise)*2-1) * np.linspace(1,0,len_noise)
    t = np.linspace(0,1,len_tone)
    snare_tone = np.sin(2*np.pi*180*t) * np.linspace(1,0,len_tone)**2
    snare_hit = np.concatenate((snare_noise, snare_tone))
    silence = np.zeros(SR//2)
    test_signal = np.concatenate([silence, snare_hit, silence, snare_hit])
    comp = Compressor(SR)
    comp.set_parameters(-20, 8.0, 1, 200, 10.0)
    compressed_signal = comp.process(test_signal)
    sf.write("comp_test_input.wav", test_signal, SR)
    sf.write("comp_test_output.wav", compressed_signal, SR)
    print("Compressor test complete.")

def test_limiter():
    print("\n--- Running Limiter Self-Test ---")
    SR = 44100
    len_noise, len_tone = int(SR * 0.05), int(SR * 0.4)
    snare_noise = (np.random.rand(len_noise)*2-1) * np.linspace(1,0,len_noise)
    t = np.linspace(0,1,len_tone)
    snare_tone = np.sin(2*np.pi*180*t) * np.linspace(1,0,len_tone)**2
    snare_hit = np.concatenate((snare_noise, snare_tone))
    silence = np.zeros(SR//2)
    test_signal = np.concatenate([silence, snare_hit, silence, snare_hit])
    test_signal *= 10**(12 / 20.0)
    limiter = LookaheadLimiter(SR, 1.5, 80, -0.1)
    BUFFER_SIZE = 1024
    num_buffers = len(test_signal) // BUFFER_SIZE
    output_signal = np.zeros_like(test_signal)
    for i in range(num_buffers):
        start, end = i * BUFFER_SIZE, (i + 1) * BUFFER_SIZE
        output_signal[start:end] = limiter.process(test_signal[start:end])
    sf.write("limiter_test_input_clipping.wav", test_signal, SR)
    sf.write("limiter_test_output_limited.wav", output_signal, SR)
    print("Limiter test complete.")

if __name__ == "__main__":
    # test_eq()
    # test_compressor()
    test_limiter()