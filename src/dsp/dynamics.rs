/// ===========================================================================
/// AURACORE: DYNAMICS PROCESSING (Compressor & Limiter) (Rust)
/// ===========================================================================
///
/// WHAT IS THIS FILE?
/// This file handles "Dynamics", which means it controls the volume of the audio
/// automatically.
///
/// 1. COMPRESSOR: Makes loud parts quieter. This helps "glue" a track together
///    and makes it sound more professional and consistent.
///
/// 2. LIMITER: A "Hard Ceiling" for volume. It ensures the audio NEVER goes above
///    a certain point, preventing digital distortion (clipping).
///
/// HOW IT WORKS (The Envelope):
/// Both tools use an "Envelope Follower". Imagine a line that follows the
/// peaks of the audio waves. We use this line to decide how much to turn
/// the volume down.
/// ===========================================================================
use std::collections::VecDeque;

/// --- COMPRESSOR ---
/// It squashes the dynamic range (the difference between quiet and loud).
pub struct Compressor {
    sample_rate: f32,
    threshold_db: f32, // The level where we start squashing.
    ratio: f32,        // How hard we squash (e.g., 4:1).
    attack_ms: f32,    // How fast we start squashing.
    release_ms: f32,   // How fast we stop squashing.
    makeup_db: f32,    // Volume boost after squashing.

    envelope: f32,      // The current "tracked" volume level.
    alpha_attack: f32,  // Math constant for attack speed.
    alpha_release: f32, // Math constant for release speed.
    makeup_linear: f32, // Makeup gain converted from dB to a multiplier.
}

impl Compressor {
    pub fn new(sample_rate: f32) -> Self {
        let mut comp = Self {
            sample_rate,
            threshold_db: -12.0,
            ratio: 3.0,
            attack_ms: 5.0,
            release_ms: 100.0,
            makeup_db: 0.0,
            envelope: 0.0,
            alpha_attack: 0.0,
            alpha_release: 0.0,
            makeup_linear: 1.0,
        };
        comp.recalc_coeffs();
        comp
    }

    /// MATH: Convert milliseconds/dB into values the computer can use in the processing loop.
    fn recalc_coeffs(&mut self) {
        let attack_samples = self.sample_rate * (self.attack_ms / 1000.0);
        self.alpha_attack = if attack_samples > 0.0 {
            (-1.0 / attack_samples).exp()
        } else {
            0.0
        };

        let release_samples = self.sample_rate * (self.release_ms / 1000.0);
        self.alpha_release = if release_samples > 0.0 {
            (-1.0 / release_samples).exp()
        } else {
            0.0
        };

        // Convert dB (user units) to Multiplier (math units).
        self.makeup_linear = 10.0f32.powf(self.makeup_db / 20.0);
    }

    /// Update settings when the user moves a slider.
    pub fn set_params(
        &mut self,
        threshold: f32,
        ratio: f32,
        attack: f32,
        release: f32,
        makeup: f32,
    ) {
        self.threshold_db = threshold;
        self.ratio = ratio.max(1.0);
        self.attack_ms = attack.max(0.1);
        self.release_ms = release.max(1.0);
        self.makeup_db = makeup;
        self.recalc_coeffs();
    }

    /// PROCESSOR: Runs for every sample.
    pub fn process(&mut self, sample: f32) -> f32 {
        let abs_sample = sample.abs(); // We only care about the "height" of the wave.

        // [1] UPDATE ENVELOPE (The volume follower)
        // If the sound is getting louder, use the attack speed.
        // If getting quieter, use the release speed.
        if abs_sample > self.envelope {
            self.envelope =
                self.alpha_attack * self.envelope + (1.0 - self.alpha_attack) * abs_sample;
        } else {
            self.envelope =
                self.alpha_release * self.envelope + (1.0 - self.alpha_release) * abs_sample;
        }

        // [2] CALCULATE GAIN REDUCTION
        let threshold_linear = 10.0f32.powf(self.threshold_db / 20.0);
        let mut gain_reduction_db = 0.0;

        // If the tracked volume is above the threshold, we need to turn it down.
        if self.envelope > threshold_linear {
            let envelope_db = 20.0 * self.envelope.log10();
            let overshoot = envelope_db - self.threshold_db;
            // The formula for compression: Gain Reduction = (Input - Threshold) * (1 - 1/Ratio)
            gain_reduction_db = overshoot * (1.0 - 1.0 / self.ratio);
        }

        // [3] APPLY CHANGES
        // Convert the reduction back to a multiplier.
        let gain_reduction = 10.0f32.powf(-gain_reduction_db / 20.0);
        // Result = Original Sound * Quieter * Makeup Boost
        sample * gain_reduction * self.makeup_linear
    }
}

/// --- LIMITER ---
/// A specialized compressor with an infinite ratio and "Lookahead".
/// It looks into the "future" to catch peaks before they happen.
pub struct Limiter {
    sample_rate: f32,
    ceiling_db: f32,
    release_ms: f32,

    envelope: f32,
    gain: f32,
    alpha_release: f32,
    ceiling_linear: f32,

    /// LOOKAHEAD BRAIN: We store a few milliseconds of audio in this buffer
    /// so the limiter can see a loud peak coming before it actually plays it.
    delay_buffer: VecDeque<f32>,
    lookahead_samples: usize,
}

impl Limiter {
    pub fn new(sample_rate: f32) -> Self {
        let lookahead_ms = 1.5; // We look 1.5ms into the future.
        let lookahead_samples = (sample_rate * lookahead_ms / 1000.0) as usize;

        let mut lim = Self {
            sample_rate,
            ceiling_db: -0.1,
            release_ms: 50.0,
            envelope: 0.0,
            gain: 1.0,
            alpha_release: 0.0,
            ceiling_linear: 1.0,
            delay_buffer: VecDeque::from(vec![0.0; lookahead_samples]),
            lookahead_samples,
        };
        lim.recalc_coeffs();
        lim
    }

    fn recalc_coeffs(&mut self) {
        let release_samples = self.sample_rate * (self.release_ms / 1000.0);
        self.alpha_release = if release_samples > 0.0 {
            (-1.0 / release_samples).exp()
        } else {
            0.0
        };
        self.ceiling_linear = 10.0f32.powf(self.ceiling_db / 20.0);
    }

    pub fn set_params(&mut self, ceiling: f32, release: f32) {
        self.ceiling_db = ceiling;
        self.release_ms = release;
        self.recalc_coeffs();
    }

    pub fn process(&mut self, sample: f32) -> f32 {
        // [1] STORE AND DELAY
        // We put the current sample in the back and pull an old one from the front.
        // This creates a 1.5ms delay.
        self.delay_buffer.push_back(sample);
        let delayed_sample = self.delay_buffer.pop_front().unwrap_or(0.0);

        // [2] ANALYZE THE "FUTURE"
        // We look at the 'sample' (which hasn't been played yet because it's delayed).
        let abs_sample = sample.abs();
        if abs_sample > self.envelope {
            self.envelope = abs_sample; // Instant attack to catch peaks perfectly.
        } else {
            self.envelope = self.alpha_release * self.envelope;
        }

        // [3] CALCULATE LIMITING GAIN
        let mut target_gain = 1.0;
        if self.envelope > self.ceiling_linear {
            // If the future sound is too loud, we need to turn it down to exactly the ceiling.
            target_gain = self.ceiling_linear / self.envelope;
        }

        // Smooth the gain changes to avoid clicks.
        if target_gain < self.gain {
            self.gain = target_gain; // Quick drop for peaks.
        } else {
            self.gain = self.alpha_release * self.gain + (1.0 - self.alpha_release) * target_gain;
            // Smooth release.
        }

        // [4] APPLY TO THE DELAYED SOUND
        delayed_sample * self.gain
    }
}
