/// ===========================================================================
/// AURACORE: 8-BAND PARAMETRIC EQ (Rust)
/// ===========================================================================
///
/// WHAT IS THIS FILE?
/// This handles frequency adjustment (Bass, Mid, Treble, etc.).
/// We use something called a "Biquad Filter". Think of it as a small mathematical
/// machine that takes a sound sample, looks at the previous samples, and
/// calculates a new value based on "Coefficients" (settings).
///
/// HOW IT WORKS:
/// 1. We have 8 of these "machines" (Filters) lined up in a row.
/// 2. Each machine is tuned to a "Center Frequency" (e.g., 60Hz for sub-bass).
/// 3. As sound passes through, the machine boosts or cuts that frequency.
///
/// LINGO FOR BEGINNERS:
/// - Hertz (Hz): The measurement of frequency. Low numbers = Bass, High = Treble.
/// - Decibels (dB): The measurement of volume change. +dB = LOUDER, -dB = QUITER.
/// - Q (Quality Factor): How "wide" the frequency boost is. Narrow = surgical, Wide = musical.
/// ===========================================================================
use std::f32::consts::PI;

/// These are the "Settings" for one filter band.
/// They control the math of the EQ.
#[derive(Clone, Copy, Debug)]
pub struct BiquadCoeffs {
    b0: f32,
    b1: f32,
    b2: f32, // Feed-forward coefficients (input side)
    a1: f32,
    a2: f32, // Feed-back coefficients (output side)
}

/// This keeps track of the "Memory" of the filter.
/// To calculate the current sample, the filter needs to know what the previous samples were.
#[derive(Clone, Debug)]
pub struct BiquadState {
    z1: f32, // Memory slot 1
    z2: f32, // Memory slot 2
}

impl BiquadState {
    pub fn new() -> Self {
        Self { z1: 0.0, z2: 0.0 }
    }
}

/// The main EQ structure containing 8 bands.
pub struct EightBandEQ {
    /// Each band has its own settings (Coeffs) and own memory (State).
    filters: Vec<(BiquadCoeffs, BiquadState)>,
    sample_rate: f32,
    /// The specific frequencies we are controlling.
    center_freqs: [f32; 8],
}

impl EightBandEQ {
    /// Creates a new EQ system.
    pub fn new(sample_rate: f32) -> Self {
        // Standard mastering frequencies from Sub-Bass to Air.
        let center_freqs = [60.0, 150.0, 400.0, 1000.0, 2400.0, 6000.0, 10000.0, 15000.0];
        let mut eq = Self {
            filters: Vec::with_capacity(8),
            sample_rate,
            center_freqs,
        };

        // Initialize all 8 bands to "Flat" (0.0 dB change).
        for &fc in &center_freqs {
            let coeffs = Self::calc_peaking(0.0, 1.0, fc, sample_rate);
            eq.filters.push((coeffs, BiquadState::new()));
        }

        eq
    }

    /// MATH ALERT: This calculates the coefficients for a Peaking/Bell EQ filter.
    /// This is standard DSP math (Audio EQ Cookbook).
    fn calc_peaking(gain_db: f32, q: f32, fc: f32, sr: f32) -> BiquadCoeffs {
        let w0 = 2.0 * PI * fc / sr;
        let alpha = w0.sin() / (2.0 * q);
        let a = 10.0f32.powf(gain_db / 40.0);

        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * w0.cos();
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * w0.cos();
        let a2 = 1.0 - alpha / a;

        // We divide everything by 'a0' to normalize the filter.
        BiquadCoeffs {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Updates the EQ settings based on new dB values from the user.
    pub fn update_params(&mut self, gains_db: &[f32; 8]) {
        for (i, &gain) in gains_db.iter().enumerate() {
            // We use a fixed Q (width) of 1.414 (standard musical width).
            self.filters[i].0 =
                Self::calc_peaking(gain, 1.414, self.center_freqs[i], self.sample_rate);
        }
    }

    /// PROCESSOR: This is the actual math loop that runs 44,100+ times per second.
    pub fn process(&mut self, sample: f32) -> f32 {
        let mut out = sample;

        // Pass the sound through each of the 8 filters consecutively.
        for (coeffs, state) in &mut self.filters {
            let input = out;

            // MATH: Transposed Direct Form II
            // This is a very efficient way to calculate a filter.
            // It uses the current input and the "Memory" (z1, z2) from previous steps.
            out = coeffs.b0 * input + state.z1;
            state.z1 = coeffs.b1 * input - coeffs.a1 * out + state.z2;
            state.z2 = coeffs.b2 * input - coeffs.a2 * out;
        }
        out
    }
}
