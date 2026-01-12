/// ===========================================================================
/// AURACORE: SHARED STATE MANAGEMENT (Rust)
/// ===========================================================================
/// 
/// WHAT IS THIS FILE?
/// This file defines the "Source of Truth" for the audio engine parameters.
/// In a real-time system, the GUI (Python) and the Audio Engine (Rust) need
/// to talk to each other. This file provides the structure that holds the
/// settings (like EQ gain or Compression threshold) and ensures they can be
/// shared safely between different parts of the program.
///
/// DATA FLOW:
/// 1. User moves a slider in the Python GUI.
/// 2. Python calls a function in the Rust library.
/// 3. Rust updates the `AudioParams` inside `SharedState`.
/// 4. The Audio Processing thread reads these updated values to change the sound.
/// ===========================================================================

use parking_lot::Mutex; // A 'Mutex' (Mutual Exclusion) ensures only one thread can change data at a time.
use std::sync::Arc;    // 'Arc' (Atomic Reference Counter) allows multiple parts of the program to "own" the same data.

/// This struct contains all the "knobs and dials" for our mastering engine.
/// We use `f32` (32-bit floating point numbers) because they are the standard for audio data.
#[derive(Clone, Debug)]
pub struct AudioParams {
    /// EQ Gains: An array of 8 numbers representing the volume boost/cut (in decibels) 
    /// for each of the 8 frequency bands.
    pub eq_gains: [f32; 8],
    
    /// --- COMPRESSOR SETTINGS ---
    /// Threshold: The volume level (in dB) where the compressor starts working.
    pub comp_threshold: f32,
    /// Ratio: How much the volume is turned down once it crosses the threshold (e.g., 3.0 means 3:1).
    pub comp_ratio: f32,
    /// Attack: How fast (in milliseconds) the compressor reacts to a loud peak.
    pub comp_attack: f32,
    /// Release: How fast (in milliseconds) the compressor stops working after the volume drops.
    pub comp_release: f32,
    /// Makeup: Extra volume added after compression to "make up" for the lost loudness.
    pub comp_makeup: f32,
    
    /// --- LIMITER SETTINGS ---
    /// Ceiling: The absolute maximum volume level (usually just below 0.0 dB to prevent digital "clipping").
    pub limiter_ceiling: f32,
    /// Release: How fast the limiter lets go of the audio signal.
    pub limiter_release: f32,
}

/// 'Default' trait provides a starting point for our settings.
/// When the app starts, these are the values everything is set to.
impl Default for AudioParams {
    fn default() -> Self {
        Self {
            eq_gains: [0.0; 8],         // Start with a "Flat" EQ (no boost or cut).
            comp_threshold: -12.0,      // Sensible default for a master bus.
            comp_ratio: 3.0,
            comp_attack: 5.0,
            comp_release: 100.0,
            comp_makeup: 0.0,
            limiter_ceiling: -0.1,      // -0.1 dB is a standard "safe" ceiling for audio.
            limiter_release: 50.0,
        }
    }
}

/// This is the container that actually gets passed around the app.
/// It wraps the `AudioParams` in an `Arc` (for sharing) and a `Mutex` (for safe editing).
#[derive(Clone)]
pub struct SharedState {
    /// The params are thread-safe and shareable.
    pub params: Arc<Mutex<AudioParams>>,
}

impl SharedState {
    /// Create a new, fresh state with default settings.
    pub fn new() -> Self {
        Self {
            params: Arc::new(Mutex::new(AudioParams::default())),
        }
    }
}
