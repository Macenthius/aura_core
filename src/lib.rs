use parking_lot::Mutex;
/// ===========================================================================
/// AURACORE: THE BRIDGE (Rust to Python)
/// ===========================================================================
///
/// WHAT IS THIS FILE?
/// This is the "Translator". Python doesn't speak "Rust", and Rust doesn't
/// speak "Python". This file creates a `PyModule` which allows Python to
/// create an object called `Engine`.
///
/// HOW IT WORKS:
/// 1. We define a Rust `struct` (like a class) called `Engine`.
/// 2. we mark it with `#[pyclass]` so Python can see it.
/// 3. When Python calls `Engine()`, it triggers the `new` function here.
/// ===========================================================================
use pyo3::prelude::*; // Pull in the Python-bridge tools.
use std::collections::VecDeque;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

// Pull in the other Rust files we wrote.
mod audio;
mod dsp;
mod state;

use crate::audio::AudioSystem;
use crate::state::{AudioParams, SharedState};

/// The main Engine class that Python interacts with.
#[pyclass]
pub struct Engine {
    state: SharedState,
    #[allow(dead_code)]
    audio_system: AudioSystem, // We keep the system held here so it doesn't stop.
    viz_buffer: Arc<Mutex<VecDeque<f32>>>,
    error_flag: Arc<AtomicBool>,
}

#[pymethods]
impl Engine {
    /// This is called when you write `engine = auracore_engine.Engine()` in Python.
    #[new]
    pub fn new() -> PyResult<Self> {
        let state = SharedState::new();
        let viz_buffer = Arc::new(Mutex::new(VecDeque::with_capacity(4096)));
        let error_flag = Arc::new(AtomicBool::new(false));

        // Start the actual audio hardware using the logic in audio.rs
        let audio_system = AudioSystem::new(state.clone(), viz_buffer.clone(), error_flag.clone())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Engine {
            state,
            audio_system,
            viz_buffer,
            error_flag,
        })
    }

    /// PYTHON: `engine.update_params(...)`
    /// This function translates a list of numbers from Python into the
    /// Rust `AudioParams` structure.
    pub fn update_params(
        &self,
        eq_gains: [f32; 8],
        c_thresh: f32,
        c_ratio: f32,
        c_attack: f32,
        c_release: f32,
        c_makeup: f32,
        l_ceiling: f32,
        l_release: f32,
    ) {
        // Lock the parameters so we can safely change them.
        let mut params = self.state.params.lock();
        params.eq_gains = eq_gains;
        params.comp_threshold = c_thresh;
        params.comp_ratio = c_ratio;
        params.comp_attack = c_attack;
        params.comp_release = c_release;
        params.comp_makeup = c_makeup;
        params.limiter_ceiling = l_ceiling;
        params.limiter_release = l_release;
        // When the 'params' lock goes out of scope, the Audio Thread will see the new values!
    }

    /// PYTHON: `engine.get_viz_data()`
    /// Gives Python the latest chunk of audio samples to draw the visualizer.
    pub fn get_viz_data(&self) -> Vec<f32> {
        let mut vb = self.viz_buffer.lock();
        let data: Vec<f32> = vb.iter().cloned().collect();
        vb.clear(); // Clear it after reading so we don't show the same sound twice.
        data
    }

    /// PYTHON: `engine.check_status()`
    /// Returns False if the audio system has crashed.
    pub fn check_status(&self) -> bool {
        // '!'(Not) because a true flag means there IS an error.
        !self.error_flag.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// This function defines the Python module itself.
#[pymodule]
fn auracore_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Engine>()?;
    Ok(())
}
