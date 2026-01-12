use crate::dsp::{
    dynamics::{Compressor, Limiter},
    eq::EightBandEQ,
}; // The sound-changing tools.
use crate::state::SharedState; // Our parameter settings from state.rs.
/// ===========================================================================
/// AURACORE: AUDIO INPUT/OUTPUT SYSTEM (Rust)
/// ===========================================================================
///
/// WHAT IS THIS FILE?
/// This is the "heart" of the real-time engine. It handles:
/// 1. Connecting to your Microphone/Virtual Cable (Input).
/// 2. Connecting to your Speakers/Headphones (Output).
/// 3. Running the DSP (Digital Signal Processing) chain on every single sound sample.
///
/// DATA FLOW:
/// [ Microphone ] -> [ Input Stream ] -> [ Ring Buffer ] -> [ Output Stream ] -> [ Speakers ]
///                                                              |
///                                                      [ DSP Processing ]
///                                           (EQ -> Compressor -> Limiter)
///
/// LINGO FOR BEGINNERS:
/// - Sample: A single point of audio data. Like a pixel in an image, CD quality
///           music has 44,100 samples every second.
/// - Stream: A continuous flow of samples from or to a piece of hardware.
/// - Ring Buffer: A circular "waiting room" for audio samples. It keeps things
///                smooth if the computer gets slightly delayed.
/// ===========================================================================
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait}; // 'cpal' is a library for talking to audio hardware.
use parking_lot::Mutex; // 'Mutex' for safe data sharing.
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
}; // 'AtomicBool' is a thread-safe True/False flag. // A double-ended queue, used here as our buffer.

/// Main structure representing the active audio system.
pub struct AudioSystem {
    /// We keep these "streams" alive here. If we dropped them, the sound would stop.
    _input_stream: cpal::Stream,
    _output_stream: cpal::Stream,
}

impl AudioSystem {
    /// Creates and starts the audio system.
    /// - `state`: The shared settings (EQ, etc.)
    /// - `viz_buffer`: A buffer for the GUI to read for the "visualizer".
    /// - `error_flag`: A flag we set to 'true' if the hardware fails.
    pub fn new(
        state: SharedState,
        viz_buffer: Arc<Mutex<VecDeque<f32>>>,
        error_flag: Arc<AtomicBool>,
    ) -> Result<Self, anyhow::Error> {
        // --- 1. FIND THE HARDWARE ---
        let host = cpal::default_host(); // Get the system audio driver (e.g., ASIO, CoreAudio, WASAPI).

        // Find the default input (mic) and output (speakers) devices.
        let input_device = host
            .default_input_device()
            .ok_or(anyhow::anyhow!("No input device"))?;
        let output_device = host
            .default_output_device()
            .ok_or(anyhow::anyhow!("No output device"))?;

        println!("Input: {}", input_device.name()?);
        println!("Output: {}", output_device.name()?);

        // Get the standard settings for the speakers (Sample Rate, Channels, etc.)
        let config: cpal::StreamConfig = output_device.default_output_config()?.into();
        let sample_rate = config.sample_rate.0 as f32; // Usually 44100 or 48000 Hz.

        // --- 2. INITIALIZE SOUND TOOLS (DSP) ---
        let mut eq = EightBandEQ::new(sample_rate);
        let mut comp = Compressor::new(sample_rate);
        let mut limiter = Limiter::new(sample_rate);

        // --- 3. CREATE THE WAITING ROOM (RING BUFFER) ---
        // This buffer holds 4096 samples between the input and output threads.
        let ring_buffer = Arc::new(Mutex::new(VecDeque::with_capacity(4096)));
        let rb_producer = ring_buffer.clone(); // Input thread "produces" sound.
        let rb_consumer = ring_buffer.clone(); // Output thread "consumes" sound.

        // --- 4. ERROR HANDLING ---
        // These functions run if the audio stream crashes (e.g., someone unplugs their headphones).
        let err_flag_in = error_flag.clone();
        let err_fn_in = move |err| {
            eprintln!("Input stream error: {}", err);
            err_flag_in.store(true, Ordering::SeqCst);
        };

        let err_flag_out = error_flag.clone();
        let err_fn_out = move |err| {
            eprintln!("Output stream error: {}", err);
            err_flag_out.store(true, Ordering::SeqCst);
        };

        // --- 5. THE INPUT STREAM (LISTENER) ---
        // This part listens to the microphone and puts the audio into our Ring Buffer.
        let input_stream = input_device.build_input_stream(
            &config,
            move |data: &[f32], _: &_| {
                let mut rb = rb_producer.lock();
                for &sample in data {
                    if rb.len() < 4096 {
                        rb.push_back(sample);
                    }
                }

                // Also copy the data to the visualizer buffer if we can get the lock.
                // We use 'try_lock' so the audio thread never waits (lags) for the GUI.
                if let Some(mut vb) = viz_buffer.try_lock() {
                    for &sample in data {
                        if vb.len() >= 4096 {
                            vb.pop_front();
                        }
                        vb.push_back(sample);
                    }
                }
            },
            err_fn_in,
            None,
        )?;

        // --- 6. THE OUTPUT STREAM (SPEAKER + PROCESSOR) ---
        // This part takes sound from the Ring Buffer, PROCESSES it, and sends it to your speakers.
        let output_stream = output_device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &_| {
                // [A] UPDATE PARAMETERS
                // Every time a new "chunk" of audio is ready, check if the user moved a slider.
                if let Some(params) = state.params.try_lock() {
                    eq.update_params(&params.eq_gains);
                    comp.set_params(
                        params.comp_threshold,
                        params.comp_ratio,
                        params.comp_attack,
                        params.comp_release,
                        params.comp_makeup,
                    );
                    limiter.set_params(params.limiter_release, params.limiter_ceiling);
                }

                // [B] PROCESS SOUND
                let mut rb = rb_consumer.lock();
                for sample in data.iter_mut() {
                    // Pull the raw sound from the buffer. If empty, just play silence (0.0).
                    let input = rb.pop_front().unwrap_or(0.0);

                    // Run the DSP Chain in order:
                    // 1. EQ (Adjust frequencies)
                    let eq_out = eq.process(input);
                    // 2. Compressor (Control volume peaks)
                    let comp_out = comp.process(eq_out);
                    // 3. Limiter (Final volume safety wall)
                    let limit_out = limiter.process(comp_out);

                    // Send the final result to the speaker hardware.
                    *sample = limit_out;
                }
            },
            err_fn_out,
            None,
        )?;

        // --- 7. START THE ENGINE ---
        input_stream.play()?;
        output_stream.play()?;

        Ok(AudioSystem {
            _input_stream: input_stream,
            _output_stream: output_stream,
        })
    }
}
