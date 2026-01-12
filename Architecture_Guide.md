# AuraCore: Project Road Map (0% to 100%)

Welcome to the internal workings of **AuraCore**, an Adaptive Real-Time Audio Mastering Engine. This guide explains how every piece fits together.

---

## 1. The Bird's Eye View
AuraCore is a **Hybrid** application. It uses two languages for different purposes:
- **Rust (The Muscles):** Handles the intense math of audio processing and talking to your hardware. It is extremely fast and can process 44,100 samples per second without breaking a sweat.
- **Python (The Brains & Face):** Handles the AI (Neural Networks) and the GUI (Sliders and Windows). Python is great for AI libraries like PyTorch and building pretty interfaces.

---

## 2. How Data Moves (The Life of a Sample)

1.  **CAPTURE:** The Rust Engine (`audio.rs`) listens to your microphone or virtual cable.
2.  **BUFFERING:** Samples are stored in a "Ring Buffer."
3.  **DSP CHAIN:** Every sample passes through three "Math Machines" in Rust (`src/dsp/`):
    - **EQ:** Adjusts frequencies (Bass/Treble).
    - **Compressor:** Smooths out volume levels.
    - **Limiter:** Prevents loud peaks from distorting.
4.  **PLAYBACK:** The processed audio is sent to your speakers.
5.  **AI LOOP:** In the background, Python (`ai_worker.py`) grabs a chunk of audio, feeds it to the Neural Network (`model.py`), and asks: *"Are these settings good?"*
6.  **UPDATE:** The AI sends new "Secret Settings" back to Rust. Rust updates its internal dials instantly.

---

## 3. Directory Guide

### 🦀 Rust Side (`src/`)
- `lib.rs`: The "Front Desk." It handles Python's requests to start the engine or change settings.
- `audio.rs`: The "Sound Engineer." It talks to the audio hardware and runs the main loop.
- `state.rs`: The "Shared Clipboard." It holds the current settings (EQ, Comp, etc.) so both Python and Rust can see them.
- `dsp/`: The "Machine Shop." Contains the math for the EQ, Compressor, and Limiter.

### 🐍 Python Side (`auracore/`)
- `ui.py`: The "Remote Control." The window you see on your screen.
- `ai_worker.py`: The "Researcher." A background thread that runs the AI.
- `ai/model.py`: The "Neural Network." The actual architecture of the AI's brain.

---

## 4. Key Concepts for Beginners

### Threading
Audio processing MUST never stop. If it halts for even 1 millisecond, you hear a "pop" or "glitch." Because of this, the **Audio Thread** runs in its own "fast lane." The **GUI** and **AI** run in slower "side lanes."

### Mutual Exclusion (Mutex)
Because multiple threads are looking at the same settings (Python changing them, Rust reading them), we use a **Mutex**. It's like a bathroom key: only one thread can "enter" and change the settings at a time to prevent data corruption.

### The Decibel (dB)
Unlike regular numbers, decibels are logarithmic. 0dB is the absolute loudest digital sound. -infinity is silence. Mastering usually happens in the -18dB to -1dB range.

---

## 5. How to Run & Build
1.  **Build Rust:** `cargo build` creates the `auracore_engine.pyd` file.
2.  **Install Python Deps:** `pip install torch numpy customtkinter`
3.  **Run App:** `python -m auracore.ui` (or run it via a main script).

---

Congratulations! You now understand the entire flow of AuraCore.
