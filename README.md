# AuraCore: Adaptive Real-Time Audio Mastering Engine

AuraCore is an AI-powered system for real-time audio mastering with adaptive learning capabilities. It analyzes incoming audio, predicts optimal mastering settings, and applies them through a custom DSP chain. The engine can learn a user's preferences in real-time by fine-tuning its neural network based on manual adjustments.

This project was developed as an undergraduate thesis with a focus on novel approaches to AI in audio with patent potential.

---
## 🚀 Features

* **Real-Time AI Processing:** A lightweight CNN feature extractor and MLP prediction model optimized for low-latency inference on a GPU (<50ms).
* **Full Mastering Chain:** Includes an 8-Band Parametric EQ, a dynamics Compressor, and a Lookahead Limiter.
* **Adaptive Learning:** The AI fine-tunes its predictions in real-time based on user-provided parameter tweaks.
* **DAW Integration:** Connects to any DAW (like FL Studio) using a virtual audio cable for seamless workflow.

---
## 🛠️ Setup & Installation

1.  **Clone Repository**
    ```bash
    git clone <your-repo-url>
    cd AuraCore
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---
## 🏃 How to Use AuraCore

There are three primary ways to run the engine.

### ### 1. Training the AI (`trainer.py`)

Before first use, the AI models must be trained.

1.  **Add Data:** Place high-quality, professionally mastered `.wav` or `.flac` files into the `data/reference_tracks/` directory.
2.  **Run Trainer:**
    ```bash
    python trainer.py
    ```
    This will train the models and save the weights to the `models/` directory.

### ### 2. Live Mastering from a DAW (`realtime_runner.py`)

This script processes audio from a DAW like FL Studio in real-time.

1.  **Setup Audio Routing:** Install and configure a virtual audio cable (e.g., VB-CABLE) to route audio from your DAW to the script.
2.  **Configure Devices:** Run `python realtime_runner.py` once to list your audio devices. Edit the script to set the `INPUT_DEVICE` (your virtual cable) and `OUTPUT_DEVICE` (your speakers/headphones) index numbers.
3.  **Run:**
    ```bash
    python realtime_runner.py
    ```
4.  Press play in your DAW. The mastered audio will be played through your selected output device.

### ### 3. Interactive Learning Mode (`auracore_interactive.py`)

This is the most powerful mode, allowing you to tweak the AI and have it learn from you.

1.  **Setup & Configure:** Follow the same setup steps as for `realtime_runner.py`.
2.  **Run:**
    ```bash
    python auracore_interactive.py
    ```
3.  **Control the AI:** While audio is playing from your DAW, type commands into the terminal to adjust the sound. The AI will learn from every command.
    * `eq <band> <gain>` (e.g., `eq 4 2.5`)
    * `comp <param> <value>` (e.g., `comp ratio 4.0`)
    * `reset` (clears your manual overrides)
    * `exit` (stops the engine)

---
## 💡 Key Innovations (Patent Potential)

1.  **Adaptive Real-Time Learning:** The core method of using user overrides during a live session to create training pairs (`feature_vector`, `target_parameters`) and immediately backpropagate to fine-tune the mastering model.
2.  **Predictive Parameter Smoothing:** The DSP chain is designed to handle rapid, AI-driven parameter changes without producing audio artifacts like clicks or zipper noise.
3.  **Efficient Mobile GPU Processing:** The entire AI and DSP pipeline is designed to run with low latency on consumer-grade laptop hardware (RTX 4050).