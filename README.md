````markdown
# AuraCore: Adaptive Real-Time Audio Mastering Engine

AuraCore is an AI-powered system for real-time audio mastering with adaptive learning capabilities. It analyzes incoming audio, predicts optimal mastering settings using a trained neural network, and applies them through a custom DSP chain. The engine's core innovation allows it to learn a user's preferences in real-time by fine-tuning its predictions based on manual adjustments made via a GUI.

Developed as an undergraduate thesis project (ending July 31, 2025), AuraCore focuses on novel approaches to AI in audio with patent potential, targeting low-latency performance (<50ms) on consumer hardware.

---
## 🚀 Features

* **Real-Time AI Processing:** Lightweight CNN feature extractor and MLP prediction model optimized for low-latency GPU inference.
* **Full Mastering Chain:** Includes an 8-Band Parametric EQ, a dynamics Compressor, and a Lookahead Limiter, all designed for artifact-free parameter changes.
* **Adaptive Learning:** The AI fine-tunes its predictions *live* based on user tweaks in the GUI, personalizing the mastering style.
* **DAW Integration:** Connects to DAWs (like FL Studio) using a virtual audio cable.
* **Efficient Data Pipeline:** Pre-processes large datasets into resumable, memory-safe chunks for robust training and evaluation.

---
## 📁 Project Structure

```text
AuraCore/
├── data/
│   ├── train/              # Source audio for training (.mp3, .wav, etc.)
│   ├── test/               # Source audio for testing
│   ├── train_chunks_temp/  # Pre-processed training chunks (.pt files)
│   └── test_chunks_temp/   # Pre-processed test chunks (.pt files)
├── models/
│   ├── feature_extractor.pth # Saved model weights
│   └── prediction_model.pth
├── src/
│   ├── ai/                 # AI model definitions
│   │   ├── feature_extractor.py
│   │   └── prediction_model.py
│   └── core/               # DSP algorithms
│       └── dsp.py
├── .gitignore              # Files ignored by Git
├── auracore_gui.py         # The main GUI application ✨
├── create_baseline_models.py # Creates initial models (run once)
├── evaluator.py            # Evaluates model performance 📊
├── preprocess_data.py      # Pre-processes audio data (run once) ⚙️
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── trainer.py              # Trains the AI models (optional after baseline) 🧠
````

-----

## 🛠️ Setup & Installation

1.  **Clone Repository**

    ```bash
    git clone <your-repo-url>
    cd AuraCore
    ```

2.  **Create Virtual Environment**

    ```bash
    python -m venv aura_env
    # On Windows:
    .\aura_env\Scripts\activate
    # On macOS/Linux:
    # source aura_env/bin/activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

    *(This includes PyTorch, torchaudio, sounddevice, tqdm, etc.)*

4.  **Install Virtual Audio Cable:** Download and install **VB-CABLE** ([https://vb-audio.com/Cable/](https://vb-audio.com/Cable/)). Remember to run the setup **as administrator** and **reboot** your computer.

-----

## 🏃 How to Use AuraCore

### \#\#\# Step 1: Prepare the Dataset (`preprocess_data.py`) - Run Once

This script converts your raw audio files into processed chunks, handling potential errors and making training/evaluation faster and safer.

1.  **Place Data:** Put your `.mp3`, `.wav`, or `.flac` audio files into `data/train/` and `data/test/`.
2.  **Run Pre-processing:**
    ```bash
    python preprocess_data.py
    ```
    *This script is resumable.* If it stops, just run it again. It will create `data/train_chunks_temp/` and `data/test_chunks_temp/`.

### \#\#\# Step 2: Create Baseline Models (`create_baseline_models.py`) - Run Once

This script saves the initial, untrained state of the AI models. This serves as our functional "neutral" baseline, effectively skipping the long initial training phase.

```bash
python create_baseline_models.py
```

This creates the `.pth` files in the `models/` directory.

### \#\#\# Step 3: Evaluate Baseline Performance (`evaluator.py`)

This script tests the baseline model on the unseen test data and calculates its performance score (MSE Loss).

1.  **Run Evaluation:**
    ```bash
    python evaluator.py --max_samples 50000
    ```
    *(Adjust `--max_samples` if needed, 50k is recommended)*.
2.  **Record Score:** Note the "Final Performance Score (Average MSE Loss)" for your paper. A low score confirms the baseline is working correctly.

### \#\#\# Step 4: Run the Live GUI (`auracore_gui.py`)

This is the main application for real-time mastering and adaptive learning.

1.  **Configure Audio Devices:** Run `python auracore_gui.py` once. It will list your audio devices. Edit the `INPUT_DEVICE` (VB-CABLE Output) and `OUTPUT_DEVICE` (your speakers) variables near the top of the script with the correct index numbers.
2.  **Setup DAW:** Configure FL Studio (or your DAW) to use **FL Studio ASIO** and set its **Output** to **CABLE Input (VB-Audio Virtual Cable)**.
3.  **Launch AuraCore:**
    ```bash
    python auracore_gui.py
    ```
4.  **Play Audio in DAW:** Press play in FL Studio. The audio will route through AuraCore.
5.  **Interact & Teach:** Use the sliders in the GUI to adjust the sound. The AI learns from your changes\! Click "Reset All to AI" to see how its suggestions have adapted.

### \#\#\# Step 5 (Optional): Further Training (`trainer.py`)

While the baseline model is functional, you can optionally run the trainer to further refine the AI on the pre-processed data. This was found to be very slow due to I/O bottlenecks in testing.

```bash
# Optional: Run if you want to train beyond the baseline
python trainer.py
```

*(Uses checkpointing, so it can be stopped and resumed)*.

-----

## 💡 Key Innovations (Patent Potential)

1.  **Real-time Adaptive Learning Loop:** The core method where user tweaks in the GUI create live training examples, immediately fine-tuning the AI's prediction model during an active session.
2.  **Efficient Baseline Model:** Demonstrating that even the initial model state achieves near-zero error on the baseline task (neutral processing of mastered audio).
3.  **Artifact-Free DSP:** The custom EQ, Compressor, and Limiter handle rapid parameter changes from the AI without audible clicks or glitches.

<!-- end list -->

```
```
