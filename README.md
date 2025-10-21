Okay, here's the updated `README.md` reflecting the final project structure, data preprocessing steps, and the refined scripts.

-----

## File: `README.md` (Updated)

````markdown
# AuraCore: Adaptive Real-Time Audio Mastering Engine 🚀

AuraCore is an AI-powered system for real-time audio mastering with adaptive learning capabilities. It analyzes incoming audio, predicts optimal mastering settings using a trained neural network, and applies them through a custom DSP chain. The engine's core innovation is its ability to **learn a user's preferences in real-time** by fine-tuning the AI based on manual adjustments made via a GUI.

This project was developed as an undergraduate thesis with a focus on novel approaches to AI in audio, aiming for patent potential and real-time performance (<50ms latency) on consumer hardware.

---
## ✨ Features

* **Real-Time AI Processing:** Lightweight CNN feature extractor and MLP prediction model optimized for low-latency GPU inference.
* **Full Mastering Chain:** Includes an 8-Band Parametric EQ, a dynamics Compressor, and a Lookahead Limiter with parameter smoothing.
* **Adaptive Learning:** The AI fine-tunes its predictions *live* based on user tweaks via the GUI.
* **DAW Integration:** Connects to DAWs (like FL Studio) using a virtual audio cable.
* **Efficient Data Handling:** Robust, resumable data preprocessing pipeline for large audio datasets.

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
    * Make sure you have PyTorch installed with CUDA support if you have an NVIDIA GPU. See [PyTorch installation instructions](https://pytorch.org/).
    * Install project requirements:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Install Virtual Audio Cable**
    * Download and install VB-CABLE ([https://vb-audio.com/Cable/](https://vb-audio.com/Cable/)). **Run the setup as administrator** and **reboot** your computer.

---
## 🏃 Running AuraCore

Follow these steps in order:

### ### 1. Prepare the Dataset (`preprocess_data.py`)

This script processes your raw audio files into efficient chunks for training and evaluation.

1.  **Download Dataset:** Obtain an audio dataset (e.g., [FMA Small](https://github.com/mdeff/fma)).
2.  **Organize Files:** Create `data/train/` and `data/test/` folders. Place ~80% of your audio files (e.g., `.mp3`) in `data/train/` and ~20% in `data/test/`. *Alternatively, use `prepare_dataset.py` if you have the raw FMA structure.*
3.  **Run Preprocessing:**
    ```bash
    python preprocess_data.py
    ```
    This will create `data/train_chunks_temp/` and `data/test_chunks_temp/` folders containing thousands of `.pt` chunk files. This script is **resumable** – run it again if it gets interrupted.

### ### 2. Create Baseline Models (`create_baseline_models.py`)

This step creates the initial "neutral" AI models.

1.  **Run:**
    ```bash
    python create_baseline_models.py
    ```
    This saves `feature_extractor.pth` and `prediction_model.pth` to the `models/` directory.

### ### 3. Evaluate Baseline Performance (`evaluator.py`)

Test the baseline model's performance on unseen data.

1.  **Run:**
    ```bash
    python evaluator.py --max_samples 50000 # Or adjust the number of samples
    ```
2.  **Record Result:** Note the **"Final Performance Score (Average MSE Loss)"**. This is crucial for your paper.

### ### 4. Run the Live GUI (`auracore_gui.py`)

This is the main application for real-time mastering and adaptive learning.

1.  **Configure FL Studio:** Set FL Studio's audio output device to **`CABLE Input (VB-Audio Virtual Cable)`** (usually via FL Studio ASIO settings). 
2.  **Configure Script Devices:** Run `python auracore_gui.py` once. It will list audio devices. Edit the `auracore_gui.py` script and set the `INPUT_DEVICE` (your virtual cable, e.g., "CABLE Output") and `OUTPUT_DEVICE` (your speakers/headphones) index numbers.
3.  **Launch:**
    ```bash
    python auracore_gui.py
    ```
4.  **Use:** Press play in FL Studio. The audio will be processed by AuraCore. Use the sliders in the GUI to adjust the sound. The AI learns from your tweaks! Click "Reset All to AI" to hear the AI's adapted suggestions.

### ### (Optional) Train a Custom Model (`trainer.py`)

If you want to train the model beyond the baseline (e.g., on a specific genre or style, though our baseline is effective), use the trainer.

1.  **Ensure Preprocessing is Done:** Make sure `data/train_chunks_temp/` exists.
2.  **Configure Epochs (Optional):** Edit `trainer.py` to set `NUM_EPOCHS`.
3.  **Run Training:**
    ```bash
    python trainer.py
    ```
    This script uses **checkpointing** (`models/training_checkpoint.pth`) and is **resumable**. Stop with `Ctrl+C` after an epoch completes, and run the script again to resume.
4.  **Finalize Model:** After training, run:
    ```bash
    python finalize_model.py
    ```
    This saves the final trained models from the checkpoint.
5.  **Re-evaluate:** Run `evaluator.py` again to get the score for your custom-trained model.

---
## 📁 Project Structure

````

AuraCore/
├── data/
│   ├── train/              \# Raw training audio (e.g., .mp3)
│   ├── test/               \# Raw test audio
│   ├── train\_chunks\_temp/  \# Preprocessed training chunks (.pt)
│   └── test\_chunks\_temp/   \# Preprocessed test chunks (.pt)
├── models/               \# Saved model weights (.pth) & checkpoints
├── src/                  \# Core source code
│   ├── ai/               \# AI models (feature\_extractor.py, prediction\_model.py)
│   ├── core/             \# DSP modules (dsp.py)
│   └── ui/               \# (Contains GUI logic if refactored)
├── .gitignore            \# Files ignored by Git
├── auracore\_gui.py       \# Main application script with GUI
├── create\_baseline\_models.py \# Creates initial neutral models
├── evaluator.py          \# Evaluates model performance on test data
├── finalize\_model.py     \# Extracts final models from training checkpoint
├── prepare\_dataset.py    \# (Optional) Helper to split raw FMA data
├── preprocess\_data.py    \# Processes raw audio into chunks
├── README.md             \# This file
├── requirements.txt      \# Python dependencies
└── trainer.py            \# Trains the AI models

```

---
## 💡 Key Innovations (Patent Potential)

1.  **Real-time Adaptive Learning Loop:** The core method where user adjustments in the GUI during live audio playback directly trigger fine-tuning of the prediction model, personalizing the mastering process dynamically.
2.  **Stable Baseline & Efficient Learning:** Training methodology focusing on achieving a neutral baseline, allowing the adaptive loop to effectively capture user preferences without extensive initial training.
3.  **Low-Latency Pipeline:** Integration of GPU-accelerated AI inference and efficient CPU-based DSP designed for real-time (<50ms) operation on consumer hardware.
```
