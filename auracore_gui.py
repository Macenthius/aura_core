# AuraCore/auracore_gui.py

import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np
import torch
import sys
import os
import threading
import time
from collections import deque

# --- Component Imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from ai.feature_extractor import FeatureExtractor
from ai.prediction_model import PredictionModel, scale_parameters
from core.dsp import EightBandEQ, Compressor, LookaheadLimiter

# --- Configuration ---
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
INPUT_DEVICE = 1   # <-- SET YOUR VIRTUAL CABLE DEVICE INDEX
OUTPUT_DEVICE = 5  # <-- SET YOUR SPEAKER DEVICE INDEX
MODEL_PATH = "models/"
FE_MODEL_FILE = os.path.join(MODEL_PATH, "feature_extractor.pth")
PM_MODEL_FILE = os.path.join(MODEL_PATH, "prediction_model.pth")

# --- Shared State & Learning ---
shared_state = {
    "overrides": {},
    "latest_feature_vector": torch.zeros(1, 32),
    "running": True
}
state_lock = threading.Lock()
replay_buffer = deque(maxlen=64)
learning_rate_finetune = 1e-5

# --- AI and DSP Initialization ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = FeatureExtractor().to(device).eval()
prediction_model = PredictionModel().to(device)
if os.path.exists(FE_MODEL_FILE): feature_extractor.load_state_dict(torch.load(FE_MODEL_FILE, map_location=device))
if os.path.exists(PM_MODEL_FILE): prediction_model.load_state_dict(torch.load(PM_MODEL_FILE, map_location=device))
optimizer = torch.optim.Adam(prediction_model.parameters(), lr=learning_rate_finetune)
loss_fn = torch.nn.MSELoss()
eq = EightBandEQ(sample_rate=SAMPLE_RATE)
comp = Compressor(sample_rate=SAMPLE_RATE)
limiter = LookaheadLimiter(sample_rate=SAMPLE_RATE)


class AuraCoreGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AuraCore Control Panel")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.controls = {} # To store sliders

        # --- Create UI Frames ---
        eq_frame = ttk.LabelFrame(root, text="8-Band EQ")
        eq_frame.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.BOTH)
        
        dyn_frame = ttk.Frame(root)
        dyn_frame.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.BOTH)
        comp_frame = ttk.LabelFrame(dyn_frame, text="Compressor")
        comp_frame.pack(fill=tk.BOTH, expand=True)
        limiter_frame = ttk.LabelFrame(dyn_frame, text="Limiter")
        limiter_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # --- EQ Controls (8 vertical sliders) ---
        self.eq_bands = {}
        for i, freq in enumerate(eq.center_freqs):
            label_text = f"{int(freq)} Hz" if freq < 1000 else f"{freq/1000:.1f} kHz"
            band_frame = ttk.Frame(eq_frame)
            band_frame.pack(side=tk.LEFT, padx=5, pady=5)
            ttk.Label(band_frame, text=label_text).pack()
            slider = ttk.Scale(band_frame, from_=15, to=-15, orient=tk.VERTICAL, length=200, command=self._on_slider_change)
            slider.set(0)
            slider.pack()
            self.eq_bands[i] = slider
            
        # --- Compressor Controls ---
        self._create_slider(comp_frame, 'comp_threshold', 'Threshold (dB)', -40, 0, -12)
        self._create_slider(comp_frame, 'comp_ratio', 'Ratio', 1, 10, 3)
        self._create_slider(comp_frame, 'comp_attack', 'Attack (ms)', 1, 100, 5)
        self._create_slider(comp_frame, 'comp_release', 'Release (ms)', 20, 500, 100)
        self._create_slider(comp_frame, 'comp_makeup', 'Makeup (dB)', 0, 15, 0)

        # --- Limiter Controls ---
        self._create_slider(limiter_frame, 'limiter_release', 'Release (ms)', 20, 200, 80)
        self._create_slider(limiter_frame, 'limiter_ceiling', 'Ceiling (dB)', -2, -0.1, -0.1)
        
        # --- Reset Button ---
        reset_button = ttk.Button(dyn_frame, text="Reset All to AI", command=self._reset_to_ai)
        reset_button.pack(pady=10)

    def _create_slider(self, parent, key, label, from_, to, default):
        frame = ttk.Frame(parent)
        frame.pack(padx=5, pady=2, fill=tk.X)
        ttk.Label(frame, text=label, width=12).pack(side=tk.LEFT)
        slider = ttk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL, command=self._on_slider_change)
        slider.set(default)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.controls[key] = slider

    def _on_slider_change(self, event=None):
        """Called when any slider moves. Updates shared state for the audio thread."""
        overrides = {}
        eq_gains = np.zeros(8)
        for i, slider in self.eq_bands.items():
            eq_gains[i] = slider.get()
        overrides[('eq', 'gains')] = eq_gains
        
        for key, slider in self.controls.items():
            module, param = key.split('_', 1)
            overrides[(module, param)] = slider.get()
        
        with state_lock:
            shared_state["overrides"] = overrides

    def _reset_to_ai(self):
        """Clears overrides and resets sliders to neutral positions."""
        with state_lock:
            shared_state["overrides"].clear()
        
        for slider in self.eq_bands.values():
            slider.set(0)
        
        # This is a simplified reset, you could refine default values
        self.controls['comp_threshold'].set(-12); self.controls['comp_ratio'].set(3)
        self.controls['comp_attack'].set(5); self.controls['comp_release'].set(100)
        self.controls['comp_makeup'].set(0)
        self.controls['limiter_release'].set(80); self.controls['limiter_ceiling'].set(-0.1)

    def _on_closing(self):
        """Handles window close event to shut down the audio thread."""
        global shared_state
        print("Closing application...")
        shared_state["running"] = False
        self.root.destroy()

def audio_thread_func():
    # This function is identical to the one in auracore_interactive.py
    # It runs the sounddevice stream and the audio_callback
    def audio_callback(indata, outdata, frames, time, status):
        audio_buffer = indata[:, 0].astype(np.float32)
        input_tensor = torch.from_numpy(audio_buffer).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            feature_vector = feature_extractor(input_tensor)
            raw_params = prediction_model(feature_vector)
        
        with state_lock:
            shared_state["latest_feature_vector"] = feature_vector.detach().clone()
        
        scaled_params = scale_parameters(raw_params)
        with state_lock:
            overrides = shared_state.get("overrides", {})
            if ('eq', 'gains') in overrides:
                scaled_params['eq_gains'] = overrides[('eq', 'gains')]
            for key, val in overrides.items():
                if key[0] != 'eq':
                    scaled_params[f"{key[0]}_{key[1]}"] = val
        
        eq.set_parameters(scaled_params['eq_gains'], scaled_params['eq_qs'])
        comp.set_parameters(scaled_params['comp_threshold'], scaled_params['comp_ratio'], scaled_params['comp_attack'], scaled_params['comp_release'], scaled_params['comp_makeup'])
        limiter.set_parameters(scaled_params['limiter_release'], scaled_params['limiter_ceiling'])

        processed_buffer = limiter.process(comp.process(eq.process(audio_buffer)))
        outdata[:] = processed_buffer.reshape(-1, 1)

    try:
        with sd.Stream(device=(INPUT_DEVICE, OUTPUT_DEVICE), samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE, channels=CHANNELS, dtype='float32', callback=audio_callback):
            while shared_state["running"]:
                time.sleep(0.1)
    except Exception as e:
        print(f"\nAudio thread error: {e}")
    finally:
        shared_state["running"] = False

if __name__ == "__main__":
    if INPUT_DEVICE is None or OUTPUT_DEVICE is None:
        print("--- Please select your audio devices in the script ---")
        print(sd.query_devices())
    else:
        # Start the audio processing in a background thread
        audio_thread = threading.Thread(target=audio_thread_func)
        audio_thread.start()

        # Start the GUI on the main thread
        root = tk.Tk()
        app = AuraCoreGUI(root)
        root.mainloop()

        # Wait for the audio thread to finish after the GUI is closed
        audio_thread.join()
        print("--- Engine Shut Down ---")