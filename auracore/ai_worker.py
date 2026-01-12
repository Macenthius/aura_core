"""
===========================================================================
AURACORE: THE AI BRAIN (Python)
===========================================================================

WHAT IS THIS FILE?
This is the "Brain" of AuraCore. While the Rust engine handles the sound,
this Python worker looks at the audio and decides which settings (EQ, 
Compression, etc.) would make the music sound best.

HOW THE MODEL WORKS:
1. INPUT: A small chunk of audio (e.g., 4096 samples).
2. BRAIN (Neural Network): A complex mathematical model (found in ai/model.py)
   processes the audio to find patterns (is it too bassy? is it too quiet?).
3. OUTPUT: It predicts the best sliders values.
4. ACTION: It sends these values to the Rust engine to apply them.

LINGO FOR BEGINNERS:
- Inference: The process of using a trained AI to make a prediction.
- Tensor: A fancy word for a "multidimensional array" of numbers. AI models
           use Tensors to perform math.
- GPU (CUDA): If you have an NVIDIA graphics card, the AI runs on it to be 
              super fast. Otherwise, it uses the CPU.
===========================================================================
"""

import torch             # The main library for Deep Learning (AI).
import numpy as np        # Used for math on arrays.
import time
import threading         # Used to run the Brain in the background.
from . import auracore_engine
from .ai.model import HybridModel # The actual architecture of our neural network.

class AIWorker:
    """A background worker that runs the AI 'Brain' periodically."""
    
    def __init__(self, engine: auracore_engine.Engine):
        self.engine = engine # A reference to the Rust audio engine.
        self.running = False
        self.thread = None
        
        # --- 1. PICK THE BEST HARDWARE ---
        # We try to use 'cuda' (NVIDIA GPU). If not available, we use 'cpu'.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"AI Worker using device: {self.device}")
        
        # --- 2. LOAD THE MODEL ---
        # We create the model and move it to our chosen hardware.
        self.model = HybridModel().to(self.device)
        self.model.eval() # Tell the model we are "using" it, not "training" it.
        
        # --- 3. LOAD THE TRAINED MEMORY (Weights) ---
        # If we have a file with a pretrained 'brain', we load it.
        try:
            self.model.load_state_dict(torch.load("auracore/ai/checkpoints/hybrid_brain_v1.pth", map_location=self.device))
            print("🧠 Loaded trained brain!")
        except:
            print("⚠️ No trained brain found. Using initialized weights (random).")
        
    def start(self):
        """Starts the AI thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stops the AI thread."""
        self.running = False
        if self.thread:
            self.thread.join()

    def _run_loop(self):
        """The main loop where the AI thinks and acts."""
        print("🧠 AI Worker started...")
        
        while self.running:
            # --- 1. WAIT ---
            # We don't want to run the AI 10,000 times a second (it would melt the computer).
            # Once every 0.1 seconds (10 times a sec) is enough.
            time.sleep(0.1) 
            
            # --- 2. PERFORM INFERENCE (Thinking) ---
            # We tell Torch not to worry about "learning" right now (no_grad).
            with torch.no_grad():
                # [MOCK] In a full version, we'd grab REAL audio from Rust here.
                # For now, we simulate audio with random numbers.
                # Input format: [Batch=1, Channels=1, audio_length=4096]
                dummy_audio = torch.randn(1, 1, 4096).to(self.device)
                
                # Run the math!
                outputs, _ = self.model(dummy_audio)
                
                # --- 3. EXTRACT RESULTS ---
                # We pull the results out of the AI's 'Tensor' format into 'Lists'.
                
                # EQ: 8 boost/cut values.
                eq_gains = outputs['eq_gains'].cpu().numpy()[0, 0].tolist()
                
                # COMPRESSOR: Threshold, Ratio, Attack, Release, Makeup.
                comp = outputs['comp_params'].cpu().numpy()[0, 0]
                c_thresh, c_ratio, c_attack, c_release, c_makeup = comp
                
                # LIMITER: Ceiling, Release.
                limit = outputs['limit_params'].cpu().numpy()[0, 0]
                l_ceiling, l_release = limit
            
            # --- 4. TELL RUST TO CHANGE THE SOUND ---
            # Now that the AI has decided on the "perfect" settings,
            # we push them into the Audio Engine.
            self.engine.update_params(
                eq_gains,
                float(c_thresh), float(c_ratio), float(c_attack), float(c_release), float(c_makeup),
                float(l_ceiling), float(l_release)
            )
