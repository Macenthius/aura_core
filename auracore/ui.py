"""
===========================================================================
AURACORE: THE USER INTERFACE (Python)
===========================================================================

WHAT IS THIS FILE?
This is the "Face" of the application. It creates the window, the sliders,
 and the visual feedback. It uses a library called 'customtkinter' for a 
 modern, dark-themed look.

HOW IT COMMUNICATES:
1. CUSTOMER MOVES SLIDER -> ui.py calls 'engine.update_params()'.
2. RUST ENGINE -> Updates the sound in real-time.
3. UI ASKS ENGINE -> "Give me the latest audio data for the visualizer."
4. UI UPDATES -> Shows the volume levels (dB) on the screen.

LINGO FOR BEGINNERS:
- Mainloop: The core loop of a GUI app that keeps the window open and responsive.
- Callback: A function that runs when you interact with something (like moving a slider).
- Thread: A separate "lane" for tasks. The AI runs in its own thread so it doesn't 
          make the GUI freeze.
===========================================================================
"""

import customtkinter as ctk  # A modern version of the standard Python UI library (Tkinter).
import threading           # Allows us to run tasks in the background.
import time
import numpy as np         # Used for doing math on audio data arrays.
from . import auracore_engine  # This is the RUST library we compiled!
from .ai_worker import AIWorker # Our AI "Brain" module.

# Set the "Look and Feel" of the app.
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class AuraCoreApp(ctk.CTk):
    """The main Application class."""
    
    def __init__(self):
        super().__init__()
        
        # Window configuration.
        self.title("AuraCore: Ironclad Edition")
        self.geometry("800x600")
        
        # --- 1. START THE RUST ENGINE ---
        # This connects Python to our high-performance Rust audio code.
        try:
            self.engine = auracore_engine.Engine()
            print("✅ Rust Engine Initialized")
        except Exception as e:
            print(f"❌ Failed to init Rust Engine: {e}")
            self.engine = None

        # --- 2. START THE AI WORKER ---
        # The AI Worker will watch the audio and try to suggest better settings.
        if self.engine:
            self.ai_worker = AIWorker(self.engine)
            self.ai_worker.start()
        
        # --- 3. BUILD THE INTERFACE ---
        self._setup_ui()
        
        # --- 4. START THE METERS ---
        # This starts a repeating timer that updates the volume display.
        self._start_metering()

    def _setup_ui(self):
        """Creates the labels, frames, and sliders."""
        
        # The big title at the top.
        self.header = ctk.CTkLabel(self, text="AURACORE", font=("Roboto", 24, "bold"))
        self.header.pack(pady=20)
        
        # A simple status text to show if things are working.
        self.status_label = ctk.CTkLabel(self, text="Status: Active (Rust + CUDA)", text_color="green")
        self.status_label.pack()
        
        # VISUALIZER AREA
        self.meter_frame = ctk.CTkFrame(self, height=100)
        self.meter_frame.pack(fill="x", padx=20, pady=20)
        self.meter_label = ctk.CTkLabel(self.meter_frame, text="[ Visualizer Placeholder ]")
        self.meter_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # CONTROLS AREA
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(self.controls_frame, text="Manual Overrides").pack(pady=10)
        
        # SLIDER: Compressor Threshold
        # When you move it, it calls 'self._on_param_change'.
        self.threshold_slider = ctk.CTkSlider(
            self.controls_frame, 
            from_=-60, to=0, 
            command=self._on_param_change
        )
        self.threshold_slider.set(-12)
        self.threshold_slider.pack(pady=10)
        ctk.CTkLabel(self.controls_frame, text="Threshold").pack()

    def _on_param_change(self, value):
        """Runs whenever a slider is moved."""
        if self.engine:
            # We tell the Rust engine to update its settings.
            # We pass: (EQ Gains, Comp Threshold, Ratio, Attack, Release, Makeup, Limiter Ceiling, Limiter Release)
            self.engine.update_params(
                [0.0]*8,            # 8 EQ bands (all flat for now)
                float(value),       # The new threshold from the slider.
                3.0, 5.0, 100.0, 0.0, # Standard compressor defaults.
                -0.1, 50.0          # Limiter defaults.
            )

    def _start_metering(self):
        """Sets up a loop that runs every 50 milliseconds."""
        self.after(50, self._update_meters)

    def _update_meters(self):
        """Asks the Rust engine for data and updates the screen."""
        if self.engine:
            # First, check if the engine has crashed.
            if not self.engine.check_status():
                self.status_label.configure(text="Status: CRITICAL ERROR (Audio Device Lost)", text_color="red")
                self.meter_label.configure(text="❌ ENGINE DIED")
                return

            # Get a tiny chunk of the latest audio samples from Rust.
            data = self.engine.get_viz_data() 
            if data:
                # MATH: RMS (Root Mean Square) is a way to calculate the average "Loudness".
                rms = np.sqrt(np.mean(np.array(data)**2))
                # Convert the raw number into Decibels (dB).
                db = 20 * np.log10(rms + 1e-9)
                self.meter_label.configure(text=f"Level: {db:.1f} dB")
        
        # Call this function again in 50ms (Creates the loop).
        self.after(50, self._update_meters)

    def on_closing(self):
        """Runs when you click the 'X' to close the window."""
        print("Shutting down...")
        if hasattr(self, 'ai_worker'):
            self.ai_worker.stop() # Tell the AI to stop thinking.
        self.destroy()

def run_app():
    """Entry point to start the program."""
    app = AuraCoreApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
