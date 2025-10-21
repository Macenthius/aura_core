"""
Real-time audio buffer management for low-latency processing
"""

import numpy as np
from collections import deque

class RealTimeBuffer:
    def __init__(self, buffer_size=1024, channels=2):
        self.buffer_size = buffer_size
        self.channels = channels
        self.buffer = deque(maxlen=buffer_size)
        
    def add_samples(self, samples):
        """Add new audio samples to buffer"""
        self.buffer.extend(samples)
        
    def get_buffer(self):
        """Get current audio buffer for processing"""
        if len(self.buffer) >= self.buffer_size:
            return np.array(list(self.buffer))
        return None
