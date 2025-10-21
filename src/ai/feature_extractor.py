# AuraCore/src/ai/feature_extractor.py

import torch
import torch.nn as nn
import time

# --- Configuration ---
INPUT_SAMPLES = 1024  # ~23ms at 44.1kHz, a good buffer size for real-time
N_FEATURES = 32       # The dimensionality of our learned audio representation

class FeatureExtractor(nn.Module):
    """
    A lightweight 1D CNN for real-time audio feature extraction.
    The model is designed to be small and fast for GPU inference.
    """
    def __init__(self, in_channels=1, num_features=N_FEATURES):
        super(FeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(256, num_features)
        )

    def forward(self, x):
        return self.net(x)

def run_performance_test():
    """
    Tests the feature extractor's performance on the available hardware (CPU or GPU).
    """
    print("\n--- AuraCore: Feature Extractor Performance Test ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = FeatureExtractor().to(device)
    model.eval()
    
    use_fp16 = True
    if device.type == 'cuda' and use_fp16:
        model.half()

    batch_size = 1
    dummy_input = torch.randn(batch_size, 1, INPUT_SAMPLES).to(device)
    if device.type == 'cuda' and use_fp16:
        dummy_input = dummy_input.half()

    print("Warming up the GPU...")
    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy_input)

    print("Running performance benchmark...")
    num_runs = 500
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_latency_ms = (total_time / num_runs) * 1000
    print(f"\n--- Performance Results ---")
    print(f"Average Latency: {avg_latency_ms:.4f} ms per buffer")

if __name__ == "__main__":
    run_performance_test()