# AuraCore/evaluator.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
from typing import List, Tuple, Optional

# --- Path and Component Imports ---
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)

# Now that the path is set, import project components
from src.ai.feature_extractor import FeatureExtractor
from src.ai.prediction_model import PredictionModel, OUTPUT_PARAMS


class DirectoryAudioDataset(Dataset):
    """A Dataset that reads pre-processed chunk files from a directory."""
    def __init__(self, chunk_dir: str, max_samples: Optional[int] = None):
        self.chunk_dir = chunk_dir
        self.index_map: List[Tuple[str, int]] = []

        print("Indexing chunks for evaluation...")
        chunk_files = sorted([os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith('.pt')])

        for file_path in tqdm(chunk_files, desc="Indexing files"):
            try:
                num_chunks_in_file = len(torch.load(file_path, map_location='cpu'))
                for i in range(num_chunks_in_file):
                    self.index_map.append((file_path, i))
                    if max_samples and len(self.index_map) >= max_samples:
                        break
                if max_samples and len(self.index_map) >= max_samples:
                    break
            except (IOError, RuntimeError) as e:
                print(f"Warning: Could not load or process {file_path}. Skipping. Error: {e}")

        if not self.index_map:
            raise RuntimeError("No valid data chunks found in the specified directory.")
            
        print(f"Test dataset created with {len(self.index_map)} total chunks.")

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_path, chunk_idx = self.index_map[idx]
        # This might be slow if I/O is a bottleneck. For evaluation, it's often acceptable.
        chunks_in_file = torch.load(file_path)
        return chunks_in_file[chunk_idx]

def evaluate(args: argparse.Namespace):
    """Runs the model evaluation process based on provided arguments."""
    print("--- AuraCore: Model Performance Evaluator ---")
    
    if not os.path.isdir(args.data_dir):
        print(f"Error: Pre-processed test data directory not found at '{args.data_dir}'")
        print("Please run 'python preprocess_data.py' first.")
        return

    # 1. Setup Device, Dataset, and DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        test_dataset = DirectoryAudioDataset(args.data_dir, max_samples=args.max_samples)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    except RuntimeError as e:
        print(f"Error creating dataset: {e}")
        return

    # 2. Load Trained Models
    print("Loading trained models...")
    fe_model_file = os.path.join(args.model_dir, "feature_extractor.pth")
    pm_model_file = os.path.join(args.model_dir, "prediction_model.pth")
    
    feature_extractor = FeatureExtractor().to(device)
    prediction_model = PredictionModel().to(device)
    
    try:
        feature_extractor.load_state_dict(torch.load(fe_model_file, map_location=device))
        prediction_model.load_state_dict(torch.load(pm_model_file, map_location=device))
    except FileNotFoundError as e:
        print(f"Error: Model file not found. {e}")
        print("Please ensure baseline or trained models exist in the specified directory.")
        return
    
    feature_extractor.eval()
    prediction_model.eval()

    # 3. Initialize Loss Function and Target
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    
    print(f"\nRunning evaluation on {len(test_dataset)} test samples...")
    # 4. Evaluation Loop
    with torch.no_grad():
        for target_audio_batch in tqdm(test_dataloader, desc="Evaluating Batches"):
            target_audio_batch = target_audio_batch.to(device)
            
            # The target for a neutral model is always zero output
            target_neutral_params = torch.zeros(target_audio_batch.size(0), OUTPUT_PARAMS).to(device)

            features = feature_extractor(target_audio_batch)
            raw_params = prediction_model(features)
            
            loss = loss_fn(raw_params, target_neutral_params)
            total_loss += loss.item()
            
    # 5. Report Final Score
    average_loss = total_loss / len(test_dataloader)
    print("\n--- Evaluation Complete ---")
    print(f"Final Performance Score (Average MSE Loss): {average_loss:.6f}")
    print("(A lower score indicates the AI is closer to the ideal neutral baseline)")

def main():
    parser = argparse.ArgumentParser(description="AuraCore Model Performance Evaluator")
    
    parser.add_argument('--data_dir', type=str, default=os.path.join(PROJ_ROOT, 'data', 'test_chunks_temp'),
                        help='Directory containing pre-processed test data chunks.')
    parser.add_argument('--model_dir', type=str, default=os.path.join(PROJ_ROOT, 'models'),
                        help='Directory where trained models are stored.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation.')
    parser.add_argument('--max_samples', type=int, default=50000,
                        help='Maximum number of audio chunks to evaluate.')

    args = parser.parse_args()
    evaluate(args)

if __name__ == "__main__":
    main()