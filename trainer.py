# AuraCore/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import os
import sys

# ... (Component Imports are the same) ...
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from ai.feature_extractor import FeatureExtractor
from ai.prediction_model import PredictionModel, OUTPUT_PARAMS

# --- Training Configuration ---
TRAINING_DATA_DIR = "data/train_chunks_temp"
MODEL_SAVE_PATH = "models/"
CHECKPOINT_FILE = os.path.join(MODEL_SAVE_PATH, "training_checkpoint.pth") # <-- ADDED
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10

# ... (DirectoryAudioDataset class is the same) ...
class DirectoryAudioDataset(Dataset):
    def __init__(self, chunk_dir):
        self.chunk_dir = chunk_dir; self.chunk_files = [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith('.pt')]
        print("Indexing chunks..."); self.index_map = []
        for file_path in self.chunk_files:
            num_chunks_in_file = len(torch.load(file_path));
            for i in range(num_chunks_in_file): self.index_map.append((file_path, i))
        print(f"Dataset created with {len(self.index_map)} total chunks.")
    def __len__(self): return len(self.index_map)
    def __getitem__(self, idx):
        file_path, chunk_idx = self.index_map[idx]; chunks_in_file = torch.load(file_path); return chunks_in_file[chunk_idx]

def main():
    print("--- AuraCore: AI Model Trainer ---")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    if not os.path.isdir(TRAINING_DATA_DIR):
        print(f"Error: Pre-processed data directory not found at '{TRAINING_DATA_DIR}'"); print("Please run 'python preprocess_data.py' first."); return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DirectoryAudioDataset(TRAINING_DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    feature_extractor = FeatureExtractor().to(device)
    prediction_model = PredictionModel().to(device)
    all_params = list(feature_extractor.parameters()) + list(prediction_model.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    target_neutral_params = torch.zeros(BATCH_SIZE, OUTPUT_PARAMS).to(device)
    
    start_epoch = 0 # <-- ADDED: Default start epoch

    # --- ADDED: Load from checkpoint if it exists ---
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Resuming training from checkpoint: {CHECKPOINT_FILE}")
        checkpoint = torch.load(CHECKPOINT_FILE)
        feature_extractor.load_state_dict(checkpoint['extractor_state_dict'])
        prediction_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    # ---------------------------------------------
    
    print(f"\nStarting training on {device} from epoch {start_epoch + 1}/{NUM_EPOCHS}...")
    for epoch in range(start_epoch, NUM_EPOCHS): # <-- MODIFIED: Use start_epoch
        total_loss = 0
        for i, target_audio_batch in enumerate(dataloader):
            target_audio_batch = target_audio_batch.to(device)
            optimizer.zero_grad()
            features = feature_extractor(target_audio_batch)
            raw_params = prediction_model(features)
            current_batch_size = raw_params.shape[0]
            loss = loss_fn(raw_params, target_neutral_params[:current_batch_size])
            loss.backward(); optimizer.step(); total_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.6f}')

        print(f"--- Epoch {epoch+1} Complete - Average Loss: {total_loss / len(dataloader):.6f} ---")
        
        # --- ADDED: Save checkpoint at the end of each epoch ---
        print("Saving checkpoint...")
        torch.save({
            'epoch': epoch,
            'extractor_state_dict': feature_extractor.state_dict(),
            'model_state_dict': prediction_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_FILE)
        # ----------------------------------------------------

    print("Training finished. Saving final models...")
    torch.save(feature_extractor.state_dict(), os.path.join(MODEL_SAVE_PATH, "feature_extractor.pth"))
    torch.save(prediction_model.state_dict(), os.path.join(MODEL_SAVE_PATH, "prediction_model.pth"))
    print("Models saved successfully.")

if __name__ == "__main__":
    main()