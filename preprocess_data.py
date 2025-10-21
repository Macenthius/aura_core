# AuraCore/preprocess_data.py

import os
import torch
import torchaudio

# --- Configuration ---
SAMPLE_RATE = 44100
CHUNK_SAMPLES = 1024
DATASETS_TO_PROCESS = {
    "train": {"source_dir": "data/train", "output_dir": "data/train_chunks_temp"},
    "test": {"source_dir": "data/test", "output_dir": "data/test_chunks_temp"}
}

def process_dataset(name, config):
    """
    Processes each audio file into its own chunk file in the output directory.
    This script is resumable and memory-efficient.
    """
    source_dir, output_dir = config["source_dir"], config["output_dir"]
    print(f"\n--- Processing '{name}' Dataset ---")
    
    if not os.path.isdir(source_dir):
        print(f"Source directory not found: '{source_dir}'. Skipping.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    file_paths = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith(('.wav', '.flac', '.mp3'))]

    print(f"Found {len(file_paths)} audio files. Starting/resuming processing...")
    
    for i, path in enumerate(file_paths):
        base_name = os.path.basename(path)
        chunk_output_path = os.path.join(output_dir, base_name + '.pt')

        if os.path.exists(chunk_output_path):
            continue
            
        if (i + 1) % 50 == 0:
            print(f"Processing file {i+1}/{len(file_paths)}: {base_name}")

        try:
            audio, sr = torchaudio.load(path)
            if sr != SAMPLE_RATE:
                audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
            audio = torch.mean(audio, dim=0, keepdim=True)
            
            num_chunks = audio.shape[1] // CHUNK_SAMPLES
            file_chunks = [audio[:, (j*CHUNK_SAMPLES):((j+1)*CHUNK_SAMPLES)] for j in range(num_chunks)]
            
            if file_chunks:
                torch.save(file_chunks, chunk_output_path)
        except Exception:
            print(f"Warning: Skipping corrupted file: {base_name}")
            continue

    print(f"--- '{name}' dataset pre-processing complete. Chunks saved in '{output_dir}'. ---")

if __name__ == "__main__":
    for name, config in DATASETS_TO_PROCESS.items():
        process_dataset(name, config)