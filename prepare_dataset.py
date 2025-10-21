# AuraCore/prepare_dataset.py

import os
import shutil
import glob

# --- CONFIGURATION ---
# 1. IMPORTANT: Set this to the path where you extracted the fma_small folders
#    (the directory that contains the '000', '001', etc. folders)
FMA_SOURCE_DIR = "D:\Main\Project Polymath\Dataset\\fma_small"  # <-- CHANGE THIS PATH

# 2. Define where the prepared data will go
PROJECT_DATA_DIR = "data"
TRAIN_DIR = os.path.join(PROJECT_DATA_DIR, "train")
TEST_DIR = os.path.join(PROJECT_DATA_DIR, "test")

# 3. Define the train/test split (e.g., 80% for training)
#    The FMA folders go up to '155'. 80% is roughly up to folder '124'.
SPLIT_FOLDER_NUMBER = 124

def main():
    """
    Copies audio files from the FMA source directory into train and test sets.
    """
    print("--- Preparing FMA Dataset for AuraCore ---")

    # Verify that the source directory exists
    if not os.path.isdir(FMA_SOURCE_DIR):
        print(f"Error: FMA source directory not found at '{FMA_SOURCE_DIR}'")
        print("Please update the FMA_SOURCE_DIR variable in this script.")
        return

    # Create destination directories if they don't exist
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # Get all subdirectories (e.g., '000', '001', ...)
    fma_folders = sorted([d for d in os.listdir(FMA_SOURCE_DIR) if os.path.isdir(os.path.join(FMA_SOURCE_DIR, d))])

    train_count, test_count = 0, 0

    for folder_name in fma_folders:
        try:
            folder_num = int(folder_name)
            source_folder_path = os.path.join(FMA_SOURCE_DIR, folder_name)
            
            # Decide if this folder is for training or testing
            if folder_num <= SPLIT_FOLDER_NUMBER:
                destination_folder = TRAIN_DIR
                print(f"Copying files from '{folder_name}' to TRAIN set...")
            else:
                destination_folder = TEST_DIR
                print(f"Copying files from '{folder_name}' to TEST set...")

            # Find all .mp3 files and copy them
            for audio_file in glob.glob(os.path.join(source_folder_path, '*.mp3')):
                shutil.copy(audio_file, destination_folder)
                if folder_num <= SPLIT_FOLDER_NUMBER:
                    train_count += 1
                else:
                    test_count += 1
        except ValueError:
            # Ignore folders that aren't numbers (like checksums)
            continue
        except Exception as e:
            print(f"An error occurred processing folder '{folder_name}': {e}")
            
    print("\n--- Dataset Preparation Complete ---")
    print(f"Copied {train_count} files to '{TRAIN_DIR}'")
    print(f"Copied {test_count} files to '{TEST_DIR}'")

if __name__ == "__main__":
    main()