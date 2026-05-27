import os
import glob
import random
import shutil
import csv
from collections import defaultdict
from pathlib import Path

def generate_train_test_split(
    source_dir: str = "my_sample_large",
    output_dir: str = "src/data/dataset_split",
    train_size_per_class: int = 3000,
    copy_files: bool = True
):
    """
    Separates the samples in `source_dir` into test and train sets.
    Allocates `train_size_per_class` samples of each culture for training,
    and the rest for testing.
    
    Saves the splits into a folder inside src (`output_dir`).
    """
    # Try to find the project root (where my_sample_large is)
    if not os.path.exists(source_dir):
        # Maybe we are inside src/dataframe
        possible_root = Path(__file__).resolve().parent.parent.parent
        source_dir_path = possible_root / source_dir
        output_dir_path = possible_root / output_dir
    else:
        source_dir_path = Path(source_dir)
        output_dir_path = Path(output_dir)

    os.makedirs(output_dir_path, exist_ok=True)
    
    if not source_dir_path.exists():
        print(f"Directory {source_dir_path} not found.")
        return None
        
    print(f"Reading samples from {source_dir_path}...")
    
    culture_samples = defaultdict(list)
    
    for culture_dir in os.listdir(source_dir_path):
        full_culture_dir = source_dir_path / culture_dir
        if not full_culture_dir.is_dir():
            continue
            
        culture_name = culture_dir.replace("arquivos_kml_", "")
        kml_files = glob.glob(str(full_culture_dir / "*.kml"))
        
        for kml in kml_files:
            culture_samples[culture_name].append({
                "culture": culture_name,
                "file_path": kml,
                "file_name": os.path.basename(kml)
            })
            
    if not culture_samples:
        print("No samples found.")
        return []

    train_list = []
    test_list = []
    
    # Set seed for reproducibility
    random.seed(42)
    
    print("Separating samples...")
    for culture, samples in culture_samples.items():
        n_samples = len(samples)
        # Shuffle in place
        random.shuffle(samples)
        
        n_train = min(train_size_per_class, n_samples)
        
        for sample in samples[:n_train]:
            sample['split'] = 'train'
            train_list.append(sample)
            
        for sample in samples[n_train:]:
            sample['split'] = 'test'
            test_list.append(sample)
            
    final_list = train_list + test_list
    
    csv_path = output_dir_path / "split_info.csv"
    
    # Save CSV using standard library
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if final_list:
            writer = csv.DictWriter(f, fieldnames=["culture", "file_path", "file_name", "split"])
            writer.writeheader()
            writer.writerows(final_list)
            
    print(f"Split metadata saved to {csv_path}")
    print(f"Total Train samples: {len(train_list)}")
    print(f"Total Test samples: {len(test_list)}")
    
    if copy_files:
        train_dir = output_dir_path / "train"
        test_dir = output_dir_path / "test"
        
        # Pre-create all necessary directories once to avoid doing it per-file
        for split in ['train', 'test']:
            for culture in culture_samples.keys():
                dest_folder = output_dir_path / split / f"arquivos_kml_{culture}"
                os.makedirs(dest_folder, exist_ok=True)
        
        print(f"Copying files to {train_dir} and {test_dir} in batches...")
        total_files = len(final_list)
        batch_size = 500
        
        # Copy in batches 
        for i in range(0, total_files, batch_size):
            batch = final_list[i:i+batch_size]
            for row in batch:
                dest_folder = output_dir_path / row['split'] / f"arquivos_kml_{row['culture']}"
                dest_file = dest_folder / row['file_name']
                
                if not os.path.exists(dest_file):
                    shutil.copy2(row['file_path'], dest_file)
            
            # Progress output
            print(f"Processed batch {(i // batch_size) + 1}: {min(i + batch_size, total_files)}/{total_files} files copied.")
            
        print("All files copied successfully!")
        
    return final_list

if __name__ == "__main__":
    # Execute the method when running the file directly
    generate_train_test_split()