import os
import shutil
from tqdm import tqdm
import glob
import numpy as np
import yaml

def sort_interleaved_folders(root_dir):
    """
    Goes through train, and test folders to separate 
    sat.jpg and mask.png into their own subdirectories.
    """
    splits = ['train']
    
    for split in splits:
        split_path = os.path.join(root_dir, split)
        
        if not os.path.exists(split_path):
            print(f"Skipping {split}: Folder not found.")
            continue

        # Define destination paths
        img_dir = os.path.join(split_path, 'images')
        mask_dir = os.path.join(split_path, 'masks')

        # Create subfolders
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        # List all files in the split folder
        files = os.listdir(split_path)
        
        print(f"Processing {split} split...")
        for f in tqdm(files):
            src_path = os.path.join(split_path, f)
            
            # Skip if it's already a directory
            if os.path.isdir(src_path):
                continue

            if f.endswith('_sat.jpg'):
                shutil.move(src_path, os.path.join(img_dir, f))
            elif f.endswith('_mask.png'):
                shutil.move(src_path, os.path.join(mask_dir, f))

    print("\nData separation complete!")

def create_three_way_split(train_dir, val_dir, test_dir, val_ratio=0.1, test_ratio=0.1):
    # Setup directories
    for d in [val_dir, test_dir]:
        os.makedirs(os.path.join(d, 'images'), exist_ok=True)
        os.makedirs(os.path.join(d, 'masks'), exist_ok=True)

    # Get all labeled images
    images = sorted(glob.glob(os.path.join(train_dir, 'images', "*.jpg")))
    np.random.shuffle(images)

    num_test = int(len(images) * test_ratio)
    num_val = int(len(images) * val_ratio)

    # Logic to move files
    splits = [
        (images[:num_test], test_dir, "Test"),
        (images[num_test:num_test+num_val], val_dir, "Val")
    ]

    for file_list, target_dir, name in splits:
        print(f"Moving {len(file_list)} pairs to {name} set...")
        for img_path in tqdm(file_list):
            img_name = os.path.basename(img_path)
            mask_name = img_name.replace('_sat.jpg', '_mask.png')
            mask_path = os.path.join(train_dir, 'masks', mask_name)

            if os.path.exists(mask_path):
                shutil.move(img_path, os.path.join(target_dir, 'images', img_name))
                shutil.move(mask_path, os.path.join(target_dir, 'masks', mask_name))

if __name__ == "__main__":

    # 1. Load the Main Config
    with open("configs/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    
    DATA_ROOT = config['data']['DATA_ROOT']
    sort_interleaved_folders(DATA_ROOT)
    create_three_way_split(train_dir=os.path.join(DATA_ROOT,'train'), val_dir=os.path.join(DATA_ROOT,'val'), test_dir=os.path.join(DATA_ROOT,'test'))
