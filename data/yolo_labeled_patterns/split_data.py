import os
import shutil
import random
from pathlib import Path

def split_dataset(src_images, src_labels, train_ratio=0.80, val_ratio=0.15):
    """
    Split dataset into train, validation and test sets
    
    Args:
        src_images: source images directory
        src_labels: source labels directory
        train_ratio: ratio of training set (default: 0.75)
        val_ratio: ratio of validation set (default: 0.20)
        test_ratio will be the remaining (0.05)
    """
    # Create necessary directories
    for path in ['data/train/images', 'data/train/labels', 
                'data/val/images', 'data/val/labels',
                'data/test/images', 'data/test/labels']:
        Path(path).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(src_images) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Randomly shuffle the files
    random.shuffle(image_files)
    
    # Calculate split points
    num_train = int(len(image_files) * train_ratio)
    num_val = int(len(image_files) * val_ratio)
    
    # Split into train, validation and test sets
    train_files = image_files[:num_train]
    val_files = image_files[num_train:num_train + num_val]
    test_files = image_files[num_train + num_val:]
    
    # Function to copy files
    def copy_files(files, src_img, src_lbl, dst_img, dst_lbl):
        for f in files:
            # Copy image
            shutil.copy2(
                os.path.join(src_img, f),
                os.path.join(dst_img, f)
            )
            
            # Copy corresponding label (assuming same name, .txt extension)
            label_file = os.path.splitext(f)[0] + '.txt'
            if os.path.exists(os.path.join(src_lbl, label_file)):
                shutil.copy2(
                    os.path.join(src_lbl, label_file),
                    os.path.join(dst_lbl, label_file)
                )
    
    # Copy files for each set
    copy_files(train_files, src_images, src_labels, 
              'data/train/images', 'data/train/labels')
    copy_files(val_files, src_images, src_labels, 
              'data/val/images', 'data/val/labels')
    copy_files(test_files, src_images, src_labels, 
              'data/test/images', 'data/test/labels')
    
    # Print statistics
    print(f"Total images: {len(image_files)}")
    print(f"Training images: {len(train_files)} ({len(train_files)/len(image_files)*100:.1f}%)")
    print(f"Validation images: {len(val_files)} ({len(val_files)/len(image_files)*100:.1f}%)")
    print(f"Testing images: {len(test_files)} ({len(test_files)/len(image_files)*100:.1f}%)")

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    RANDOM_SEED = 42  # RANDOM SEED >> 可以修改这个值来得到不同的随机分配
    print(f"\nUsing random seed: {RANDOM_SEED}")
    random.seed(RANDOM_SEED)
    
    # Source directories
    src_images = "dataset/images"
    src_labels = "dataset/labels"
    
    # Split dataset
    split_dataset(src_images, src_labels)
    
    # Create dataset.yaml
    yaml_content =  """ path: ./data
                        train: train/images
                        val: val/images
                        test: test/images  # Added test set
                        nc: 1  # number of classes (modify as needed)
                        names: ['your_class']  # class names (modify as needed)
                    """
    
    with open('dataset.yaml', 'w') as f:
        f.write(yaml_content)