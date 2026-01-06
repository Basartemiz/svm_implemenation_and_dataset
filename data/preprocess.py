#we shall get preprocessed data from fruit_images folder and return X and y arrays 



import numpy as np
import os
from PIL import Image


def load_data(path,image_size=(64,64)):
    fruit_folders = ['apple', 'banana', 'orange', 'cucumber', 'carrot']
    X = []
    y = []
    label_map = {fruit: idx for idx, fruit in enumerate(fruit_folders)}
    
    # Valid image file extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif'}
    
    # Files to skip (system files)
    skip_files = {'.DS_Store', 'Thumbs.db'}

    for fruit in fruit_folders:
        folder_path = os.path.join(path, fruit)
        if not os.path.exists(folder_path):
            continue
        for img_name in os.listdir(folder_path):
            # Skip hidden/system files
            if img_name.startswith('.') or img_name in skip_files:
                continue
            
            # Check if file has valid image extension
            _, ext = os.path.splitext(img_name.lower())
            if ext not in valid_extensions:
                continue
            
            img_path = os.path.join(folder_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(image_size)
                img_array = np.array(img).flatten() / 255.0  # Normalize pixel values
                X.append(img_array)
                y.append(label_map[fruit])
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    X = np.array(X)
    y = np.array(y)
    return X, y