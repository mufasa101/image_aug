import os

import numpy as np
from PIL import Image
from tqdm import tqdm

# Function to load the images
def load_images(parent_folder):
    # These are the lists where we will store the images and their labels
    train_images, train_labels = [], []
    test_images, test_labels = [], []

    # Start loading images from the folders
    print("🚀 Loading Images...")
    for subfolder in ['training_data', 'testing_data']:  # Check both training and testing folders
        subfolder_path = os.path.join(parent_folder, subfolder)  # Get the path to the folder
        for class_folder in os.listdir(subfolder_path):  # Check each class (e.g., cracked/normal)
            class_path = os.path.join(subfolder_path, class_folder)  # Path to the class folder
            if os.path.isdir(class_path):  # Only proceed if it's a folder
                # Go through each image in the class folder
                for file in tqdm(os.listdir(class_path), desc=f"Processing {subfolder}/{class_folder}"):
                    file_path = os.path.join(class_path, file)  # Get the full image path
                    # Check if it's an image file we can use
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        try:
                            # Open and prepare the image
                            img = Image.open(file_path).convert('RGB').resize((128, 128))  # Fix to RGB and resize
                            img_array = np.array(img) / 255.0  # Normalize the pixel values (0 to 1)
                            if subfolder == 'training_data':  # If it's a training image, add to training
                                train_images.append(img_array)
                                train_labels.append(class_folder)  # Save the label
                            elif subfolder == 'testing_data':  # If it's for testing, add to testing
                                test_images.append(img_array)
                                test_labels.append(class_folder)  # Save the label
                        except Exception as e:  # Catch any errors (e.g., corrupted image file)
                            print(f"Haiya! Problem with image {file_path}: {e}")

    # Return the images and their labels for both training and testing
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)
