from os.path import join
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from data_loader import load_images
from augmentation_visualizer import visualizer, train_datagen

# Example usage (assuming you have a folder called `TireTextures`):
_parent_folder = join(Path(__file__).parent, "data", "TireTextures")
_train_images, _train_labels, _test_images, _test_labels = load_images(_parent_folder)

# Use LabelEncoder to turn the labels (e.g., "cracked", "normal") into numbers
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(_train_labels)  # Training labels as numbers
test_labels_encoded = label_encoder.transform(_test_labels)  # Testing labels as numbers

# Show what we have
print(f"Classes: {label_encoder.classes_}")  # The different labels in the dataset
print(f"Training Data Shape: {_train_images.shape}")  # Number of training images and size
print(f"Testing Data Shape: {_test_images.shape}")  # Number of testing images and size

# Example usage (assuming your `train_images` and `train_labels` are already loaded):
# generator = training_generator(join(_parent_folder, "training_data"))
visualizer(_train_images, _train_labels, train_datagen)
