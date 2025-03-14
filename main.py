from os.path import join
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from data_loader import load_images
from augmentation_visualizer import visualizer, train_datagen
from training import train_on_original, train_on_augmented
from performance_analysis import (
    compare_validation_loss, compare_validation_accuracy, show_confusion_matrix, confusion_matrix
)
from cnn_model import create_cnn


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
# visualizer(_train_images, _train_labels, train_datagen)

model = create_cnn((128, 128, 3), 2)  # Build the CNN
print(model.summary()) # Show us the layers we just created

original_history = train_on_original(model, _train_images, train_labels_encoded, _test_images, test_labels_encoded)
augmented_history= train_on_augmented(model, train_datagen, _train_images, train_labels_encoded, _test_images, test_labels_encoded)

# Performance analysis
compare_validation_accuracy(original_history, augmented_history)
compare_validation_loss(original_history, augmented_history)
show_confusion_matrix(test_labels_encoded, predictions=None, label_encoder=label_encoder)
