from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import random
import numpy as np

train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalize pixel values
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# No augmentation for validation data (just rescale)
val_datagen = ImageDataGenerator(rescale=1.0/255)

def training_generator(train_dir):
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),  # Resize images
        batch_size=32,
        class_mode='categorical'
    )

    return train_generator

def validation_generator(test_dir):
    val_generator = val_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    return val_generator

# Function to show original and augmented images side-by-side (so we compare)
def visualizer(images, labels, generator):
    # Pick 5 random images from the dataset
    indices = random.sample(range(len(images)), 5)

    for idx in indices:
        original = images[idx]  # Get the original image
        augmented = generator.flow(np.expand_dims(original, 0), batch_size=1)[0][0]  # Make an augmented version

        plt.figure(figsize=(8, 4))  # Make the graph wider so it's easy to see
        # Show the original image
        plt.subplot(1, 2, 1)  # First column
        plt.imshow(original)
        plt.title(f"Original - {labels[idx]}", color="blue")  # Give it a blue title
        plt.axis("off")  # Remove those annoying axis ticks

        # Show the augmented image
        plt.subplot(1, 2, 2)  # Second column
        plt.imshow(augmented)
        plt.title("Augmented", color="green")  # Give it a green title
        plt.axis("off")  # Remove the axis ticks again

        plt.tight_layout()  # Make sure there's no overlap in the images
    plt.show()
