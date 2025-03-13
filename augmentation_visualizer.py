from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import random
import  numpy as np


# Function to show original and augmented images side-by-side (so we compare)
def visualize_augmentations(images, labels):

    # Set up the different ways to make the images look "different" (augmentation techniques)
    generator = ImageDataGenerator(
        rotation_range=30,  # Rotate the image by up to 30 degrees, just a little twist!
        width_shift_range=0.2,  # Move the image left or right by 20% of its width
        height_shift_range=0.2,  # Move the image up or down by 20% of its height
        shear_range=0.2,  # Add a slight slant to the image
        zoom_range=0.3,  # Zoom in or out by 30%
        horizontal_flip=True,  # Flip the image like a mirror (left to right)
        brightness_range=[0.8, 1.2]  # Make the image slightly brighter or darker
    )

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


def augment_images(images, labels):
    pass


def random_brightness(images, labels):
    pass


def rotate(images, labels):
    pass


def horizontal_flip(images, labels):
    pass


def vertical_flip(images, labels):
    pass


def random_contrast(images, labels):
    pass