import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns  # For colorful heatmaps

# Function to load images dynamically based on file structure
def load_images(parent_folder):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    print("🚀 Starting image loading...")
    for subfolder in ['training_data', 'testing_data']:
        subfolder_path = os.path.join(parent_folder, subfolder)
        for class_folder in os.listdir(subfolder_path):
            class_path = os.path.join(subfolder_path, class_folder)
            if os.path.isdir(class_path):
                for filename in tqdm(os.listdir(class_path), desc=f"🔍 Loading '{subfolder}/{class_folder}'"):
                    file_path = os.path.join(class_path, filename)
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        try:
                            img = Image.open(file_path).convert('RGB')
                            img = img.resize((128, 128))  # Resize to (128x128)
                            img_array = np.array(img) / 255.0  # Normalize pixel values
                            if img_array.shape == (128, 128, 3):  # Ensure correct shape
                                if subfolder == 'training_data':
                                    train_images.append(img_array)
                                    train_labels.append(class_folder)
                                elif subfolder == 'testing_data':
                                    test_images.append(img_array)
                                    test_labels.append(class_folder)
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
    print("🎉 Image loading complete!")
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)

# Load the dataset
parent_folder = '/content/Tire Textures'
if os.path.exists(parent_folder):
    # Load and process images
    train_images, train_labels, test_images, test_labels = load_images(parent_folder)

    # Encode labels
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)

    print(f"\n📊 Classes: {label_encoder.classes_}")
    print(f"Training Data: {train_images.shape}, Labels: {train_labels_encoded.shape}")
    print(f"Testing Data: {test_images.shape}, Labels: {test_labels_encoded.shape}")

    # Visualize random samples
    print("\n🎨 Displaying sample images...")
    unique_labels = label_encoder.classes_
    for label in unique_labels:
        indices = np.where(np.array(train_labels) == label)[0]
        sample_indices = random.sample(list(indices), min(3, len(indices)))
        plt.figure(figsize=(10, 3))
        for i, idx in enumerate(sample_indices):
            plt.subplot(1, 3, i + 1)
            plt.imshow(train_images[idx])
            plt.title(f"Class: {label}", color='blue')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    # Define the CNN architecture
    def create_cnn(input_shape, num_classes):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),  # Dropout for regularization
            layers.Dense(num_classes, activation='softmax')
        ])
        return model

    # Step 1: Train CNN on original dataset
    print("\n🤖 Training on original dataset...")
    model_original = create_cnn((128, 128, 3), len(np.unique(train_labels_encoded)))
    model_original.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    original_history = model_original.fit(
        train_images, train_labels_encoded,
        epochs=20,
        validation_split=0.2,
        batch_size=32,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(filepath='best_original_model.keras', save_best_only=True, monitor='val_accuracy', verbose=1)
        ]
    )

    # Evaluate on test data (original dataset)
    test_loss_original, test_acc_original = model_original.evaluate(test_images, test_labels_encoded)
    print(f"Original Dataset - Test Loss: {test_loss_original:.2f}, Test Accuracy: {test_acc_original:.2f}")

    # Step 2: Train CNN on augmented dataset
    print("\n🔧 Applying data augmentation...")
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    print("\n🤖 Training on augmented dataset...")
    model_augmented = create_cnn((128, 128, 3), len(np.unique(train_labels_encoded)))
    model_augmented.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    augmented_history = model_augmented.fit(
        datagen.flow(train_images, train_labels_encoded, batch_size=32),
        epochs=20,
        validation_data=(test_images, test_labels_encoded),
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(filepath='best_augmented_model.keras', save_best_only=True, monitor='val_accuracy', verbose=1)
        ]
    )

    # Evaluate on test data (augmented dataset)
    test_loss_augmented, test_acc_augmented = model_augmented.evaluate(test_images, test_labels_encoded)
    print(f"Augmented Dataset - Test Loss: {test_loss_augmented:.2f}, Test Accuracy: {test_acc_augmented:.2f}")

    # Step 3: Compare performance
    print("\n📊 Comparing performance...")
    plt.figure(figsize=(12, 6))
    plt.plot(original_history.history['val_accuracy'], label='Original Dataset', color='blue')
    plt.plot(augmented_history.history['val_accuracy'], label='Augmented Dataset', color='green')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(original_history.history['val_loss'], label='Original Dataset', color='blue')
    plt.plot(augmented_history.history['val_loss'], label='Augmented Dataset', color='green')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.show()

    # Confusion Matrix for Augmented Model
    print("\n📉 Generating confusion matrix for augmented model...")
    predictions_augmented = model_augmented.predict(test_images)
    predicted_labels = np.argmax(predictions_augmented, axis=1)
    cm = confusion_matrix(test_labels_encoded, predicted_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix - Augmented Model", fontsize=16, color='darkred')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

else:
    print(f"❌ Folder '{parent_folder}' not found. Please check the path.")
