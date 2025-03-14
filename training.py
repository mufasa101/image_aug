


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Function to train the model on the original (non-augmented) dataset
def train_on_original(model, train_images, train_labels_encoded, test_images, test_labels_encoded):
    
    # Callback to save the model with the best validation accuracy during training
    checkpoint = ModelCheckpoint(
        filepath="original_model.keras",      # Filepath to save the best model
        save_best_only=True,                   # Only save when validation accuracy improves
        monitor="val_accuracy",                # Monitor validation accuracy
        verbose=1                              # Verbose mode for progress messages
    )
    
    # Callback to stop training early if validation loss does not improve for 5 epochs
    early_stopping = EarlyStopping(
        monitor="val_loss",                    # Monitor validation loss
        patience=5,                            # Wait for 5 epochs without improvement before stopping
        restore_best_weights=True              # Restore model weights from the epoch with the best value of the monitored quantity
    )

    # Train the model using the original dataset with a validation split
    history = model.fit(
        train_images, train_labels_encoded,    # Training data and labels
        validation_split=0.2,                    # Reserve 20% of training data for validation
        epochs=20,                             # Maximum number of training epochs
        batch_size=32,                         # Batch size for training
        callbacks=[checkpoint, early_stopping] # Use callbacks for checkpointing and early stopping
    )
    
    # Evaluate the trained model on the separate test dataset
    loss, accuracy = model.evaluate(test_images, test_labels_encoded)
    print(f"✅ Original Model - Test Loss: {loss:.2f}, Test Accuracy: {accuracy:.2f}")
    return history

# Function to train the model on the augmented dataset
def train_on_augmented(model, datagen, train_images, train_labels_encoded, test_images, test_labels_encoded):
    print("🔥 Starting training on the augmented dataset...")
    
    # Callback to save the model with the best validation accuracy during training
    checkpoint = ModelCheckpoint(
        filepath="augmented_model.keras",      # Filepath to save the best model for augmented data
        save_best_only=True,                   # Only save when validation accuracy improves
        monitor="val_accuracy",                # Monitor validation accuracy
        verbose=1                              # Verbose mode for progress messages
    )
    
    # Callback to stop training early if validation loss does not improve for 5 epochs
    early_stopping = EarlyStopping(
        monitor="val_loss",                    # Monitor validation loss
        patience=5,                            # Wait for 5 epochs without improvement before stopping
        restore_best_weights=True              # Restore best model weights upon early stopping
    )

    # Train the model using data generated from image augmentation on the fly
    history = model.fit(
        datagen.flow(train_images, train_labels_encoded, batch_size=32),  # Use augmented data generator
        validation_data=(test_images, test_labels_encoded),               # Use the test dataset for validation
        epochs=20,                                # Maximum number of training epochs
        callbacks=[checkpoint, early_stopping]    # Use callbacks for checkpointing and early stopping
    )
    
    # Evaluate the trained model on the test dataset
    loss, accuracy = model.evaluate(test_images, test_labels_encoded)
    print(f"✅ Augmented Model - Test Loss: {loss:.2f}, Test Accuracy: {accuracy:.2f}")
    return history




# Overall Summary:
# This code defines two separate functions that train and evaluate a Keras model.
# One function (train_on_original) trains the model on the original dataset with a validation split, using callbacks to save the best model and stop early if the validation loss stops improving.
# The other function (train_on_augmented) trains the model using an image augmentation generator to provide dynamically augmented training data, with similar checkpointing and early stopping. 
# Both functions then evaluate the model on a test dataset and print the test loss and accuracy.