from tensorflow.keras import layers, models

# This function builds our CNN model (the thing that learns to recognize tire images)
def create_cnn(input_shape, num_classes):
    # Start building the model
   _model =  models.Sequential([
        # First convolutional layer: filters out image features (like edges, textures)
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),  # 32 filters of 3x3 size
        layers.MaxPooling2D((2, 2)),  # Pooling reduces the size to make it faster (picks the best info)

        # Second convolutional layer: digs deeper into the image features
        layers.Conv2D(64, (3, 3), activation='relu'),  # 64 filters, better at details
        layers.MaxPooling2D((2, 2)),  # Pool again to summarize the info

        # Third convolutional layer: even deeper details
        layers.Conv2D(128, (3, 3), activation='relu'),  # More filters for complex features
        layers.MaxPooling2D((2, 2)),  # Reduce size once more

        # Flatten the data so it can be fed into the final decision layers
        layers.Flatten(),

        # Fully connected (dense) layer: mixes everything to make sense of it
        layers.Dense(128, activation='relu'),

        # Dropout layer to prevent the model from "cramming" (overfitting)
        layers.Dropout(0.5),

        # Final output layer: the softmax gives probabilities for each class (e.g., cracked or normal)
        layers.Dense(num_classes, activation='softmax')
    ])

   _model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

   return _model
