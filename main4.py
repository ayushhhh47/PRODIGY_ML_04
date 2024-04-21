import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image

# Load and preprocess dataset
data_dir = 'path/to/dataset'
gesture_classes = sorted(os.listdir(data_dir))
num_classes = len(gesture_classes)

images = []
labels = []

for gesture_index, gesture_class in enumerate(gesture_classes):
    gesture_dir = os.path.join(data_dir, gesture_class)
    for img_name in os.listdir(gesture_dir):
        img_path = os.path.join(gesture_dir, img_name)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize((64, 64))  # Resize image
        img_array = np.array(img) / 255.0  # Normalize pixel values
        images.append(img_array)
        labels.append(gesture_index)

images = np.array(images)
labels = np.array(labels)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

# Evaluate model
test_loss, test_acc = model.evaluate(X_val, y_val)
print(f"Test Accuracy: {test_acc}")

# Save model for future use
model.save('hand_gesture_model.h5')

# Perform inference on new data
# Load model and use it for prediction
