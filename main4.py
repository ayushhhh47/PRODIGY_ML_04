import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# Function to load images from the Leap Motion hand gesture dataset
def load_images_from_dir(dataset_dir, image_size=(100, 100)):
    images = []
    labels = []

    # Iterate over subject folders (00, 01, ..., 09)
    for subject_folder in sorted(os.listdir(dataset_dir)):
        subject_dir = os.path.join(dataset_dir, subject_folder)

        # Iterate over gesture subfolders (01_palm, 02_l, ..., 10_down)
        for gesture_folder in sorted(os.listdir(subject_dir)):
            gesture_dir = os.path.join(subject_dir, gesture_folder)

            # Iterate over image files in the gesture subfolder
            for filename in sorted(os.listdir(gesture_dir)):
                filepath = os.path.join(gesture_dir, filename)

                try:
                    # Load image in grayscale and resize
                    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        image = cv2.resize(image, image_size)
                        images.append(image)
                        labels.append(gesture_folder)  # Use gesture folder name as label
                except Exception as e:
                    print(f"Error processing image '{filepath}': {e}")

    return images, labels


def main():
    # Directory containing the Leap Motion hand gesture dataset
    dataset_dir = r'leapGestRecog'

    # Load images and labels from the dataset directory
    images, labels = load_images_from_dir(dataset_dir)

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Check if any images were loaded
    if len(images) == 0:
        print("No images loaded. Please check the dataset directory.")
        return

    # Encode gesture labels using LabelEncoder
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

    # Flatten the image data for SVM input
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    # Train an SVM classifier with a linear kernel
    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(X_train_flat, y_train)

    # Predict on the validation set
    y_pred = svm_classifier.predict(X_val_flat)

    # Evaluate the model
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.2f}")

    # Generate classification report
    class_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names))


if __name__ == "__main__":
    main()
