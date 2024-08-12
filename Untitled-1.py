import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Load and preprocess the dataset
def load_custom_dataset(train_path, label_path, img_size=(32, 32)):
    images = []
    labels = []
    label_map = {}
    label_counter = 0

    for label_name in os.listdir(train_path):
        train_dir = os.path.join(train_path, label_name)
        label_dir = os.path.join(label_path, label_name)  # Corresponding label directory
        
        if os.path.isdir(train_dir) and os.path.isdir(label_dir):
            if label_name not in label_map:
                label_map[label_name] = label_counter
                label_counter += 1

            for img_name in os.listdir(train_dir):
                img_path = os.path.join(train_dir, img_name)
                image = Image.open(img_path)
                image = image.resize(img_size)
                images.append(np.array(image))

                # Assuming labels are also images with the same name
                label_img_path = os.path.join(label_dir, img_name)
                label_image = Image.open(label_img_path)
                label_array = np.array(label_image)
                labels.append(label_map[label_name])  # Assuming label image corresponds to class

    images = np.array(images) / 255.0  # Normalize images
    labels = np.array(labels)
    return images, labels, label_map

# 2. Split the dataset
def split_dataset(images, labels, test_size=0.2):
    return train_test_split(images, labels, test_size=test_size, random_state=42)

# 3. Build a simple CNN model
class CustomCNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 4. Training loop
def train_model(model, train_images, train_labels, test_images, test_labels, epochs=10):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=epochs,
                        validation_data=(test_images, test_labels))

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

# 5. Predict on new images
def predict(model, new_images):
    logits = model(new_images)
    return tf.argmax(logits, axis=1)

# Main function
def main():
    train_path = r'C:\Users\Shreenithi\Downloads\archive\RescueNet\train'
    label_path = r'C:\Users\Shreenithi\Downloads\archive\RescueNet\label'
    
    images, labels, label_map = load_custom_dataset(train_path, label_path, img_size=(32, 32))
    train_images, test_images, train_labels, test_labels = split_dataset(images, labels)

    # Convert arrays to tensors
    train_images = tf.convert_to_tensor(train_images, dtype=tf.float32)
    test_images = tf.convert_to_tensor(test_images, dtype=tf.float32)
    train_labels = tf.convert_to_tensor(train_labels, dtype=tf.int64)
    test_labels = tf.convert_to_tensor(test_labels, dtype=tf.int64)

    model = CustomCNN(num_classes=len(label_map))
    train_model(model, train_images, train_labels, test_images, test_labels)

    # Example of predicting on a few test images
    predicted_labels = predict(model, test_images[:5])
    for i, pred in enumerate(predicted_labels):
        plt.imshow(test_images[i])
        plt.title(f"Predicted: {list(label_map.keys())[pred.numpy()]}, True: {list(label_map.keys())[test_labels[i].numpy()]}")
        plt.show()

if __name__ == "__main__":
    main()
