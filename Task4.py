import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

(x_train, y_train), (_, _) = cifar10.load_data()
x_train = x_train.astype('float32') / 255

gun_images_dir = "gun_images"
gun_images = []
for filename in os.listdir(gun_images_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(gun_images_dir, filename)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(32, 32))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        gun_images.append(img_array)
gun_images = np.array(gun_images)
gun_labels = np.ones((len(gun_images),), dtype=int)

x_train_combined = np.concatenate([x_train, gun_images])
y_train_combined = np.concatenate([y_train, gun_labels])

indices = np.arange(len(x_train_combined))
np.random.shuffle(indices)
x_train_combined = x_train_combined[indices]
y_train_combined = y_train_combined[indices]

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_combined, y_train_combined, epochs=5, batch_size=32)
