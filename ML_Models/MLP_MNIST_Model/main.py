import matplotlib
matplotlib.use('Agg')
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import keras
from keras import layers


mnist = tf.keras.datasets.mnist
(x_train , y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


model = keras.Sequential([
    layers.Input(shape=(28, 28), name="inputs"),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax'),
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train.astype('float32'), y_train.astype('int32'), epochs=3)

model.save("numerical.keras")


model = tf.keras.models.load_model('numerical.keras')

loss, accuracy = model.evaluate(x_train, y_train)
print(f"Accuracy:{accuracy}")
print(f"Loss: {loss}")


img = cv2.imread(r"D:\Repos\COS30018-Intelligent-Systems-Team-Project\ML_Models\test_images\one_1.png", cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (28, 28))
img = 255 - img
img = np.expand_dims(img, axis=0)

pred = model.predict(img)

plt.imshow(img[0], cmap="gray")
plt.show()
print(f"result: {np.argmax(pred)}")