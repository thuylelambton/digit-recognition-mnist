import tensorflow as tf
import numpy as np


def get_mnist_train_test_set():
    # Load the MNIST dataset 
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize the data and expand dimensions
    x_train, x_test = x_train[..., tf.newaxis] / 255.0, x_test[..., tf.newaxis] / 255.0

    # One-hot encode the labels
    y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)

    return (x_train, y_train), (x_test, y_test)


def build_model():
    # Define the CNN model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (5, 5), padding="same", activation="relu", input_shape=(28, 28, 1)
            ),
            tf.keras.layers.Conv2D(32, (5, 5), padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(strides=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08),
        loss="categorical_crossentropy",
        metrics=["acc"],
    )

    return model


def train_model():
    (x_train, y_train), (x_test, y_test) = get_mnist_train_test_set()
    model = build_model()
    model.fit(x_train, y_train, epochs=5)
    return model


if __name__ == "main":
    model = train_model()
    model.save("mnist_model.hdf5")
