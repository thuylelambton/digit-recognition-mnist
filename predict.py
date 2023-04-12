import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import pathlib
import math


def load_model():
    return tf.keras.models.load_model("mnist-cnn.hdf5")


def load_images():
    image_dir = pathlib.Path("./data")
    image_paths = list(image_dir.glob("digit-*.png"))
    images = []
    for path in image_paths:
        img = tf.keras.preprocessing.image.load_img(
            path, color_mode="grayscale", target_size=(28, 28)
        )
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img.reshape(28, 28, 1).astype("float32") / 255.0
        images.append(img)
    return np.array(images)


def max_idx(arr):
    cur_max = arr[0]
    res = 0
    for i, num in enumerate(arr):
        if num > cur_max:
            res = i
            cur_max = num
    return res


def predict(images):
    model = load_model()
    return model.predict(images)


def plot_predictions(images, predictions):
    plt.close("all")
    plt.figure(figsize=(12, 3 * len(images) / 5))
    p = [max_idx(p) for p in predictions]
    for idx, img in enumerate(images):
        plt.subplot(math.ceil(len(images) / 5), 5, idx + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"{p[idx]}")
    plt.tight_layout()


if __name__ == "main":
    images = load_images()
    predictions = predict(images)
    plot_predictions(images, predictions)
