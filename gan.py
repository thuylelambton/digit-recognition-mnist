# Adapted from https://www.tensorflow.org/tutorials/generative/dcgan
import os
import glob
import imageio
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from IPython import display

try:
    os.rmdir("./images")
except Exception:
    print("")

try:
    os.rmdir("./training_checkpoints")
except:
    print("")


def get_image_directory_path(digit):
    return f"./images/{digit}"


def split_by_label(images, labels):
    d = dict()
    unique_labels = np.unique(labels)
    for label in unique_labels:
        d[label] = []

    for idx, label in enumerate(train_labels):
        d[label].append(train_images[idx])

    for i in unique_labels:
        d[i] = np.array(d[i])

    return d


# load the mnist dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

# split the train images in the mnist dataset by labels
train_images_by_label = split_by_label(train_images, train_labels)
BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 2


def get_train_images(label):
    train_images = train_images_by_label[label]
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(
        "float32"
    )
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    return train_images


def get_train_dataset(label):
    train_images = get_train_images(label)
    # Batch and shuffle the data
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
    )
    return train_dataset


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(
        layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh"
        )
    )
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 1]
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def get_train_checkpoint(
    generator, discriminator, generator_optimizer, discriminator_optimizer, digit
):
    checkpoint_dir = f"./training_checkpoints/{digit}"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )
    return checkpoint, checkpoint_prefix


noise_dim = 100
num_examples_to_generate = 16

# # You will reuse this seed overtime (so it's easier)
# # to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


def get_gan(generator, discriminator, generator_optimizer, discriminator_optimizer):
    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(images):
        seed = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(seed, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )

        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables)
        )
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables)
        )

    return train_step


def generate_and_save_images(model, epoch, test_input, digit):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    image_directory = get_image_directory_path(digit)
    plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")

    plt.savefig("{}/image_at_epoch_{:04d}.png".format(image_directory, epoch))
    plt.show()


def train(digit, epochs):
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    train_step = get_gan(
        generator, discriminator, generator_optimizer, discriminator_optimizer
    )
    dataset = get_train_dataset(digit)
    checkpoint, checkpoint_prefix = get_train_checkpoint(
        generator, discriminator, generator_optimizer, discriminator_optimizer, digit
    )
    os.makedirs(get_image_directory_path(digit), exist_ok=True)

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed, digit)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed, digit)
    return generator


def generate_gif(digit):
    gif_filename = f"dcgan_digit_{digit}.gif"
    image_directory = get_image_directory_path(digit)
    with imageio.get_writer(gif_filename, mode="I") as writer:
        filenames = glob.glob(f"{image_directory}/image*.png")
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


if __name__ == "main":
    digits = [i for i in range(10)]
    for digit in digits:
        model = train(digit, EPOCHS)
        model.save(f"./models/mnist-gan-{digits}.hdf5")
        generate_gif(digit)
