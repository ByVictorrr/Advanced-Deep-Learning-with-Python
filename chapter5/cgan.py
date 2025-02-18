import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (
    BatchNormalization, Input, Dense, Reshape,
    Flatten, Embedding, multiply, LeakyReLU,
    Conv2DTranspose, Conv2D, Dropout,
)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam


# Implement the `build_generator` function, and we will follow the guidelines that we outlined at the begging of the
# section - upsampling with the tranpose convolutions, batch normalization and LeakyReLU activations. The moddule
# starts with a fuly connected layer to upsample the 1D latent vector, then the vector is upsampled with a series of
# Conv2Dtranspose

def build_generator(z_input: Input, label_input: Input):
    """Build generator for a Conditional GAN (cGAN)

    :param z_input: Latent input (random noise vector)
    :param label_input: Conditional label input (digit 0-9 for MNIST)
    """

    # Extract the size of the latent space (e.g., 100) from the input shape
    latent_dim = z_input.shape[1]  # Should be an integer like 100

    # Define the fully connected generator model
    model = Sequential([
        # Fully connected layer (128 neurons) with no bias, input shape is latent_dim
        Dense(128, use_bias=False, input_shape=(latent_dim,)),
        LeakyReLU(alpha=0.2),  # LeakyReLU helps gradients flow better than ReLU
        BatchNormalization(momentum=0.8),  # Normalize activations for stability

        # Increase feature size (256 neurons)
        Dense(256),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),

        # Increase feature size further (512 neurons)
        Dense(512),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),

        # Output layer: Generates 28×28 pixels for MNIST, scaled between [-1,1] using tanh
        Dense(28 * 28, activation="tanh"),

        # Reshape the output from a flat 784 vector to (28,28,1) grayscale image
        Reshape((28, 28, 1)),
    ])

    # Print model summary for debugging
    model.summary()

    # 🔹 Handling the Conditional Label Input 🔹

    # Convert the label (0-9) into an embedding vector of size latent_dim
    # This helps the generator learn relationships between digit classes
    label_embedding = Embedding(input_dim=10, output_dim=latent_dim)(label_input)

    # Flatten the embedding to match the shape of the latent noise vector
    flat_embedding = Flatten()(label_embedding)

    # 🔹 Combine Noise and Label Information 🔹

    # Element-wise multiplication of noise (z_input) and label embedding
    # This forces the generator to use both the random noise and class information
    model_input = multiply([z_input, flat_embedding])

    # Pass the combined input through the generator model to produce an image
    image = model(model_input)

    # Return the final cGAN generator model
    # It takes two inputs: (1) Noise vector, (2) Label input
    # It outputs a (28,28,1) grayscale image
    return Model([z_input, label_input], image)


def build_discriminator():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(128),
        LeakyReLU(alpha=0.2),
        Dense(1, activation="sigmoid")
    ])

    model.summary()

    label_input = Input(shape=(1,), dtype="int32")
    label_embedding = Embedding(input_dim=10, output_dim=28 * 28 * 1)(label_input)
    flat_embedding = Flatten()(label_embedding)

    image = Input(shape=(28, 28, 1))
    flat_img = Flatten()(image)
    # combine the noise and label by element-wise multiplication
    model_input = multiply([flat_img, flat_embedding])
    validity = model(model_input)

    return Model([image, label_input], validity)


def train(generator, discriminator, combined, steps, batch_size):
    """Train the cGAN System.

    :param generator: generator model
    :param discriminator: discriminator model
    :param combined: stacked generator and discriminator we will use the combined network when we train the generator
    :param steps: number of altef the minibatch
    """
    # load the dataset
    (x_train, x_labels), _ = mnist.load_data()
    # rescale in [-1, 1] interval
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=-1)

    # discriminator ground truths
    real, fake = np.ones((batch_size, 1)), np.zeros((batch_size, 1))

    for step in range(steps):
        # Setup to Train the discriminator

        # Select a random batch of images
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_images, labels = x_train[idx], x_labels[idx]

        # Random batch of noise
        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Generate a batch of new images
        generated_images = generator.predict([noise, labels])

        # Train the discriminator
        discriminator.trainable = True
        discriminator_real_loss = discriminator.train_on_batch([real_images, labels], real)
        discriminator_fake_loss = discriminator.train_on_batch([generated_images, labels], fake)
        discriminator_loss = 0.5 * np.add(discriminator_real_loss, discriminator_fake_loss)

        # Setup train the generator
        # random latent vector z
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        # condition on labels
        sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

        # Train the generator
        # note that we use the "valid" labels for the generated images
        # That's because we try to maximize the discriminator loss
        discriminator.trainable = False  # Ensure it is frozen again
        generator_loss = combined.train_on_batch([noise, sampled_labels], real)
        # Display progress
        if step % 500 == 0:
            print("%d [D loss: %.4f, acc.: %.2f%%] [G loss: %.4f]" %
                  (step, discriminator_loss[0], 100 * discriminator_loss[1], generator_loss))


def plot_generated_images(generator, label: int):
    """
    Display a nxn 2D manifold of digits
    :param generator: the generator
    :param label: generate images of particular label
    """
    n = 10
    digit_size = 28

    # big array containing all images
    figure = np.zeros((digit_size * n, digit_size * n))

    # n*n random latent distributions
    noise = np.random.normal(0, 1, (n * n, latent_dim))
    sampled_labels = np.full(n * n, label, dtype=np.int64).reshape(-1, 1)

    # generate the images
    generated_images = generator.predict([noise, sampled_labels])

    # fill the big array with images
    for i in range(n):
        for j in range(n):
            slice_i = slice(i * digit_size, (i + 1) * digit_size)
            slice_j = slice(j * digit_size, (j + 1) * digit_size)
            figure[slice_i, slice_j] = np.reshape(generated_images[i * n + j], (28, 28))

    # plot the results
    plt.figure(num=label, figsize=(6, 5))
    plt.axis('off')
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
    plt.close()


if __name__ == '__main__':
    print("CGAN for new MNIST images with Keras")

    latent_dim = 64

    # we'll use Adam optimizer
    optimizer = Adam(0.0002, 0.5)

    # Build and compile the discriminator
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

    # Build the generator
    z = Input(shape=(latent_dim,))
    label = Input(shape=(1,))

    generator = build_generator(z, label)

    # Generator input z
    generated_image = generator([z, label])

    # Only train the generator for the combined model
    discriminator.trainable = False

    # The discriminator takes generated image as input and determines validity
    real_or_fake = discriminator([generated_image, label])

    # Stack the generator and discriminator in a combined model
    # Trains the generator to deceive the discriminator
    combined = Model([z, label], real_or_fake)
    combined.compile(loss='binary_crossentropy',
                     optimizer=optimizer)

    # train the GAN system
    train(generator=generator,
          discriminator=discriminator,
          combined=combined,
          # steps=50000,
          steps=5000,
          batch_size=100)

    # display some random generated images
    plot_generated_images(generator, 1)
    plot_generated_images(generator, 2)
    plot_generated_images(generator, 3)
    plot_generated_images(generator, 4)
    plot_generated_images(generator, 5)
    plot_generated_images(generator, 6)
    plot_generated_images(generator, 7)
    plot_generated_images(generator, 8)
    plot_generated_images(generator, 9)
    plot_generated_images(generator, 0)
