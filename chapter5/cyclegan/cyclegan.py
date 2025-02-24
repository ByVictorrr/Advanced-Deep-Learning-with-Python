"""
MIT License

This example is based on https://github.com/eriklindernoren/Keras-GAN
Copyright (c) 2017 Erik Linder-Norén
Copyright (c) 2019 Ivan Vasilev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import print_function, division

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from data_loader import DataLoader
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers.normalizations import InstanceNormalization

IMG_SIZE = 256


class GeneratorBuilder:
    """Build a model that represents a CycleGAN generator or type of autoencoder U-net like."""

    @classmethod
    def build(cls, img: Input, start_filters=32) -> Model:
        """U-Net Generator
        :param img: input image placeholder
        :param start_filters: number of filters to start downsampling at.
        """
        # build the enncoder
        d1 = cls.downsampling2d(img, start_filters)
        d2 = cls.downsampling2d(d1, start_filters * 2)
        d3 = cls.downsampling2d(d2, start_filters * 4)
        d4 = cls.downsampling2d(d3, start_filters * 8)
        # build the decoder
        u1 = cls.upsamepling2d(d4, d3, start_filters * 4)
        u2 = cls.upsamepling2d(u1, d2, start_filters * 2)
        u3 = cls.upsamepling2d(u2, d1, start_filters)
        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(filters=3, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
        model = Model(img, output_img)
        model.summary()
        return model

    @staticmethod
    def downsampling2d(layer_input, filters: int):
        """Layers used in the encoder"""
        d = Conv2D(filters=filters,
                   kernel_size=4,
                   strides=2,
                   padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        return InstanceNormalization()(d)

    @staticmethod
    def upsamepling2d(layer_input, skip_input, filters: int):
        """Layers used in the decoder.

        :param layer_input: input layer
        :param skip_input: another input from the corresponding level encoder (unet-like)
        :param filters: number of filters
        """
        # UpSampling2D(size=2) doubles the height and width of the feature map.
        # It uses nearest-neighbor interpolation to copy pixel values.
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters=filters, kernel_size=4, strides=1, padding="same", activation="relu")(u)
        u = InstanceNormalization()(u)
        return Concatenate()([u, skip_input])


class DiscriminatorBuilder:
    """Build a model that represents a CycleGAN discriminator."""

    @staticmethod
    def d_layer(layer_input, filters, f_size=4, normalization=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = InstanceNormalization()(d)
        return d

    @classmethod
    def build(cls, img: Input, start_filters=64) -> Model:
        """CNN Discriminator.

        :param img: input image placeholder
        :param start_filters: number of filters to start with for feature extraction.
        """
        d1 = cls.d_layer(img, start_filters, normalization=False)
        d2 = cls.d_layer(d1, start_filters * 2)
        d3 = cls.d_layer(d2, start_filters * 4)
        d4 = cls.d_layer(d3, start_filters * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        model = Model(img, validity)

        model.summary()

        return model

def train(epochs: int,
          data_loader: DataLoader,
          g_XY: Model,
          g_YX: Model,
          d_X: Model,
          d_Y: Model,
          combined: Model,
          batch_size=1,
          sample_interval=50):
    """
    :param epochs: number of steps to train
    :param data_loader: the data loader
    :param g_XY: generator G = X -> Y
    :param g_YX: generator F = Y -> X
    :param d_X: discriminator for X
    :param d_Y: discriminator for Y
    :param combined: the combined generator/discriminator model
    :param batch_size: mini batch size
    :param sample_interval: how often to sample images
    """
    start_time = datetime.datetime.now()
    # calculate the output shape of D (PatchGAN)
    patch = int(IMG_SIZE / 2 ** 4)
    disc_patch = (patch, patch, 1)

    # GAN loss ground truths
    valid = np.ones((batch_size,) + disc_patch)
    fake = np.zeros((batch_size,) + disc_patch)

    for epoch in range(epochs):
        for batch_i, (imgs_X, imgs_Y) in enumerate(data_loader.load_batch(batch_size)):
            # Train the discriminators

            # Translate image to opposite domain
            fake_Y = g_XY.predict(imgs_X)
            fake_X = g_YX.predict(imgs_Y)

            # train the discriminators (original images= real / translated = Fake)
            dX_loss_real = d_X.train_on_batch(imgs_X, valid)
            dX_loss_fake = d_X.train_on_batch(fake_X, fake)
            dX_loss = 0.5 * np.add(dX_loss_real, dX_loss_fake)

            dY_loss_real = d_Y.train_on_batch(imgs_Y, valid)
            dY_loss_fake = d_Y.train_on_batch(fake_Y, fake)
            dY_loss = 0.5 * np.add(dY_loss_real, dY_loss_fake)

            # total discriminator loss
            d_loss = 0.5 * np.add(dX_loss, dY_loss)

            # Train the generators
            g_loss = combined.train_on_batch([imgs_X, imgs_Y],
                                             [valid, valid,
                                              imgs_X, imgs_Y,
                                              imgs_X, imgs_Y])

            elapsed_time = datetime.datetime.now() - start_time

            # Plot the progress
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                  % (epoch, epochs, batch_i, data_loader.n_batches, d_loss[0], 100 * d_loss[1], g_loss[0], np.mean(g_loss[1:3]), np.mean(g_loss[3:5]), np.mean(g_loss[5:6]), elapsed_time))

            # If at save interval => save generated image samples
            if batch_i % sample_interval == 0:
                sample_images(epoch, batch_i, g_XY, g_YX, data_loader)

def sample_images(epoch: int, current_batch: int, g_XY: Model, g_YX: Model, data_loader: DataLoader):
    """Sample new generated images.

    :param epoch: current epoch
    :param current_batch: current batch
    :param g_XY: generator G: X->Y
    :param g_YX: generator F: Y->X
    :param data_loader: data loader
    """
    os.makedirs('images/%s' % data_loader.dataset_name, exist_ok=True)
    r, c = 2, 3

    imgs_X = data_loader.load_data(domain="A", batch_size=1, is_testing=True)
    imgs_Y = data_loader.load_data(domain="B", batch_size=1, is_testing=True)

    # Translate images to the other domain
    fake_Y = g_XY.predict(imgs_X)
    fake_X = g_YX.predict(imgs_Y)

    # Translate back to original domain
    reconstr_X = g_YX.predict(fake_Y)
    reconstr_Y = g_XY.predict(fake_X)

    gen_imgs = np.concatenate([imgs_X, fake_Y, reconstr_X, imgs_Y, fake_X, reconstr_Y])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    titles = ['Original', 'Translated', 'Reconstructed']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[j])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("images/%s/%d_%d.png" % (data_loader.dataset_name, epoch, current_batch))
    plt.close()


if __name__ == "__main__":
    # Input shape
    img_shape = (IMG_SIZE, IMG_SIZE, 3)
    # configure data loader
    data_loader = DataLoader(dataset_name="facades",
                             img_res=(IMG_SIZE, IMG_SIZE))
    # Loss weights
    lambda_cycle = 10.0  # Cycle-consistency loss
    lambda_id = 0.1 * lambda_cycle

    optimizer = Adam(0.0002, 0.5)

    # build and compile the discriminators
    d_X = DiscriminatorBuilder.build(Input(shape=img_shape))
    d_Y = DiscriminatorBuilder.build(Input(shape=img_shape))
    d_X.compile(loss="mse",
                optimizer=optimizer,
                metrics=["accuracy"])
    d_Y.compile(loss="mse",
                optimizer=optimizer,
                metrics=["accuracy"])

    # Build the generators
    img_X = Input(shape=img_shape)
    img_Y = Input(shape=img_shape)

    # Translate images back to other domain
    g_XY = GeneratorBuilder.build(img_X)
    g_YX = GeneratorBuilder.build(img_Y)

    # Translate images to the other domain
    fake_Y = g_XY(img_X)
    fake_X = g_YX(img_Y)

    # Translate images back to original domain
    reconstr_X = g_YX(fake_Y)
    reconstr_Y = g_XY(fake_X)

    # Identity mapping of images
    img_X_id = g_YX(img_X)  # Identity: Domain X should remain unchanged
    img_Y_id = g_XY(img_Y)  # Identity: Domain Y should remain unchanged

    # for the combined model we will only train the generators
    d_X.trainable = False
    d_Y.trainable = False

    # Discriminators determines validity of translated images
    valid_X = d_X(fake_X)
    valid_Y = d_Y(fake_Y)

    # combined model trains both the generators to fool the discriminators
    combined = Model(inputs=[img_X, img_Y],
                     outputs=[valid_X, valid_Y,
                              reconstr_X, reconstr_Y,
                              img_X_id, img_Y_id,
                              ])
    combined.compile(loss=["mse", "mse",
                           "mae", "mae",
                           "mae", "mae"],
                     loss_weights=[1, 1,
                                   lambda_cycle, lambda_cycle,
                                   lambda_id, lambda_id],
                     optimizer=optimizer)

    train(epochs=200, batch_size=1, data_loader=data_loader,
          g_XY=g_XY,
          g_YX=g_YX,
          d_X=d_X,
          d_Y=d_Y,
          combined=combined,
          sample_interval=200)