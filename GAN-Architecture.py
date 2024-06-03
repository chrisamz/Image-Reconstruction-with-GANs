# gan_architecture.py

"""
GAN Architecture Module for Image Reconstruction with GANs

This module contains functions for developing and training GAN models for high-quality image reconstruction.

Techniques Used:
- Deep Convolutional GAN (DCGAN)
- Conditional GAN (cGAN)
- Pix2Pix
- CycleGAN

Libraries/Tools:
- tensorflow
- keras
- numpy
- matplotlib

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

class GAN:
    def __init__(self, image_size=(256, 256, 1), latent_dim=100):
        """
        Initialize the GAN class.
        
        :param image_size: tuple, size of the input images (height, width, channels)
        :param latent_dim: int, dimension of the latent space
        """
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        """
        Build the generator model.
        
        :return: compiled generator model
        """
        model = tf.keras.Sequential()
        model.add(layers.Dense(128 * 64 * 64, activation="relu", input_dim=self.latent_dim))
        model.add(layers.Reshape((64, 64, 128)))
        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(128, kernel_size=3, padding="same"))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Activation("relu"))
        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(64, kernel_size=3, padding="same"))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Activation("relu"))
        model.add(layers.Conv2D(self.image_size[2], kernel_size=3, padding="same"))
        model.add(layers.Activation("tanh"))
        model.summary()
        return model

    def build_discriminator(self):
        """
        Build the discriminator model.
        
        :return: compiled discriminator model
        """
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, kernel_size=3, strides=2, input_shape=self.image_size, padding="same"))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(512, kernel_size=3, strides=2, padding="same"))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        model.summary()
        return model

    def build_gan(self):
        """
        Build the combined GAN model.
        
        :return: compiled GAN model
        """
        self.discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.discriminator.trainable = False
        
        model = tf.keras.Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        model.compile(loss="binary_crossentropy", optimizer="adam")
        return model

    def train(self, X_train, epochs=10000, batch_size=32, save_interval=1000):
        """
        Train the GAN model.
        
        :param X_train: array, training images
        :param epochs: int, number of training epochs
        :param batch_size: int, size of the training batches
        :param save_interval: int, interval for saving generated image samples
        """
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, valid)

            print(f"{epoch + 1}/{epochs} [D loss: {d_loss[0]} | D accuracy: {d_loss[1] * 100}%] [G loss: {g_loss}]")

            if (epoch + 1) % save_interval == 0:
                self.save_images(epoch + 1)

    def save_images(self, epoch, save_dir='images/'):
        """
        Save generated image samples.
        
        :param epoch: int, current epoch number
        :param save_dir: str, directory to save generated images
        """
        os.makedirs(save_dir, exist_ok=True)
        noise = np.random.normal(0, 1, (25, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(5, 5)
        cnt = 0
        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(save_dir, f"gan_generated_image_epoch_{epoch}.png"))
        plt.close()

    def save_model(self, model_dir='models/'):
        """
        Save the trained generator and discriminator models.
        
        :param model_dir: str, directory to save models
        """
        os.makedirs(model_dir, exist_ok=True)
        self.generator.save(os.path.join(model_dir, 'generator.h5'))
        self.discriminator.save(os.path.join(model_dir, 'discriminator.h5'))
        print(f"Models saved to {model_dir}")

if __name__ == "__main__":
    data_filepath = 'data/processed/processed_data.npy'
    X_train = np.load(data_filepath)
    X_train = np.expand_dims(X_train, axis=-1)

    gan = GAN(image_size=(256, 256, 1), latent_dim=100)
    gan.train(X_train, epochs=10000, batch_size=32, save_interval=1000)
    gan.save_model()
