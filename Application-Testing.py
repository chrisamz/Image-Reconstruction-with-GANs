# application_testing.py

"""
Application Testing Module for Image Reconstruction with GANs

This module contains functions for testing the trained GAN models on real-world image reconstruction tasks.

Use Cases:
- Medical Imaging
- Satellite Image Analysis
- Digital Restoration

Libraries/Tools:
- numpy
- pandas
- tensorflow
- keras
- matplotlib

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from skimage import io

class ApplicationTesting:
    def __init__(self, generator_model_path, test_data_dir, output_dir):
        """
        Initialize the ApplicationTesting class.
        
        :param generator_model_path: str, path to the trained generator model
        :param test_data_dir: str, directory containing test images
        :param output_dir: str, directory to save the reconstructed images
        """
        self.generator = load_model(generator_model_path)
        self.test_data_dir = test_data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_test_images(self):
        """
        Load test images from the specified directory.
        
        :return: np.array, loaded test images
        """
        images = []
        for file_name in os.listdir(self.test_data_dir):
            if file_name.endswith(('jpg', 'jpeg', 'png')):
                img_path = os.path.join(self.test_data_dir, file_name)
                img = io.imread(img_path)
                img = (img / 127.5) - 1.0  # Normalize to [-1, 1]
                images.append(img)
        return np.array(images)

    def reconstruct_images(self, test_images):
        """
        Reconstruct images using the trained generator model.
        
        :param test_images: np.array, test images to reconstruct
        :return: np.array, reconstructed images
        """
        reconstructed_images = self.generator.predict(test_images)
        reconstructed_images = (reconstructed_images + 1) / 2.0  # Rescale to [0, 1]
        return reconstructed_images

    def save_images(self, images, prefix='reconstructed'):
        """
        Save images to the output directory.
        
        :param images: np.array, images to save
        :param prefix: str, prefix for the saved image filenames
        """
        for i, img in enumerate(images):
            img_path = os.path.join(self.output_dir, f"{prefix}_image_{i}.png")
            io.imsave(img_path, img)
        print(f"Images saved to {self.output_dir}")

    def visualize_results(self, original_images, reconstructed_images):
        """
        Visualize original and reconstructed images side by side.
        
        :param original_images: np.array, original test images
        :param reconstructed_images: np.array, reconstructed images
        """
        plt.figure(figsize=(20, 10))
        for i in range(min(10, len(original_images))):
            plt.subplot(2, 10, i + 1)
            plt.imshow((original_images[i] + 1) / 2.0)
            plt.axis('off')
            plt.title('Original')
            plt.subplot(2, 10, i + 11)
            plt.imshow(reconstructed_images[i])
            plt.axis('off')
            plt.title('Reconstructed')
        plt.show()

    def run_application_testing(self):
        """
        Run the full application testing pipeline.
        """
        # Load test images
        test_images = self.load_test_images()

        # Reconstruct images
        reconstructed_images = self.reconstruct_images(test_images)

        # Save reconstructed images
        self.save_images(reconstructed_images)

        # Visualize results
        self.visualize_results(test_images, reconstructed_images)

if __name__ == "__main__":
    generator_model_path = 'models/generator.h5'
    test_data_dir = 'data/test/'
    output_dir = 'results/reconstructed_images/'

    app_testing = ApplicationTesting(generator_model_path, test_data_dir, output_dir)
    app_testing.run_application_testing()
    print("Application testing completed and results saved.")
