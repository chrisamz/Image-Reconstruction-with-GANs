# model_evaluation.py

"""
Model Evaluation Module for Image Reconstruction with GANs

This module contains functions for evaluating the performance of GAN models for image reconstruction.

Techniques Used:
- Inception Score
- Frechet Inception Distance (FID)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)

Libraries/Tools:
- numpy
- pandas
- tensorflow
- keras
- scikit-image
- scipy

"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

class ModelEvaluation:
    def __init__(self):
        """
        Initialize the ModelEvaluation class.
        """
        self.inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    def calculate_inception_score(self, images, splits=10):
        """
        Calculate the Inception Score (IS) for a set of images.
        
        :param images: np.array, images to evaluate
        :param splits: int, number of splits for evaluation
        :return: float, Inception Score
        """
        processed_images = preprocess_input(images)
        activations = self.inception_model.predict(processed_images)
        scores = []
        for i in range(splits):
            part = activations[i * (len(activations) // splits): (i + 1) * (len(activations) // splits)]
            p_yx = part.mean(axis=0)
            p_y = p_yx.mean(axis=0)
            scores.append(np.exp(np.sum(p_yx * np.log(p_yx / p_y))))
        return np.mean(scores)

    def calculate_fid(self, real_images, generated_images):
        """
        Calculate the Frechet Inception Distance (FID) between real and generated images.
        
        :param real_images: np.array, real images
        :param generated_images: np.array, generated images
        :return: float, FID
        """
        real_images = preprocess_input(real_images)
        generated_images = preprocess_input(generated_images)
        act1 = self.inception_model.predict(real_images)
        act2 = self.inception_model.predict(generated_images)
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def calculate_psnr(self, real_images, generated_images):
        """
        Calculate the Peak Signal-to-Noise Ratio (PSNR) between real and generated images.
        
        :param real_images: np.array, real images
        :param generated_images: np.array, generated images
        :return: float, PSNR
        """
        psnr_values = []
        for real, generated in zip(real_images, generated_images):
            psnr = peak_signal_noise_ratio(real, generated)
            psnr_values.append(psnr)
        return np.mean(psnr_values)

    def calculate_ssim(self, real_images, generated_images):
        """
        Calculate the Structural Similarity Index (SSIM) between real and generated images.
        
        :param real_images: np.array, real images
        :param generated_images: np.array, generated images
        :return: float, SSIM
        """
        ssim_values = []
        for real, generated in zip(real_images, generated_images):
            ssim = structural_similarity(real, generated, multichannel=True)
            ssim_values.append(ssim)
        return np.mean(ssim_values)

    def evaluate(self, real_images, generated_images, output_dir):
        """
        Evaluate and visualize the performance of the GAN model.
        
        :param real_images: np.array, real images
        :param generated_images: np.array, generated images
        :param output_dir: str, directory to save the evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Calculate evaluation metrics
        inception_score = self.calculate_inception_score(generated_images)
        fid = self.calculate_fid(real_images, generated_images)
        psnr = self.calculate_psnr(real_images, generated_images)
        ssim = self.calculate_ssim(real_images, generated_images)

        # Print evaluation metrics
        print(f"Inception Score: {inception_score}")
        print(f"FID: {fid}")
        print(f"PSNR: {psnr}")
        print(f"SSIM: {ssim}")

        # Save evaluation results
        with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
            f.write(f"Inception Score: {inception_score}\n")
            f.write(f"FID: {fid}\n")
            f.write(f"PSNR: {psnr}\n")
            f.write(f"SSIM: {ssim}\n")
        print(f"Evaluation results saved to {os.path.join(output_dir, 'evaluation_results.txt')}")

        # Plot real vs. generated images
        plt.figure(figsize=(10, 5))
        for i in range(10):
            plt.subplot(2, 10, i + 1)
            plt.imshow(real_images[i])
            plt.axis('off')
        for i in range(10):
            plt.subplot(2, 10, i + 11)
            plt.imshow(generated_images[i])
            plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'real_vs_generated.png'))
        plt.show()

if __name__ == "__main__":
    # Example usage
    real_images_filepath = 'data/processed/real_images.npy'
    generated_images_filepath = 'data/processed/generated_images.npy'
    real_images = np.load(real_images_filepath)
    generated_images = np.load(generated_images_filepath)
    output_dir = 'results/evaluation/'

    evaluator = ModelEvaluation()
    evaluator.evaluate(real_images, generated_images, output_dir)
    print("Model evaluation completed and results saved.")
