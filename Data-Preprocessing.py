# data_preprocessing.py

"""
Data Preprocessing Module for Image Reconstruction with GANs

This module contains functions for collecting, cleaning, normalizing, and preparing image data for model training and evaluation.

Techniques Used:
- Data cleaning
- Normalization
- Augmentation
- Resizing

Libraries/Tools:
- pandas
- numpy
- scikit-image
- OpenCV

"""

import os
import cv2
import numpy as np
import pandas as pd
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split

class DataPreprocessing:
    def __init__(self, image_size=(256, 256), augmentation=True):
        """
        Initialize the DataPreprocessing class.
        
        :param image_size: tuple, desired size for resizing images
        :param augmentation: bool, whether to apply data augmentation
        """
        self.image_size = image_size
        self.augmentation = augmentation

    def load_images(self, dir_path):
        """
        Load images from a directory.
        
        :param dir_path: str, path to the directory containing images
        :return: list, loaded images
        """
        images = []
        for file_name in os.listdir(dir_path):
            if file_name.endswith(('jpg', 'jpeg', 'png')):
                img_path = os.path.join(dir_path, file_name)
                img = io.imread(img_path)
                images.append(img)
        return images

    def clean_images(self, images):
        """
        Clean the images by converting to grayscale and resizing.
        
        :param images: list, loaded images
        :return: list, cleaned images
        """
        cleaned_images = []
        for img in images:
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = resize(img, self.image_size)
            cleaned_images.append(img_resized)
        return cleaned_images

    def normalize_images(self, images):
        """
        Normalize the images to the range [0, 1].
        
        :param images: list, cleaned images
        :return: np.array, normalized images
        """
        images_normalized = np.array(images, dtype=np.float32) / 255.0
        return images_normalized

    def augment_images(self, images):
        """
        Apply data augmentation to the images.
        
        :param images: np.array, normalized images
        :return: np.array, augmented images
        """
        if not self.augmentation:
            return images

        augmented_images = []
        for img in images:
            augmented_images.append(img)
            augmented_images.append(np.fliplr(img))
            augmented_images.append(np.flipud(img))
            augmented_images.append(np.rot90(img, k=1))
            augmented_images.append(np.rot90(img, k=3))
        return np.array(augmented_images)

    def save_images(self, images, dir_path):
        """
        Save processed images to a directory.
        
        :param images: np.array, processed images
        :param dir_path: str, path to the directory to save images
        """
        os.makedirs(dir_path, exist_ok=True)
        for i, img in enumerate(images):
            img_path = os.path.join(dir_path, f'image_{i}.png')
            io.imsave(img_path, img)

    def preprocess(self, raw_data_dir, processed_data_dir):
        """
        Execute the full preprocessing pipeline.
        
        :param raw_data_dir: str, path to the directory with raw images
        :param processed_data_dir: str, directory to save processed images
        :return: np.array, preprocessed images
        """
        # Load images
        images = self.load_images(raw_data_dir)

        # Clean images
        images = self.clean_images(images)

        # Normalize images
        images = self.normalize_images(images)

        # Augment images
        images = self.augment_images(images)

        # Save processed images
        self.save_images(images, processed_data_dir)
        print(f"Processed data saved to {processed_data_dir}")

        return images

if __name__ == "__main__":
    raw_data_dir = 'data/raw/'
    processed_data_dir = 'data/processed/'
    image_size = (256, 256)
    augmentation = True

    preprocessing = DataPreprocessing(image_size, augmentation)

    # Preprocess the data
    processed_data = preprocessing.preprocess(raw_data_dir, processed_data_dir)
    print("Data preprocessing completed and data saved.")
