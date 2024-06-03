# deployment.py

"""
Deployment Module for Image Reconstruction with GANs

This module contains functions for deploying the trained GAN models for real-time image reconstruction.

Techniques Used:
- Model loading
- Real-time data handling
- Image reconstruction

Libraries/Tools:
- numpy
- tensorflow
- keras
- flask
- scikit-image

"""

import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from skimage import io
from skimage.transform import resize

app = Flask(__name__)

# Load the trained GAN model
generator_model_path = 'models/generator.h5'
generator = load_model(generator_model_path)

def preprocess_image(image_path, image_size=(256, 256)):
    """
    Preprocess the input image.
    
    :param image_path: str, path to the input image
    :param image_size: tuple, desired size for resizing the image
    :return: np.array, preprocessed image
    """
    image = io.imread(image_path)
    image = resize(image, image_size)
    image = (image / 127.5) - 1.0  # Normalize to [-1, 1]
    return np.expand_dims(image, axis=0)

def postprocess_image(image):
    """
    Postprocess the output image.
    
    :param image: np.array, output image from the generator
    :return: np.array, postprocessed image
    """
    image = (image + 1) / 2.0  # Rescale to [0, 1]
    return (image * 255).astype(np.uint8)

@app.route('/reconstruct', methods=['POST'])
def reconstruct():
    """
    API endpoint to reconstruct an image using the trained GAN model.
    
    :return: JSON response with the reconstructed image
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    image_path = os.path.join('uploads', image_file.filename)
    os.makedirs('uploads', exist_ok=True)
    image_file.save(image_path)
    
    # Preprocess the input image
    input_image = preprocess_image(image_path)
    
    # Reconstruct the image using the generator model
    reconstructed_image = generator.predict(input_image)
    reconstructed_image = postprocess_image(reconstructed_image[0])
    
    # Save the reconstructed image
    reconstructed_image_path = os.path.join('results', 'reconstructed_image.png')
    os.makedirs('results', exist_ok=True)
    io.imsave(reconstructed_image_path, reconstructed_image)
    
    return jsonify({'reconstructed_image_path': reconstructed_image_path})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
