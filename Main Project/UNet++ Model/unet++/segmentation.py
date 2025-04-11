import os
import cv2
import numpy as np

from tensorflow.keras.models import load_model
from preprocessing import Preprocessing

class Segmentation:
    # Loade the UNet++ model
    def __init__(self, model_path="best_unetpp_hyperspectral2.keras"):
        #convert to absolute path to prevent path issues
        model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
            print(f'ERROR: Model file NOT found at: {model_path}')
            raise FileNotFoundError(f'Model file NOT found at:{model_path}. Ensure it exist and try again.')

        #load model with custom load function
        self.model = load_model(model_path)
        print(f'Model successfully loaded from: {model_path}')

        self.preprocessing = Preprocessing()

    # apply morphological operations to refine segmentation
    def post_process_mask(self, mask):
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations =2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations =1)
        return mask

    #segment arteries using the UNet++ model
    def segment_arteries(self, img_path, is_hyperspectral=True):
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            raise ValueError(f'Invalid image path: {img_path}')

        img_resized, gradient_img, artery_mask = self.preprocessing.preprocess_image(img_path, is_hyperspectral)

        # resize the image and expand the dimensions to match it
        img_resized = img_resized.astype('float32') / 255.0
        img_resized = (img_resized - np.min(img_resized)) / (np.max(img_resized) - np.min(img_resized) + 1e-8)
        img_resized = np.expand_dims(img_resized, axis=0)

        #check the image shape
        if img_resized.shape[-1] == 5:
            img_resized = img_resized[:, :, :, :3]

        #predict segmentation mask
        artery_mask = self.model.predict(img_resized)[0, :, :, 0]
        # resize it
        artery_mask = cv2.resize(artery_mask, (img_resized.shape[2], img_resized.shape[1]))

        #apply gaussian blur for smoothness
        artery_mask = cv2.GaussianBlur(artery_mask, (9, 9), 0)

        # apply dynamic normalisation with adaptive contrast stretching
        min_val, max_val = np.percentile(artery_mask, [2, 98]) # avoid extreme outliers
        artery_mask = np.clip((artery_mask - min_val) / (max_val - min_val + 1e-8), 0, 1)
        artery_mask = (artery_mask * 255).astype(np.uint8) # scale to 0-255

        # apply CLAHE
        artery_mask = self.preprocessing.apply_clahe(artery_mask)

        # apply thresholding for better visibility
        _, artery_mask_binary = cv2.threshold(artery_mask, 35, 255, cv2.THRESH_BINARY)

        #morphological operations for refining
        artery_mask_binary = self.post_process_mask(artery_mask_binary)

        # use color mapping
        artery_mask_rgb = cv2.applyColorMap(artery_mask_binary, cv2.COLORMAP_JET)

        return artery_mask_binary, artery_mask_rgb