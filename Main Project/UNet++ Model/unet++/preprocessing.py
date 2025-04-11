import os
import cv2
import numpy as np
import spectral.io.envi as envi

from skimage.filters import frangi

# this class handles all image preprocessing operations
class Preprocessing:

    #extract the hyperspectral images with usefull bands for artery detection
    def load_hyperspectral_image(self, bil_file):
        hdr_file = bil_file + '.hdr'

        #quick check if the file exist
        if not os.path.exists(hdr_file):
            raise FileNotFoundError(f'Header file not found for : {bil_file}')
            exit()

        #try to open ENVI for .bil and .hdr files
        try:
            img = envi.open(hdr_file, bil_file).load()
        except Exception as err:
            raise ValueError(f'Failed to load ENVI file from {bil_file}: {err}')
            exit()

        img = np.array(img)
        #select bands for blood vessel analysis
        band_indices = [115,135,160]
        img_selected = img[:, :, band_indices]

        #normalise back the image
        img_selected = img_selected.astype('float32') / np.max(img_selected)
        img_resized = cv2.resize(img_selected, (96, 96))
        return img_resized #the result is resized image

    #apply max gradient for edge artery detection
    def max_gradient(self, img):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

        #calculate the max gradient magnitute
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        max_value =gradient_magnitude.max()

        if max_value == 0:
            return np.zeros_like(gradient_magnitude, dtype=np.uint8)
        return (gradient_magnitude / max_value * 255).astype(np.uint8)

    #extract the region of interest using GrabCut
    def grabcut_segmentation(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # compute grabcut
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (10, 10, img.shape[1] - 10, img.shape[0] - 10)

        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)

        return img * mask2[:,:,np.newaxis]

    # frangi filtering
    def frangi_filter(self, img):
        frangi_img = frangi(img, beta=0.3, gamma=12, sigmas=[4,6,8,10], black_ridges=False)
        return (frangi_img * 255).astype(np.uint8)

    # apply CLAHE
    def apply_clahe(self,img):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return clahe.apply(img)

    # preprocesses an image using segmentation, gradient and filtering
    def preprocess_image(self, image_path, is_hyperspectral=True):
        if is_hyperspectral:
            img_resized = self.load_hyperspectral_image(image_path)

            #placeholder values for the dataset
            gradient_img = np.zeros_like(img_resized[:, :, 0])
            artery_mask = np.zeros_like(img_resized[:, :, 0])
            return img_resized, gradient_img, artery_mask
        else:
            # turn to grayscale and do a check up in it is read the image
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

            if img is None:
                raise FileNotFoundError(f'Unable to read image: {image_path}')
            #resize back the image to match the segmentation demenstions
            img_resized = cv2.resize(img, (96, 96))
            segmentation_img = self.grabcut_segmentation(img_resized)
            gradient_img = self.max_gradient(segmentation_img)
            artery_mask = self.frangi_filter(gradient_img)
            return img_resized, gradient_img, artery_mask