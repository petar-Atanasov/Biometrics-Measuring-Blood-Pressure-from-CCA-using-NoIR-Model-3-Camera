import cv2
import numpy as np

# this class is used to display red overlay channels
class Visualisation:
    def __init__(self, threshold=80):
        self.threshold = threshold

    #overlay red vessels on original image for better visualisation
    def overlay_red_vessels(self, image, artery_mask):
        alpha = 0.6
        # normalising the artery mask to accept the image
        artery_mask = cv2.normalize(artery_mask, None, alpha=0, beta=255, dtype=cv2.NORM_MINMAX)
        # resize the image to original image dimensions
        resized_artery_mask = cv2.resize(artery_mask, (image.shape[1], image.shape[0]))

        #threshold the artery mask to highlight strong detections
        _, strong_vessels = cv2.threshold(resized_artery_mask, 35, 255, cv2.THRESH_BINARY)

        if len(strong_vessels) == 3:
            strong_vessels = cv2.cvtColor(strong_vessels, cv2.COLOR_BGR2GRAY)

        #create red overlay mask (BGR: R=255, G=0, B=0)
        red_overlay = np.zeros_like(image[:, :, :3])
        red_overlay[:, :, 2] = strong_vessels #apply the overlay

        #blend the original image with red overlay
        result = cv2.addWeighted(image[:, :, :3], 1 - alpha, red_overlay, 0.4, 0)
        return result