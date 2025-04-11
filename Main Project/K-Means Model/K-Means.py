import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import frangi
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

#get the directory and sort the files
image_folder = "C:/Users/thega/OneDrive/Desktop/BSc Computer Science Year 3/CST3990 Undergraduate Individual Project/Project Folder/Videos and Photos/RPI 5"
image_files = sorted([img for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png', '.jpeg'))])
# quick check if the files exists
if not image_files:
    print("No files found")
    exit()

# load the images from the directory path
def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

# perform contrast limited adaptive histogram equalization
def clahe_contrast(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(img)

# denoising the image
def reduce_noise(img, h=12):
    return cv2.fastNlMeansDenoising(img, None, h, templateWindowSize=7, searchWindowSize=21)

#extract the vesselness mask with frangi
def extract_features(img):
    vessel_mask = frangi(img, sigmas=[4,6], beta=0.8, black_ridges=False)
    return (vessel_mask * 255).astype(np.uint8)

# k-means segmentation
def segment_image(img, n_clusters):
    img_resized = cv2.resize(img, (64,64), interpolation=cv2.INTER_AREA)
    # flatten the image to match ward linkage labels
    img_flat = img_resized.flatten().reshape(-1, 1)

    #ward linkage labels
    ward_labels = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(img_flat)
    #apply the kmeans
    kmeans_labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=5).fit_predict(ward_labels.reshape(-1, 1))

    #segment the kmeans leables into image
    segmented = kmeans_labels.reshape(img_resized.shape)
    segmented_resize = cv2.resize(segmented, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return (segmented_resize * 255 // n_clusters).astype(np.uint8), kmeans_labels, img_flat

def display(enhanced, artery_mask, segmented):
    cv2.imshow('Enhanced', enhanced)
    cv2.imshow('Frangi Features', artery_mask)
    cv2.imshow('Segmentation', segmented)

# define cluster range to be used in silhouette score plot
cluster_range = range(2,12)
silhouette_scores = {k: [] for k in cluster_range}

total_images = len(image_files)
index = 0
# main functionality
while index < total_images:
    image_path = os.path.join(image_folder, image_files[index])
    print(f'Showing {index + 1}/{total_images}: {image_files[index]}')

    img = load_image(image_path) # load the image
    clahe = clahe_contrast(img) # apply the CLAHe
    denoised = reduce_noise(clahe) # remove the noise
    artery_mask = extract_features(denoised) # apply frangi gilters

    #perform the cluster range over the silhouette score
    for n_clusters in cluster_range:
        segmented, labels, features = segment_image(artery_mask, n_clusters)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(features, labels)
            silhouette_scores[n_clusters].append(score)

    display(denoised, artery_mask, segmented)

    # keyboard execution keys
    key = cv2.waitKey(0) & 0xFF
    if key == ord('n'):
        index = (index + 1) % total_images
    elif key == ord('p'):
        index = (index - 1) % total_images
    elif key == ord('q'):
        break

# compute the average silhouette score
avg_scores = []
for cluster in cluster_range:
    avg = np.mean(silhouette_scores[cluster]) if silhouette_scores[cluster] else 0
    avg_scores.append(avg)

# plot the result from silhouette score
plt.figure(figsize=(10,6))
plt.plot(cluster_range, avg_scores, marker='o')
plt.title('Silhouette Score vs. Number of Clusters (Across 544 Images)')
plt.xlabel('Number of Clusters')
plt.ylabel('Average Silhouette Score')
plt.xticks(cluster_range)
plt.grid(True)
plt.show()
cv2.destroyAllWindows()