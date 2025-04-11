import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import Preprocessing
from UNetPlusPlusModel import UNetPlusPlus
from segmentation import Segmentation
from visualisation import Visualisation

#neck dataset folder
# image_folder = "C:/Users/thega/OneDrive/Desktop/BSc Computer Science Year 3/CST3990 Undergraduate Individual Project/Project Folder/Videos and Photos/RPI 5"
# image_files = sorted([img for img in os.listdir(image_folder) if img.endswith(('.jpg', '.jpeg', '.png'))])

image_folder = "C:/Users/thega/OneDrive/Desktop/BSc Computer Science Year 3/CST3990 Undergraduate Individual Project/Project Folder/dataset-10610238/Extracted"
bil_files = sorted([img for img in os.listdir(image_folder) if img.endswith(".bil")])
hdr_files = sorted([img for img in os.listdir(image_folder) if img.endswith(".bil.hdr")])

#quick check if the files existing
if not bil_files or not hdr_files:
    raise ValueError(f"No files found in {image_folder}")
    exit()

# set absolute path for the model
model_path= os.path.abspath("best_unetpp_hyperspectral2.keras")

#initilise preprocessing components
preprocessor = Preprocessing()
is_hyperspectral = True

def resize_with_aspect_ratio(image, width=300):
    height = int(image.shape[0] * (width / image.shape[1]))
    return cv2.resize(image, (width,height))

# if model does not exist train new one
if not os.path.exists(model_path):
    print(f'Model NOT found at: {model_path}, training a new one...')

    #initilise model
    unet = UNetPlusPlus()

    #prepare training data
    x_train, y_train = [],[]

    for bil in bil_files: # hyperspectral
        image_path = os.path.join(image_folder, bil)

        img_resized, _, artery_mask = preprocessor.preprocess_image(image_path, is_hyperspectral=is_hyperspectral)

        # add to the training data
        x_train.append(img_resized)
        y_train.append(artery_mask)

    #used in hyperspectral dataset
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    print(f'Checking x_train shape before processing: {x_train.shape}')

    # the shape has to have 5 channels
    if x_train.shape[-1] == 5:
        x_train = x_train[:, :, :, :3]
    print(f'Final x_train shape: {x_train.shape}')

    print(f'Checking y_train shape before processing: {y_train.shape}')
    if len(y_train.shape) == 3:
        print(f'y_train missing channel dimension, expanding...')
        y_train = np.expand_dims(y_train, axis=-1)

    print(f'Final y_train shape: {y_train.shape}')

    #train the model
    history = unet.train(
        x_train,
        y_train,
        save_path=model_path
    )

    # plot training & validation loss
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'], label='Training loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation loss', color='red')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # check if the model was save
    if os.path.exists(model_path):
        print(f'Model successfully trained and saved at: {model_path}')
    else:
        raise FileNotFoundError(f"Training completed, but model file not found at: {model_path}!")

else:
    print(f'Model found at: {model_path}, skipping training.')

# now attempting to load the model
print('Loading trained model...')
# initilise the rest of the features
segmentor = Segmentation(model_path)
visualiser = Visualisation()

#image processing loop
index = 0
total_images = len(bil_files)
print(f'Found {len(bil_files)} images to process.')

while True:
    #load images
    image_path = os.path.join(image_folder, bil_files[index])
    image = os.path.abspath(image_path) # check for the correct path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image NOT found at: {image_path}")
        exit()

    # preprocess image
    img_resized, gradient_img, artery_mask = preprocessor.preprocess_image(image_path, is_hyperspectral=True)

    if is_hyperspectral:
        img = img_resized[:, :, :3] # use the preprocessed hyperspectral image
    else:
        #convert from grayscale to color
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    #check if there is an image
    if img is None:
        raise FileNotFoundError(f"Unable to read image from main: {image_path}")
        exit()

    # segment the artery using unet++
    artery_mask, artery_mask_display = segmentor.segment_arteries(image_path)

    #resize the artery mask
    resized_artery_mask = cv2.resize(artery_mask, (img.shape[1], img.shape[0]))
    # normalise the artery displayed image
    artery_mask_display = cv2.normalize(artery_mask_display, None, 0, 255, cv2.NORM_MINMAX)

    #apply contrast enhancement
    min_val, max_val = np.percentile(artery_mask_display, [1,99])
    artery_mask_display = np.clip((artery_mask_display - min_val) / (max_val - min_val + 1e-8), 0, 1)
    artery_mask_display = (artery_mask_display * 255).astype(np.uint8)
    artery_mask_display = cv2.convertScaleAbs(artery_mask_display, alpha=5.0, beta=-60)

    # check if it is accepted channel
    if len(artery_mask_display.shape) == 2:
        artery_mask_display = cv2.cvtColor(artery_mask_display, cv2.COLOR_GRAY2BGR)

    # set a colormaps for better visualisation of the artery
    colormaps = [
        cv2.COLORMAP_JET,
        cv2.COLORMAP_HOT,
        cv2.COLORMAP_MAGMA,
        cv2.COLORMAP_PLASMA,
        cv2.COLORMAP_VIRIDIS,
        cv2.COLORMAP_TWILIGHT_SHIFTED
        ]
    selected_colormap = colormaps[np.random.randint(len(colormaps))] # test different colormaps
    # apply it over the segmented artery
    artery_mask_display = cv2.applyColorMap(artery_mask_display, selected_colormap)
    artery_mask_display = cv2.resize(artery_mask_display, (img.shape[1], img.shape[0]))

    #check for proper image dimension
    if len(img.shape) == 2 or img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = img.astype(np.uint8)
    artery_mask_display = artery_mask_display.astype(np.uint8)

    # resize with aspecting ratio
    artery_mask_display = resize_with_aspect_ratio(artery_mask_display)
    cv2.imshow('UNet++', artery_mask_display)

    if is_hyperspectral:
        # display result as RGB
        rgb_display = img_resized[:, :, :3]
        #normalise the image
        rgb_display = cv2.normalize(rgb_display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # then resize it with aspected ratio
        rgb_display = resize_with_aspect_ratio(rgb_display)
        cv2.imshow('Hyperspectral Image', rgb_display)

    print(f'Showing {index+1}/{total_images}: {bil_files[index]}')

    # keyboard control keys for execution
    key = cv2.waitKey(0) & 0xFF
    if key == ord('n'):
        index = (index + 1) % total_images
    elif key == ord('p'):
        index = (index -1) % total_images
    elif key == ord('v'):
        '''Uses 'V' key to visualise the plots'''
        gray_img = cv2.cvtColor((img_resized[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        clahe_img = preprocessor.apply_clahe(gray_img)
        gradient_img = preprocessor.max_gradient(clahe_img)
        frangi_img = preprocessor.frangi_filter(clahe_img)
        grabcut_img = preprocessor.grabcut_segmentation(gray_img)

        # resizing to match model execution
        gray = cv2.resize(gray_img, (96, 96))
        artery_mask_disp = cv2.resize(artery_mask, (96, 96))
        img_display = cv2.resize(rgb_display, (96, 96))
        gradient_display = cv2.resize(gradient_img, (96, 96))
        clahe_display = cv2.resize(clahe_img, (96, 96))
        frangi = cv2.resize(frangi_img, (96, 96))
        grabcut_display = cv2.resize(grabcut_img, (96, 96))
        overlay = visualiser.overlay_red_vessels(img_display, artery_mask_disp) # if not correct change it as a variable artery_mask_display

        #fixing the plot architecture
        fig, axs = plt.subplots(3, 3, figsize=(16, 8))
        axs[0,0].imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
        axs[0,0].set_title('Original Image')
        axs[0,1].imshow(grabcut_display)
        axs[0,1].set_title('GrabCut Image')
        axs[0,2].imshow(gradient_display, cmap='gray')
        axs[0,2].set_title('Gradient Image')
        axs[1,0].imshow(clahe_display, cmap='gray')
        axs[1,1].imshow(gray, cmap='gray')
        axs[1,1].set_title('Grayscale')
        axs[1,2].imshow(frangi, cmap='gray')
        axs[1,2].set_title('Frangi Filter')
        axs[2,0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axs[2,0].set_title('Overlay')
        axs[2,1].imshow(artery_mask_disp, cmap='gray')
        axs[2,1].set_title('UNet++ Artery Mask')

        for ax in axs.ravel():
            ax.axis('off')

        plt.tight_layout()
        plt.show()
        index = (index + 1) % total_images
        continue
    elif key == ord('q'):
        break

cv2.destroyAllWindows()