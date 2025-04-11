import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import frangi
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#get the directory and sort the files
image_folder = "C:/Users/thega/OneDrive/Desktop/BSc Computer Science Year 3/CST3990 Undergraduate Individual Project/Project Folder/Videos and Photos/RPI 5"
image_files = sorted([img for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png', '.jpeg'))])
# quick check if the files exists
if not image_files:
    print("No files found")
    exit()

# image preprocessing with frangi and clache
def preprocess_with_frangi(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img, (128, 128))
    artery_mask = frangi(img_resize, sigmas=[4,6,8,10], beta=0.8, black_ridges=False)
    artery_mask = (artery_mask * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_mask = clahe.apply(artery_mask)

    return img_resize, enhanced_mask

# Convolutional Autoencoder Model
def build_autoencoder():
    input_img = Input(shape=(128, 128, 1))
    # perform the encoder
    x = Conv2D(32,(3,3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(16,(3,3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2,2), padding='same')(x)

    #decoder
    x = Conv2D(16,(3,3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32,(3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.000005), loss='binary_crossentropy', metrics=['accuracy'])
    return autoencoder

# train or load the model depending if the model is new or alredy existing
def train_or_load_model(x_train, y_train, model_path='trained_autoencoder.keras'):
    # quick check if the model exist if not we create it
    if os.path.exists(model_path):
        autoencoder = load_model(model_path)
        print(f'Loaded saved model from {model_path}')
        return autoencoder

    #if it is not being created, train it
    autoencoder = build_autoencoder()
    # apply early stopping for overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    # models history
    history = autoencoder.fit(
        x_train,
        y_train,
        epochs=100,
        batch_size=8,
        shuffle=True,
        validation_split=0.3,
        callbacks=[early_stopping]
    )
    # save the model after training
    autoencoder.save(model_path)
    print(f'Model trained and saved as {model_path}')

    #plot the training and validation loss after the model is being trained
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return autoencoder

# autoencoder image segmentation
def autoencoder_segmentation(img, model):
    # resize the image to accept the model input
    img_resize = cv2.resize(img, (128, 128))
    img_resize = img_resize.astype('float32') / 255.0
    img_resize = np.expand_dims(img_resize, axis=(0, -1))

    #apply segmentation
    artery_mask = model.predict(img_resize)
    artery_mask = cv2.resize(artery_mask[0, :, :, 0], (img.shape[1], img.shape[0]))
    return (artery_mask * 255).astype(np.uint8)

# apply red channel for overlay
def overlay(img, artery_mask):
    color_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _, strong_vessels = cv2.threshold(artery_mask, 25, 255, cv2.THRESH_BINARY)
    red_overlay = np.zeros_like(color_image)
    red_overlay[:, :, 2] = strong_vessels

    # get the result from the overlay
    results = cv2.addWeighted(color_image, 0.7, red_overlay, 0.8, 0)
    return results

# main exectuion of the file
x_train, y_train = [], []

#loop through out each image and apply preprocessing
for img in image_files:
    img_path = os.path.join(image_folder, img)
    img, artery_mask = preprocess_with_frangi(img_path)

    # apply to the training by downgrading the image size
    x_train.append(img.astype('float32') / 255.0)
    y_train.append(artery_mask.astype('float32') / 255.0)

x_train = np.expand_dims(np.array(x_train), axis=-1)
y_train = np.expand_dims(np.array(y_train), axis=-1)

# load or train the model
autoencoder = train_or_load_model(x_train, y_train)

index = 0
total_images = len(image_files)

while True:
    # get the path and load the image as grayscale
    image_path = os.path.join(image_folder, image_files[index])
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # segment the image
    artery_mask = autoencoder_segmentation(img, autoencoder)
    # display the overlay over it
    overlay_image = overlay(img, artery_mask)

    # print the result
    cv2.imshow('Enhanced', img)
    cv2.imshow('Autoencoder Artery Detection', artery_mask)
    cv2.imshow('Red Overlay', overlay_image)

    print(f'Showing {index + 1}/{total_images}: {image_files[index]}')

    # keyboard execution keys
    key = cv2.waitKey(0) & 0xFF
    if key == ord('n'):
        index = (index + 1) % total_images
    elif key == ord('p'):
        index = (index - 1) % total_images
    elif key == ord('q'):
        break

cv2.destroyAllWindows()