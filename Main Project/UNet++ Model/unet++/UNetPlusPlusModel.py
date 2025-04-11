import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from real_time_monitoring import RealTimeMonitoring
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate, BatchNormalization, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

#initilizes the unet++ model
class UNetPlusPlus:

    def __init__(self, input_size=(96, 96, 3), learning_rate=0.00005):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.model = self._build_model()

    @tf.keras.utils.register_keras_serializable()
    @staticmethod
    # dice loss function used in segmentation
    def dice_loss(y_true, y_pred):
        smooth = 1.0
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return 1 - (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    @tf.keras.utils.register_keras_serializable()
    @staticmethod
    # custom combined binary cross-entropy with dice loss
    def combined_dice_bce_loss(y_true, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
        dice_loss = UNetPlusPlus.dice_loss(y_true, y_pred)
        return bce + dice_loss

    # Defines the UNet++ architecture
    def _build_model(self):
        inputs = Input(self.input_size)

        # add gaussina noise
        x = GaussianNoise(0.05)(inputs)

        # Encoder with convolutional layers with batch normalisation and dropout
        c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        c1 = BatchNormalization()(c1)
        c1 = Dropout(0.5)(c1)
        c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
        c2 = BatchNormalization()(c2)
        c2 = Dropout(0.5)(c2)
        c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        # Bottleneck middle of unet++
        c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
        c3 = Dropout(0.5)(c3)
        c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
        # print(f'Shape of c3: {c3.shape}')

        #Decoder which fix issues caused from concatenation mismatch
        u4 = UpSampling2D((2, 2))(c3)
        # print(f'Shape before concatenation: u4 {u4.shape}, c2 {c2.shape}')
        # convert from 128 channel to 64
        u4 = Conv2D(64, (1, 1), activation='relu', padding='same')(u4)
        # print(f'Shape after convertion: u4 {u4.shape}, c2 {c2.shape}')
        m4 = concatenate([u4, c2], axis=-1)
        c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(m4)
        c5 = Dropout(0.5)(c5)
        c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

        u5 = UpSampling2D((2, 2))(c5)
        #print(f'Shape before concatenation: u5 {u5.shape}, c1 {c1.shape}')
        # convert from 64 channel to 32
        u5 = Conv2D(32, (1, 1), activation='relu', padding='same')(u5)
        # print(f'Shape after concatenation: u5 {u5.shape}, c1 {c1.shape}')
        m5 = concatenate([u5, c1], axis = -1)
        c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(m5)
        c6 = Dropout(0.5)(c6)
        c6 = Conv2D(32,(3, 3), activation='relu', padding='same')(c6)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c6)
        model = Model(inputs, outputs)

        # compile the model with combined bce and dice loss function
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.combined_dice_bce_loss, metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, epochs=50, batch_size=8, save_path="best_unetpp_hyperspectral2.keras"):
        '''Train the UNet++ model
        :param x_train: training images
        :param y_train: training mask
        :param epochs: number of epochs
        :param batch_size: training batch size
        :param save_path: path to save the best model
        '''

        # split dataset into training (80%) and validation (20%)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        print(f'Training Data shape: {x_train.shape}, Validation Data shape: {x_val.shape}')

        # defining data augmention pipeline
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.15,
            height_shift_range=0.15,
            zoom_range=0.15,
            brightness_range=[0.8, 1.2],
            horizontal_flip=True,
            fill_mode="nearest"
        )

        datagen.fit(x_train)

        # structure the callbacks fro training
        checkpoint = ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

        # additional real-time monitoring and visual tracking
        real_time_monitoring = RealTimeMonitoring()
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
        print(f'Training model with Data Augmentation and Callbacks saving to: {save_path}')

        # train the model with validation
        history = self.model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            validation_data=(x_val, y_val),
            epochs=epochs,
            steps_per_epoch=len(x_train) // batch_size,
            callbacks=[checkpoint, early_stopping, real_time_monitoring, tensorboard],
            shuffle=True
        )
        # save the trained model
        print(f'Manually saving trained model to: {save_path}')
        self.model.save(save_path)

        print(f'Model training completed and saved at: {save_path}')
        print("Files in current directory:", os.listdir(os.getcwd()))
        return history
    # get the trained model
    def load_trained_model(self, model_path="best_unetpp_hyperspectral2.keras"):
        self.model = load_model(model_path)

    def predict(self, img):
        img_resized = cv2.resize(img, (96, 96)).astype('float32') / 255.0

        #ensure greyscale or hyperspectral conversion to correct dimensions
        if len(img_resized.shape) == 2:
            img_resized = np.expand_dims(img_resized, axis=(0, -1))

        img_resized = np.expand_dims(img_resized, axis=0)

        # get the predicitons
        artery_mask = self.model.predict(img_resized)[0, :, :, 0]

        # normalise correctly
        min_val, max_val = np.min(artery_mask), np.max(artery_mask)
        if max_val - min_val > 1e-6: # prevent division by zero
            artery_mask = (artery_mask - min_val) / (max_val - min_val) #normalise it to 0-1
        else:
            artery_mask = np.zeros_like(artery_mask) # if all values are similiar, make blank

        # convert to 8-bit scale
        artery_mask = (artery_mask * 255).astype(np.uint8)
        # resize it back to match original image size
        artery_mask = cv2.resize(artery_mask, (img.shape[1], img.shape[0]))
        return artery_mask