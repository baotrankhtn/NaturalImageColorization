import tensorflow as tf
import keras.backend as K
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Dropout, Input, RepeatVector, Reshape, concatenate, Flatten
from keras.layers import Activation, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import Sequence 

# Helper libraries
# import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab, rgb2hsv, hsv2rgb
from skimage.feature import canny
from skimage.io import imsave, imread, imshow
import numpy as np
import os
import random
import math

# PATH_DATASET = 'R:/Projects/AutoColorization/dataset/train/train_small/'
# PATH_DATASET = 'R:/Projects/AutoColorization/dataset/train/train_large/'
PATH_DATASET = 'R:/Projects/AutoColorization/dataset/train/train_huge/'
PATH_SAVED_MODEL = 'R:/Projects/AutoColorization/scratch/saved_models/train/model.{epoch:02d}-{loss:.5f}-{val_loss:.5f}.h5'
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_WIDTH_SMALL = 128
IMG_HEIGHT_SMALL = 128
BATCH_SIZE = 16
EPOCHS = 40
NUM_SAMPLES = 228942#117097 #685 #
SPLIT_TRAIN_VAL = 0.9
T = 1.8 # Constant for adjusting saturation
STEPS_PER_EPOCH_TRAIN = math.ceil(NUM_SAMPLES*SPLIT_TRAIN_VAL/BATCH_SIZE)
STEPS_PER_EPOCH_VAL = math.ceil(NUM_SAMPLES*(1-SPLIT_TRAIN_VAL)/BATCH_SIZE)

# Build model

# Encoder
encoder_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1,))
encoder = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
encoder = Conv2D(128, (3,3), activation='relu', padding='same')(encoder)
encoder = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder)
encoder = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder)
encoder = Conv2D(512, (3,3), activation='relu', padding='same')(encoder)
encoder = Conv2D(256, (3,3), activation='relu', padding='same')(encoder) 


# Classifier
classifier_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1,))
classifier = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(classifier_input)
classifier = Conv2D(128, (3,3), activation='relu', padding='same')(classifier)
classifier = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(classifier)
classifier = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(classifier)
classifier = Conv2D(512, (3,3), activation='relu', padding='same')(classifier)
classifier = Conv2D(256, (3,3), activation='relu', padding='same')(classifier)
classifier = Conv2D(128, (3,3), activation='relu', padding='same')(classifier)
classifier = Conv2D(32, (3,3), activation='relu', padding='same')(classifier)
classifier = Conv2D(8, (3,3), activation='relu', padding='same')(classifier)
classifier = Conv2D(1, (3,3), activation='relu', padding='same')(classifier)
classifier = Flatten()(classifier)
classifier = Dense(1500)(classifier)
classifier = Dense(1000)(classifier)

# Edge detection
# edge_detector_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1,))
# edge_detector = Conv2D(8, (3,3), activation='relu', padding='same', strides=2)(edge_detector_input)
# edge_detector = Conv2D(8, (3,3), activation='relu', padding='same', strides=2)(edge_detector)
# edge_detector = Conv2D(8, (3,3), activation='relu', padding='same', strides=2)(edge_detector)

# Fusion
repeat_classifier = RepeatVector(32 * 32)(classifier) 
repeat_classifier = Reshape(([32, 32, 1000]))(repeat_classifier)
fusion = concatenate([encoder, repeat_classifier], axis=3) #, edge_detector], axis=3) 

# Decoder
colorizer = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion) 
colorizer = Conv2D(128, (3,3), activation='relu', padding='same')(colorizer)
colorizer = UpSampling2D((2, 2))(colorizer)
colorizer = Conv2D(64, (3,3), activation='relu', padding='same')(colorizer)
colorizer = UpSampling2D((2, 2))(colorizer)
colorizer = Conv2D(32, (3,3), activation='relu', padding='same')(colorizer)
colorizer = Conv2D(16, (3,3), activation='relu', padding='same')(colorizer)
colorizer = Conv2D(2, (3, 3), activation='tanh', padding='same')(colorizer)
colorizer = UpSampling2D((2, 2))(colorizer)

model = Model(inputs=[encoder_input, classifier_input], outputs=colorizer) #edge_detector_input], outputs=colorizer)

print(model.summary())

# Prepare data
# X = []
# for filename in os.listdir(PATH_DATASET):
#     X.append(img_to_array(load_img(PATH_DATASET+filename)))
# X = np.array(X, dtype=float)
# X = 1.0/255*X
# print (X.shape)

class GeneratorSequence(Sequence):
    def __init__(self, file_names, batch_size):
        self.file_names = file_names
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.file_names) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_file_names = self.file_names[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        images = [] 
        for file_name in batch_file_names:
            image = img_to_array(load_img(file_name))
            images.append(image)
            # print(file_name)


        # Generate L and ab
        images = np.array(images, dtype=float)
        images = 1.0/255*images
        
        images_lab = rgb2lab(images)
        images_l = images_lab[:, :, :, 0]
        # images_ab = images_lab[:, :, :, 1:] / 128

        # Adjust saturation
        images_hsv = []
        for image in images:
            image_hsv = rgb2hsv(image)
            image_hsv[:, :, 1] = image_hsv[:, :, 1] * T
            images_hsv.append(image_hsv)

        images_ajusted = []
        for image_hsv in images_hsv:
            images_ajusted.append(hsv2rgb(image_hsv))
        
        
        images_lab_adjusted = rgb2lab(images_ajusted)
        images_ab_adjusted = images_lab_adjusted[:, :, :, 1:] / 128

        # Edge detection
        edges = []
        for i in range(len(images)):
            edge = canny(images_l[i, :, :]).astype(int)
            edges.append(edge)

        edges = np.array(edges, dtype=float)
        edges = edges.reshape(edges.shape + (-1,))
        images_l = images_l.reshape(images_l.shape + (-1,))

        return ([images_l, images_l], images_ab_adjusted)#, edges], images_ab_adjusted)

    def on_epoch_end(self):
        random.shuffle(self.file_names)

# aug = ImageDataGenerator(
#     shear_range=0.2,
#     zoom_range=0.2,
#     rotation_range=20,
#     horizontal_flip=True)


# def generate_data(batch_size):
#     # for batch in aug.flow(X, batch_size=batch_size):
#     #     lab_batch = rgb2lab(batch)
#     #     X_batch = lab_batch[:, :, :, 0]
#     #     Y_batch = lab_batch[:, :, :, 1:] / 128
#     #     yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

#     file_names = os.listdir(PATH_DATASET)
#     file_index = 0

#     # Loop indefinitely
#     while True:
#         # Initialize batches of images and labels
#         images = []
#         labels = []
#         while len(images) < batch_size:
#             if file_index == len(file_names):
#                 file_index = 0
#             image = img_to_array(load_img(PATH_DATASET+file_names[file_index]))
#             images.append(image)
#             # print(file_names[file_index])
#             file_index = file_index + 1

#         images = np.array(images, dtype=float)
#         images = 1.0/255*images
#         images_lab = rgb2lab(images)
#         images = images_lab[:, :, :, 0]
#         labels = images_lab[:, :, :, 1:] / 128

#         # Augmentation
#         # (images, labels) = next(aug.flow(np.array(images), labels, batch_size=batch_size))

#         yield (images.reshape(images.shape + (-1,)), labels)

# model.compile(optimizer='rmsprop', loss='mse')
# model.fit_generator(generate_data(BATCH_SIZE),
#     steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
#     epochs=EPOCHS)

# # Save model
# model.save(PATH_SAVED_MODEL)

if __name__ == '__main__':
    # Get files 
    file_names = []
    for r, d, f in os.walk(PATH_DATASET):
        for file in f:
            if '.jpg' in file:
                file_names.append(os.path.join(r, file))

    # Train & valadition generator
    random.shuffle(file_names)
    split = int(SPLIT_TRAIN_VAL*len(file_names))

    print("==> Images: {}".format(len(file_names)))
    print("==> Train images: {}".format(len(file_names[:split])))
    print("==> Validation images: {}".format(len(file_names[split:])))

    data_generator_train = GeneratorSequence(file_names[:split], BATCH_SIZE)
    data_generator_val = GeneratorSequence(file_names[split:], BATCH_SIZE)

    # Train
    model.compile(optimizer='sgd', loss='mse')

    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(PATH_SAVED_MODEL, save_best_only=False, save_weights_only=False, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min')

    model.fit_generator(data_generator_train,
        steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
        epochs=EPOCHS,
        callbacks=[mcp_save, reduce_lr_loss],
        validation_data=data_generator_val,
        validation_steps=STEPS_PER_EPOCH_VAL,
        use_multiprocessing=False,
        workers=6,
        max_queue_size=10)