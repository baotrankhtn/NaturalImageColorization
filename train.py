import tensorflow as tf
import keras.backend as K
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Dropout, Input, RepeatVector, Reshape, concatenate, Flatten
from keras.layers import Activation, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import Sequence 

# Helper libraries
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab, rgb2hsv, hsv2rgb
from skimage.feature import canny
from skimage.io import imsave, imread, imshow
import numpy as np
import os
import random
import configs

# Build model
# Local feature
encoder_input = Input(shape=(configs.IMG_HEIGHT, configs.IMG_WIDTH, 1,))
encoder = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
encoder = Conv2D(128, (3,3), activation='relu', padding='same')(encoder)
encoder = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder)
encoder = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder)
encoder = Conv2D(512, (3,3), activation='relu', padding='same')(encoder)
encoder = Conv2D(256, (3,3), activation='relu', padding='same')(encoder) 


# Classifier
classifier_input = Input(shape=(configs.IMG_HEIGHT, configs.IMG_WIDTH, 1,))
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
edge_detector_input = Input(shape=(configs.IMG_HEIGHT, configs.IMG_WIDTH, 1,))
edge_detector = Conv2D(8, (3,3), activation='relu', padding='same', strides=2)(edge_detector_input)
edge_detector = Conv2D(8, (3,3), activation='relu', padding='same', strides=2)(edge_detector)
edge_detector = Conv2D(8, (3,3), activation='relu', padding='same', strides=2)(edge_detector)

# Fusion
repeat_classifier = RepeatVector(32 * 32)(classifier) 
repeat_classifier = Reshape(([32, 32, 1000]))(repeat_classifier)
fusion = concatenate([encoder, repeat_classifier, edge_detector], axis=3) 

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

model = Model(inputs=[encoder_input, classifier_input, edge_detector_input], outputs=colorizer)

print(model.summary())

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

        # Adjust saturation
        images_hsv = []
        for image in images:
            image_hsv = rgb2hsv(image)
            image_hsv[:, :, 1] = image_hsv[:, :, 1] * configs.T
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

        return ([images_l, images_l, edges], images_ab_adjusted)

    def on_epoch_end(self):
        random.shuffle(self.file_names)

if __name__ == '__main__':
    # Get files 
    file_names = []
    for r, d, f in os.walk(configs.PATH_DATASET):
        for file in f:
            if '.jpg' in file:
                file_names.append(os.path.join(r, file))

    # Train & valadition generator
    random.shuffle(file_names)
    split = int(configs.SPLIT_TRAIN_VAL*len(file_names))

    print("==> Images: {}".format(len(file_names)))
    print("==> Train images: {}".format(len(file_names[:split])))
    print("==> Validation images: {}".format(len(file_names[split:])))

    data_generator_train = GeneratorSequence(file_names[:split], configs.BATCH_SIZE)
    data_generator_val = GeneratorSequence(file_names[split:], configs.BATCH_SIZE)

    # Train
    model.compile(optimizer='sgd', loss='mse')

    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(configs.PATH_SAVED_MODEL, save_best_only=False, save_weights_only=False, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min')

    model.fit_generator(data_generator_train,
        steps_per_epoch=configs.STEPS_PER_EPOCH_TRAIN,
        epochs=configs.EPOCHS,
        callbacks=[mcp_save, reduce_lr_loss],
        validation_data=data_generator_val,
        validation_steps=configs.STEPS_PER_EPOCH_VAL,
        use_multiprocessing=False,
        workers=6,
        max_queue_size=10)