from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
from skimage import feature
import numpy as np
import os
import configs

# Load black and white images
# def converToGrayscaleAndSave():
for r, d, f in os.walk(configs.PATH_TEST_DATASET):
        for file in f:
            if '.jpg' in file:
                image = (img_to_array(load_img(os.path.join(r, file))))
                imsave('{}{}'.format(configs.PATH_TEST_DATASET_GRAYSCALE, file), rgb2gray(image))