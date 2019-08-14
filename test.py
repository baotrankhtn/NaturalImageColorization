# Tensorflow
import tensorflow as tf
import keras.backend as K
import os
import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave, imread, imshow
from skimage.feature import canny
import cv2

# Helper libraries
import numpy as np

PATH_DATASET = 'R:/Projects/AutoColorization/dataset/test/input/'
PATH_DATASET_PREDICTION = 'R:/Projects/AutoColorization/dataset/test/no_edge_output/'
PATH_SAVED_MODEL = 'R:/Projects/AutoColorization/scratch/saved_models/release/final.model.T18.noedge.14-0.02690-0.02689.h5' #9 14
IMG_WIDTH = 256
IMG_HEIGHT = 256

# Load saved model
model = load_model(PATH_SAVED_MODEL)
model.compile(optimizer='sgd', loss='mse')

# Load black and white images
images = []
file_names = []
# shapes = []
for filename in os.listdir(PATH_DATASET):
    file_names.append(filename)
    images.append(img_to_array(load_img(PATH_DATASET+filename, target_size=(IMG_WIDTH, IMG_HEIGHT))))

    # img = cv2.imread(PATH_DATASET+filename)
    # shapes.append(img.shape)

images = np.array(images, dtype=float)
images_l = rgb2lab(1.0/255*images)[:, :, :, 0]
print(images_l.shape)

# edges = []
# for i in range(len(images)):
#     edge = canny(images_l[i, :, :]).astype(int)
#     edges.append(edge)

# edges = np.array(edges, dtype=float)
# edges = edges.reshape(edges.shape + (-1,))
images_l = images_l.reshape(images_l.shape + (-1,))

# Test model
output = model.predict([images_l, images_l])#, edges])
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3))
    cur[:, :, 0] = images_l[i][:, :, 0]
    cur[:, :, 1:] = output[i]
    out = lab2rgb(cur)
    # out = cv2.resize(out, (shapes[i][1], shapes[i][0])) 
    imsave('{}{}'.format(PATH_DATASET_PREDICTION, file_names[i]), out)

