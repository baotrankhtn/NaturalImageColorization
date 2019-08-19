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
import configs

# Load saved model
model = load_model(configs.PATH_TEST_SAVED_MODEL)

# Load images
images = []
file_names = []
# shapes = []

for r, d, f in os.walk(configs.PATH_TEST_DATASET):
        for file in f:
            if '.jpg' in file:
                file_names.append(file)
                images.append(img_to_array(load_img(os.path.join(r, file), target_size=(configs.IMG_WIDTH, configs.IMG_HEIGHT))))
                # img = cv2.imread(os.path.join(r, file))
                # shapes.append(img.shape)

images = np.array(images, dtype=float)
images_l = rgb2lab(1.0/255*images)[:, :, :, 0]
print(images_l.shape)

edges = []
for i in range(len(images)):
    edge = canny(images_l[i, :, :]).astype(int)
    edges.append(edge)

edges = np.array(edges, dtype=float)
edges = edges.reshape(edges.shape + (-1,))
images_l = images_l.reshape(images_l.shape + (-1,))

# Test model
output = model.predict([images_l, images_l, edges])
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((configs.IMG_WIDTH, configs.IMG_HEIGHT, 3))
    cur[:, :, 0] = images_l[i][:, :, 0]
    cur[:, :, 1:] = output[i]
    out = lab2rgb(cur)
    # out = cv2.resize(out, (shapes[i][1], shapes[i][0])) 
    imsave('{}{}'.format(configs.PATH_TEST_DATASET_PREDICTION, file_names[i]), out)

