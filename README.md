# Introduction
Paper: https://link.springer.com/chapter/10.1007%2F978-3-030-41964-6_53

Natural image automatic colorization using deep neural network! This model architecture is based on the model of Satoshi Iizuka, Edgar Simo-Serra and Hiroshi Ishikawa in the paper [Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/en/) with **2 big modifications: Adding edge detection network and changing loss function**

About 200,000 natural images from Places2 dataset are used to train the network and about 20,000 to validate. We use CIELab colorspace, the input is L (grayscale images) and the output is ab

Framework: Keras

# Architecture
There are four components: Local features network (LFN), Classification network (CN), Edge detection network (EDN) and Colorization network. LFN, CN and EDN combine at Fusion layer 

<img src="https://user-images.githubusercontent.com/18632073/63316953-21db0780-c33b-11e9-9ca3-f6133ae01621.png" width="600">

# Loss function
To make the output more saturated, we first convert images to HSV then increase S with S' = T.S (T is a constant) then convert back to Lab. With **T = 1.8**, the model provides good result. Ours loss function is based on MSE, but instead of using Y (a, b from ground truth), we use K (a, b after adjusting saturation). So the final loss function:

<img src="https://user-images.githubusercontent.com/18632073/65568098-bbbd5200-df82-11e9-9349-d32e8bcbcf8e.png" width="200">

We have do experiments with T = 1 (no adjustment), T = 1.4, T = 1.6, T = 1.8, T = 2
<img src="https://user-images.githubusercontent.com/18632073/63317796-5f8d5f80-c33e-11e9-9b75-69b17e79e03e.png" width="600">

# The magic of edge detection network
Edge detection network can prevent colors of different objects in image from mixing
<img src="https://user-images.githubusercontent.com/18632073/63327587-a8eaa880-c358-11e9-96d8-a7c727060436.png" width="400">

# Results
<img src="https://user-images.githubusercontent.com/18632073/63317927-e6423c80-c33e-11e9-8973-558379f31bd2.png" width="600">

# Compare with the models of Richard Zhang and Satoshi Iizuka
<img src="https://user-images.githubusercontent.com/18632073/63325900-3fb56600-c355-11e9-829d-ac5f2ab9094d.png" width="600">

# Project struture
train.py: Define model and training process

test.py: Load model and colorize grayscale images

configs.py: Path to dataset and saved models

utils.py: convert RGB to grayscale,...

# A try on ink wash painting
<img src="https://user-images.githubusercontent.com/18632073/63318188-dc6d0900-c33f-11e9-99e9-9c6c0c0f2ece.png" width="600">

# Conclusion
The model performs well with natural images but there are also many bad cases, too. We train our model on about 200,000 natural images but it can be trained on larger dataset to colorize more objects and achieve better results.
