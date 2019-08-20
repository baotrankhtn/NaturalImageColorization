# Introduction
Natural image automatic colorization using deep neural network!. This model architecture is based on the model of Satoshi Iizuka, Edgar Simo-Serra and Hiroshi Ishikawa in the paper [Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/en/) with **2 big modifications: Add edge detection network and change loss function**

CIELab colorspace is used. 

Input: L channel

Output: a and b

Framework: Keras

About 200,000 natural images from Places2 dataset are used to train the network and about 20,000 to validate

# Architecture
There are four component: Local features network (LFN), Classification network (CN), Edge detection network (EDN) and Colorization network. LFN, CN and EDN combine at Fussion layer 

<img src="https://user-images.githubusercontent.com/18632073/63316953-21db0780-c33b-11e9-9ca3-f6133ae01621.png" width="600">

# Loss function
To make the output more saturated, we first convert images to HSV then increase S with S' = T.S (T is a constant) then convert back to Lab. With **T = 1.8**, the model provide good result. Ours loss function is based on MSE, but instead of using Y (a, b from ground truth), we will use K (a, b after adjusting saturation). So the finally loss function:

<img src="https://user-images.githubusercontent.com/18632073/63317572-8bf4ac00-c33d-11e9-86e1-5210124938af.png" width="200">

We have do experiments with T = 1 (no adjustment), T = 1.4, T = 1.6, T = 1.8, T = 2
<img src="https://user-images.githubusercontent.com/18632073/63317796-5f8d5f80-c33e-11e9-9b75-69b17e79e03e.png" width="600">

# Result
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
