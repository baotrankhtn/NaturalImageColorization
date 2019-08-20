# Introduction

Natural image automatic colorization using deep neural network. The model architecture is based on the model of Satoshi Iizuka, Edgar Simo-Serra and Hiroshi Ishikawa in the paper [Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/en/) with some modifications

Lab colorspace is used. 
Input: L channel
Output: a and b

# Architecture
There are four component: Local features network (LFN), Classification network (CN), Edge detection network (EDN) and Colorization network. LFN, CN and EDN combine at Fussion layer 

![](https://user-images.githubusercontent.com/18632073/63316953-21db0780-c33b-11e9-9ca3-f6133ae01621.png)

# Loss function
To make the output more saturated, we first convert images to HSV then increase S with S' = T.S (T is a constant) then convert back to Lab. With T = 1.8, the model provide good result. Ours loss function is based on MSE, but instead of using Y (a, b from ground truth), we will use K (a, b after adjusting saturation). So the finally loss function:

![](https://user-images.githubusercontent.com/18632073/63317572-8bf4ac00-c33d-11e9-86e1-5210124938af.png)

We have do experiments with T = 1 (no adjustment), T = 1.4, T = 1.6, T = 1.8, T = 2
![](https://user-images.githubusercontent.com/18632073/63317796-5f8d5f80-c33e-11e9-9b75-69b17e79e03e.png)

# Result


# Compare with the model of Richard Zhang and Satoshi Iizuka


# Project struture
