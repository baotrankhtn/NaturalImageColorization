# Introduction

Natural image automatic colorization using deep neural network. The model architecture is based on the model of Satoshi Iizuka, Edgar Simo-Serra and Hiroshi Ishikawa in the paper [Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/en/) with some modifications

# Architecture
There are four component: Local features network (LFN), Classification network (CN), Edge detection network (EDN) and Colorization network. LFN, CN and EDN combine at Fussion layer 

![](https://user-images.githubusercontent.com/18632073/63316953-21db0780-c33b-11e9-9ca3-f6133ae01621.png)

# Loss function
