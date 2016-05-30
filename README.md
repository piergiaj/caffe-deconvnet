# Caffe Deconvolutional Network

Caffe implmentation of a inverse max pooling as described in "Visualizing and understanding convolutional networks" (http://www.matthewzeiler.com/pubs/arxive2013/eccv2014.pdf). Also includes an example of a full deconvolutional network.


## Building

Replace the caffe files with the included in the caffe folder. Most of the files just have the code to setup the layers and are simple modifications of caffe code. The layers added are pooling_switches_layer.cpp modifies max pooling to collect the "switch" variables, the slice_half layer splits the output in half (used to remove the switches from output of max pooling). The inv_pooling layer takes the switches and pooling and reconstructs the input as described the paper. Once the files are added to caffe, build caffe again.

## Usage

There is a python example on using a deconvolutional network with AlexNet. This shows the use of the pooling switches layers and the slicing layer to separate the switches from the pooled data. The invdepoly reconstructs the network using the inv_pooling and deconvolution layers. The python file runs AlexNet with an image, gathers the switches and reconstructs the input.
